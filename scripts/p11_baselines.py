"""Phase 11: external memory systems as baselines on MemEye, MemLens-32k (test part) and VisualMem.
Run with the baseline environment:  .venv_bl/bin/python scripts/p11_baselines.py --system omni --bench memeye

Protocol (same for every system):
  * memory construction: the system's own pipeline, with gemini-3.8-flash (low thinking) as its LLM through the
    OpenAI-compatible Gemini endpoint; images are described by the same one-sentence captions that our index uses;
  * retrieval: the system's own retrieval for the question (full question text, as given to our selector);
  * reading: the retrieved memory goes to the SAME readers, system prompt and answer format as all our methods
    (lightweight gemini-2.5-flash-lite and strong gemini-3.8-flash); images that the system returns are attached;
  * judging: the official judge of each benchmark, as in run_mm2.
Systems:
  simplemem  SimpleMem (Liu et al., 2026), text memory: dialogue turns with [Image: caption] markers.
  omni       Omni-SimpleMem (Liu et al., 2026), multimodal memory: SimpleMem-style restatements as text units and one
             on-demand image unit per image (caption embedding, raw image loaded at answer time). Following its default
             model roles, per-unit summaries, events and entities use a small model (gemini-2.5-flash-lite).
Every LLM call of a system is cached by request content (cache/openai_compat.sqlite) and logged to cache/usage.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
BL = ROOT / "baselines/SimpleMem"
sys.path.insert(0, str(BL / "OmniSimpleMem"))
GEMINI_OAI = "https://generativelanguage.googleapis.com/v1beta/openai/"
BUILD_MODEL = "gemini-3.8-flash"
PRICES = {"gemini-3.8-flash": (0.75, 3.75), "gemini-2.5-flash-lite": (0.10, 0.40)}
READERS = {"lite": ("gemini-2.5-flash-lite", 0, "medium"), "strong": ("gemini-3.8-flash", "low", "low")}
TAG = {"v": "bl"}            # process-global tag (one episode per process)
_db_lock = threading.Lock()


def _patch_openai():
    """Cache + usage logging + low thinking for gemini-3.x on every OpenAI-compatible chat call."""
    from openai.resources.chat.completions import Completions
    from openai.types.chat import ChatCompletion
    if getattr(Completions, "_p11", False):
        return
    orig = Completions.create
    db = sqlite3.connect(str(ROOT / "cache/openai_compat.sqlite"), check_same_thread=False, timeout=60)
    db.execute("CREATE TABLE IF NOT EXISTS r (k TEXT PRIMARY KEY, v TEXT)")
    db.commit()

    def create(self, *args, **kw):
        kw.pop("stream", None)
        kw.pop("extra_body", None)
        kw.setdefault("timeout", 420)             # long extraction outputs need minutes; the client default is 600 s
        model = kw.get("model", "")
        if model.startswith("gemini-3") and "reasoning_effort" not in kw:
            kw["reasoning_effort"] = "low"
        key = hashlib.sha256(json.dumps({k: kw.get(k) for k in sorted(kw) if k != "timeout"}, sort_keys=True, default=str).encode()).hexdigest()
        with _db_lock:
            row = db.execute("SELECT v FROM r WHERE k=?", (key,)).fetchone()
        if os.getenv("P13_LATENCY") == "1" and TAG.get("v", "").endswith(":query"):
            row = None                            # latency runs: query-time calls are always fresh
        if row:
            resp = ChatCompletion.model_validate_json(row[0])
            _log_usage(model, resp, replay=True)
            return resp
        if os.getenv("P12_LEDGER_DIR"):          # phase 16: fresh calls are charged to a ledger with a cap
            import p12_budget as B
            B.check(TAG["v"].split(":")[1] if ":" in TAG["v"] else "bl")
        kw.setdefault("max_tokens", 8192)        # new requests only (the cache key above excludes it): a degenerate
        for attempt in range(6):                 # generation otherwise runs to the 65k default and times out forever
            try:
                resp = orig(self, *args, **kw)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                time.sleep(min(60, 2 ** attempt * 3))
        with _db_lock:
            db.execute("INSERT OR REPLACE INTO r VALUES (?, ?)", (key, resp.model_dump_json()))
            db.commit()
        _log_usage(model, resp, replay=False)
        return resp

    Completions.create = create
    Completions._p11 = True


def _log_usage(model, resp, replay):
    """Every call (fresh or replayed from the cache) is charged to the method in cache/usage_bl.jsonl; only fresh calls
    are real spend and also go to cache/usage.jsonl."""
    u = resp.usage
    out = (u.completion_tokens or 0) if u else 0
    th = 0
    if u is not None and getattr(u, "completion_tokens_details", None) is not None:
        th = getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0
    rec = {"model": model, "tag": TAG["v"], "prompt": (u.prompt_tokens or 0) if u else 0, "cached": 0, "out": out,
           "thoughts": th, "img_tokens": 0, "t": time.time(), "oai": True, "pid": os.getpid()}
    if not replay and os.getenv("P12_LEDGER_DIR") and ":" in TAG["v"]:
        import p12_budget as B
        B.charge(TAG["v"].split(":")[1], B.usd(model, rec["prompt"], 0, rec["out"], rec["thoughts"]), TAG["v"].split(":")[-1] + ":" + model)
    with _db_lock:
        with open(ROOT / "cache/usage_bl.jsonl", "a") as f:
            f.write(json.dumps({**rec, "replay": replay}) + "\n")
        if not replay:
            with open(ROOT / "cache/usage.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")


def _usd_since(t0, tag_prefix):
    tot = 0.0
    if not (ROOT / "cache/usage_bl.jsonl").exists():
        return 0.0
    for l in open(ROOT / "cache/usage_bl.jsonl"):
        r = json.loads(l)
        if r.get("pid") != os.getpid():
            continue
        if r["t"] >= t0 and r["tag"].startswith(tag_prefix):
            pi, po = PRICES.get(r["model"], (0.75, 3.75))
            tot += r["prompt"] * pi / 1e6 + (r["out"] + r["thoughts"]) * po / 1e6
    return tot


def _iso(date: str) -> str:
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:\D+(\d{1,2}):(\d{2}))?", date or "")
    if not m:
        return date or ""
    y, mo, d, hh, mm_ = m.groups()
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}T{int(hh or 0):02d}:{int(mm_ or 0):02d}:00"


def _turns(ep, caps):
    """(speaker, text, iso timestamp, image paths) per message, with [Image: caption] markers for images."""
    out = []
    for r in ep.rounds:
        imgs = list(r.images)
        cap = " ".join(f"[Image: {caps.get(p, 'image')}]" for p in imgs)
        if r.user or imgs:
            out.append(("User", (r.user + (" " + cap if cap else "")).strip(), _iso(r.date), imgs))
        if r.assistant:
            out.append(("Assistant", r.assistant, _iso(r.date), []))
    return out


# ----------------------------------------------------------------------------------------- systems
class SimpleMemText:
    name = "simplemem"

    def __init__(self, ep, caps, run_dir):
        os.environ.update({"OPENAI_API_KEY": _key(), "OPENAI_BASE_URL": GEMINI_OAI, "LLM_MODEL": BUILD_MODEL,
                           "USE_STREAMING": "false", "MAX_PARALLEL_WORKERS": "4", "MAX_RETRIEVAL_WORKERS": "4"})
        import simplemem.text.system as _sm
        from simplemem.text.system import SimpleMemSystem
        if not getattr(_sm, "_p16_shared_embedder", False):
            # Phase 16: SimpleMemSystem loads a new copy of its embedding model (Qwen3-Embedding-0.6B) for every history
            # and the copies are not released, so a worker that builds many histories runs out of memory. Each worker
            # process now loads the model once and reuses it (same model, same encoding calls).
            _cls, _one = _sm.EmbeddingModel, {}
            _enc = _cls.encode

            def _locked_encode(self, texts, is_query=False):      # thread mode: one encoder call at a time
                with _ST_LOCK:
                    return _enc(self, texts, is_query=is_query)

            _cls.encode = _locked_encode

            def _shared(*a, **k):
                with _INIT:
                    if "m" not in _one:
                        _one["m"] = _cls(*a, **k)
                return _one["m"]

            _sm.EmbeddingModel = _shared
            _sm._p16_shared_embedder = True
        self.s = SimpleMemSystem(api_key=_key(), model=BUILD_MODEL, base_url=GEMINI_OAI, db_path=str(run_dir),
                                 table_name="mem", clear_db=True, use_streaming=False,
                                 enable_parallel_processing=True, max_parallel_workers=4,
                                 enable_parallel_retrieval=True, max_retrieval_workers=4)
        for spk, text, ts, _ in _turns(ep, caps):
            self.s.add_dialogue(spk, text, ts or None)
        self.s.finalize()

    def retrieve(self, question):
        entries = self.s.hybrid_retriever.retrieve(question)
        lines = []
        for e in entries:
            meta = ", ".join(x for x in [e.timestamp or "", e.location or ""] if x)
            lines.append(f"- {('[' + meta + '] ') if meta else ''}{e.lossless_restatement}")
        return "\n".join(lines), [], len(entries)


class OmniSimpleMem:
    name = "omni"

    def __init__(self, ep, caps, run_dir):
        from omni_memory import OmniMemoryOrchestrator, OmniMemoryConfig
        from omni_memory.processors import image_processor as ip
        from benchmarks.locomo.run_locomo import _extract_memory_entries
        from openai import OpenAI
        from omni_memory.utils import embedding as _oe
        if not getattr(_oe.EmbeddingService, "_p16", False):
            # Phase 16: with several episodes in threads, each EmbeddingService loaded its own SentenceTransformer at the
            # same time and torch raised "Cannot copy out of meta tensor"; Omni-SimpleMem then skipped those text units
            # silently. All services now share the model loaded in _init_worker, and an empty embedding is an error.
            _emb_text = _oe.EmbeddingService.embed_text

            def _shared_model(self):
                return TAG["st_model"]

            def _embed_text_strict(self, text):
                with _ST_LOCK:
                    v = _emb_text(self, text)
                if (text or "").strip() and (v is None or len(v) == 0):
                    raise RuntimeError("Omni-SimpleMem: empty text embedding")
                return v

            _emb_batch = _oe.EmbeddingService.embed_texts_batch

            def _embed_batch_strict(self, texts):
                with _ST_LOCK:
                    vs = _emb_batch(self, texts)
                if any((t or "").strip() and (v is None or len(v) == 0) for t, v in zip(texts, vs)):
                    raise RuntimeError("Omni-SimpleMem: empty text embedding in a batch")
                return vs

            _oe.EmbeddingService._get_local_text_model = _shared_model
            _oe.EmbeddingService.embed_text = _embed_text_strict
            _oe.EmbeddingService.embed_texts_batch = _embed_batch_strict
            _oe.EmbeddingService._p16 = True
        orig = ip.ImageProcessor.process
        if not getattr(ip.ImageProcessor, "_p11", False):   # captions are supplied; do not caption a second time
            ip.ImageProcessor.process = lambda self, image, session_id=None, force=False, generate_caption=False, **kw: \
                orig(self, image, session_id=session_id, force=force, generate_caption=False, **kw)
            ip.ImageProcessor._p11 = True
        cfg = OmniMemoryConfig()
        cfg.llm.api_key, cfg.llm.api_base_url = _key(), GEMINI_OAI
        cfg.llm.query_model = cfg.llm.caption_model = BUILD_MODEL
        cfg.llm.summary_model = "gemini-2.5-flash-lite"   # Omni-SimpleMem's default roles: small model for per-unit summaries
        cfg.embedding.model_name, cfg.embedding.embedding_dim = "all-MiniLM-L6-v2", 384
        cfg.entropy_trigger.visual_encoder = "none"
        cfg.retrieval.default_top_k = 20
        cfg.retrieval.enable_hybrid_search = True
        cfg.retrieval.enable_graph_traversal = True
        if hasattr(cfg, "router"):
            cfg.router.router_mode = "heuristic"
        self.o = OmniMemoryOrchestrator(config=cfg, data_dir=str(run_dir))
        # Its query heuristics restrict questions that mention images to VISUAL units, but caption-only image units are
        # stored as text, so such questions would retrieve nothing (its Mem-Gallery adapter routes them to a separate
        # caption search instead). We drop the modality filter and keep the rest of the strategy.
        _strat = self.o.query_processor.determine_retrieval_strategy
        self.o.query_processor.determine_retrieval_strategy = lambda parsed: {**_strat(parsed), "modality_filter": None}
        self.tag = f"ep_{re.sub(r'[^A-Za-z0-9_]', '_', ep.name)}"
        client = OpenAI(api_key=_key(), base_url=GEMINI_OAI)
        turns = _turns(ep, caps)
        prev = []
        for s in range(0, len(turns), 40):                    # the window size of the Omni-SimpleMem LoCoMo runner
            win = [{"speaker": a, "content": b, "timestamp": c} for a, b, c, _ in turns[s:s + 40]]
            ents = _extract_memory_entries(win, client, BUILD_MODEL, prev)
            prev = ents[-10:] if ents else prev
            for stmt in ents:
                self.o.add_text(stmt, tags=[self.tag, "simplemem_text"], force=True)
        for r in ep.rounds:
            for p in r.images:
                self.o.add_image_on_demand_caption_only(p, caps.get(p, ""), tags=[self.tag, "vision_on_demand", "img:" + p],
                                                         force=True)
        if self.o.mau_store.count() == 0 or not self.o.query("the user", top_k=3, tags_filter=[self.tag]).items:
            raise RuntimeError("Omni-SimpleMem memory is empty or not searchable (embedding failure?)")

    def retrieve(self, question):
        from omni_memory.retrieval.pyramid_retriever import ExpansionRequest, RetrievalLevel
        # MAU timestamps record ingestion time in this offline setting, so a date in the question (e.g. MemLens's
        # 'Question date:' line) would become a time filter that matches nothing; we pass a range covering all units.
        res = self.o.query(question, top_k=20, auto_expand=True, tags_filter=[self.tag], time_range=(0.0, 4.1e9))
        if not res.items:
            return "", [], 0
        ids = [it["id"] for it in res.items if "vision_on_demand" in (it.get("tags") or []) and it.get("has_raw_data")]
        if ids:
            exp = {e["id"]: e for e in self.o.retriever.expand(ExpansionRequest(mau_ids=ids, level=RetrievalLevel.EVIDENCE)).items}
            res.items = [exp.get(it["id"], it) for it in res.items]
        ctx = self.o.retriever.format_for_llm(res, include_instructions=False)
        imgs = []
        for it in res.items:
            for t in it.get("tags") or []:
                if t.startswith("img:") and t[4:] not in imgs:
                    imgs.append(t[4:])
        return ctx, imgs[:20], len(res.items)


SYSTEMS = {"simplemem": SimpleMemText, "omni": OmniSimpleMem}


def _key():
    from keys import gemini_key
    return gemini_key()


# ----------------------------------------------------------------------------------------- episode worker
def _read(ep, qa, ctx, imgs, reader):
    import mm
    import mm2
    from gem import Img, generate
    model, thinking, res = READERS[reader]
    parts = [f"User profile: {json.dumps(ep.profile, ensure_ascii=False)[:1500]}\n"] if ep.profile else []
    parts.append(f"\n=== RETRIEVED MEMORY ===\n{ctx or '(no memory retrieved)'}\n")
    for p in imgs:
        parts += ["\n[Image from memory]", Img(p, res)]
    parts.append("\n=== END OF MEMORY ===\n")
    parts = mm._merge(parts + mm2._question_parts(qa) + ["\nAnswer:"])
    r = generate(parts, model=model, system=mm.system_prompt(ep.bench), thinking=thinking, tag=f"{TAG['v']}:read",
                 max_output_tokens=2048, seed=None)
    from analyze_p2 import p1_usd
    return r["text"].strip(), p1_usd({"usage": r["usage"]}, model) if "usage" in r else 0.0


def _judge(bench, qa, pred):
    import mm
    import mm2
    if os.getenv("P13_LATENCY") == "1":     # latency runs: scores are not needed
        return None
    if bench == "memlens":
        return mm2.memlens_judge(qa, pred)["score"]
    if bench in ("visualmem", "camroll"):
        from data_visualmem import mc_score
        return mc_score(pred, qa.answer)
    return mm.judge(qa.question, qa.answer, pred)["score"]


_INIT = threading.Lock()
_ST_LOCK = threading.Lock()


def _init_worker():
    with _INIT:
        if TAG.get("init"):
            return
        import torch                              # keep embedding models on the CPU: MPS copies cost ~3 GB per worker
        from transformers import PreTrainedModel  # noqa: F401  (resolve transformers' lazy import before the patch)
        from sentence_transformers import SentenceTransformer
        torch.backends.mps.is_available = lambda: False
        TAG["st_model"] = SentenceTransformer("all-MiniLM-L6-v2")   # fail loudly here instead of a silent fallback inside a system
        _patch_openai()
        if os.getenv("P13_LATENCY") == "1":
            import gem
            gem.FRESH_TAGS = ["bl:"]             # latency runs: reader calls are always fresh
        TAG["init"] = True


def run_episode(args):
    system, bench, ep_name, qids, readers = args
    _init_worker()
    import mm
    from run_mm import episodes
    split = "32k" if bench == "memlens" else "all"
    eps = episodes(bench, split, [ep_name] if bench == "memeye" else qids)
    ep = [e for e in eps if e.name == ep_name][0]
    ep.qas = [q for q in ep.qas if q.qid in set(qids)]
    caps = mm.build_captions([ep], "short")
    run_dir = ROOT / "baselines/runs" / system / bench / re.sub(r"[^A-Za-z0-9_.-]", "_", ep.name)
    import shutil
    shutil.rmtree(run_dir, ignore_errors=True)   # phase 16: never build on top of an earlier (partial) store
    run_dir.mkdir(parents=True, exist_ok=True)
    TAG["v"] = f"bl:{system}:{bench}:build"
    t0 = time.time()
    try:
        mem = SYSTEMS[system](ep, caps, run_dir)
    except Exception as e:  # noqa: BLE001
        return [{"qid": q.qid, "error": f"build: {e!r}"[:300]} for q in ep.qas]
    build_usd = _usd_since(t0, f"bl:{system}:{bench}:build")
    t_build = time.time() - t0
    out = []
    for k, qa in enumerate(ep.qas):
        TAG["v"] = f"bl:{system}:{bench}:query"
        t1 = time.time()
        try:
            ctx, imgs, n = mem.retrieve(qa.question)
            t_ret = time.time() - t1
            q_usd = _usd_since(t1, f"bl:{system}:{bench}:query")
            for rd in readers:
                t2 = time.time()
                pred, r_usd = _read(ep, qa, ctx, imgs, rd)
                t_read = time.time() - t2
                out.append({"qid": qa.qid, "reader": rd, "system": system, "pred": pred, "score": _judge(bench, qa, pred),
                            "n_items": n, "n_images": len(imgs), "ctx_chars": len(ctx),
                            "usd_query": q_usd, "usd_read": r_usd, "usd_build_share": build_usd / max(1, len(ep.qas)),
                            "usd": q_usd + r_usd + build_usd / max(1, len(ep.qas)), "t_build_s": round(t_build, 2),
                            "t_retrieve_s": round(t_ret, 3), "t_read_s": round(t_read, 3), "warm": k == 0})
        except Exception as e:  # noqa: BLE001
            out.append({"qid": qa.qid, "error": repr(e)[:300]})
    del mem
    import gc
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--bench", required=True, choices=["memeye", "memlens", "visualmem", "camroll"])
    ap.add_argument("--qids_file", default=None)
    ap.add_argument("--readers", default="lite,strong")
    ap.add_argument("--out", default=None, help="output file (default results/baselines/<bench>/<system>.jsonl)")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0,
                    help="run episodes in threads of one process (shared models; construction cost is then charged as the "
                         "run's total divided by its questions)")
    a = ap.parse_args()
    from run_mm import episodes
    if a.bench == "memlens":
        qids = json.load(open(a.qids_file or ROOT / "results/memlens_test.json"))
        eps = episodes("memlens", "32k", qids)
    else:
        eps = episodes(a.bench, "all")
        qids = json.load(open(a.qids_file)) if a.qids_file else [q.qid for e in eps for q in e.qas]
    want = set(qids)
    outp = Path(a.out) if a.out else ROOT / "results/baselines" / a.bench / f"{a.system}.jsonl"
    outp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outp.exists():
        for l in open(outp):
            r = json.loads(l)
            if not r.get("error"):
                done.add((r["qid"], r["reader"]))
    readers = a.readers.split(",")
    jobs = []
    for e in eps:
        todo = [q.qid for q in e.qas if q.qid in want and any((q.qid, rd) not in done for rd in readers)]
        if todo:
            jobs.append((a.system, a.bench, e.name, todo, readers))
    print(f"{a.system} on {a.bench}: {len(jobs)} episodes, {sum(len(j[3]) for j in jobs)} questions to run", flush=True)
    n = 0
    if a.threads:
        # records are written as soon as an episode finishes; construction and query costs of thread-mode runs are
        # recomputed per benchmark from cache/usage_bl.jsonl in the analysis (run totals divided by questions)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(a.threads) as ex, open(outp, "a") as f:
            futs = [ex.submit(run_episode, j) for j in jobs]
            for fut in as_completed(futs):
                for r in fut.result():
                    if not r.get("error"):
                        r["cost_note"] = "thread mode: construction/query cost recomputed per benchmark"
                    f.write(json.dumps(r) + "\n")
                f.flush()
                n += 1
                print(f"  episodes done {n}/{len(jobs)}", flush=True)
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.procs) as pool, open(outp, "a") as f:
        for recs in pool.imap_unordered(run_episode, jobs):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n += 1
            print(f"  episodes done {n}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
