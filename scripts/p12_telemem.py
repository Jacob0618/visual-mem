"""Phase 12: TeleMem (Chen et al., 2026; arXiv 2601.06037; pip telemem 1.10.1), a Mem0-compatible long-term memory
with per-character profiles, semantic deduplication, and buffered fusion, on our benchmarks.
  .venv_bl/bin/python scripts/p12_telemem.py --bench memeye
TeleMem extracts memories from each new message given its recent context (its add_batch pipeline: one window per
message, the user's profile scope plus the shared events scope), fuses near-duplicates in a buffer, and stores them in
FAISS; search returns the most similar memories of both scopes. Its default backbone is a small local model
(Qwen3-8B with Qwen3-8B embeddings); we use gemini-2.5-flash-lite and gemini-embedding-001 through the
OpenAI-compatible Gemini endpoint. Images enter as [Image: caption] markers (TeleMem's multimodal pipeline targets
video). Its extraction prompts are written in Chinese for its Chinese role-play benchmark; we use English
translations with the same instructions. Search returns up to 30 memories per scope; the memories go to our readers.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import contextlib
import json
import os
import shutil
import sqlite3
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p12_common as C  # noqa: E402
import p12_budget as B  # noqa: E402

SYSTEM = "telemem"
LLM_MODEL = "gemini-2.5-flash-lite"
EMB_MODEL = "gemini-embedding-001"
EMB_DIMS = 3072
CONTEXT = 3                 # previous messages given as context to each extraction window
LIMIT = 30                  # memories returned per scope
_lock = threading.Lock()


def patch_embeddings():
    from openai.resources.embeddings import Embeddings
    from openai.types import CreateEmbeddingResponse
    if getattr(Embeddings, "_p12", False):
        return
    orig = Embeddings.create
    db = sqlite3.connect(str(ROOT / "cache/openai_compat.sqlite"), check_same_thread=False, timeout=60)

    def create(self, *args, **kw):
        key = "emb:" + hashlib.sha256(json.dumps({k: kw.get(k) for k in sorted(kw) if k != "timeout"}, sort_keys=True,
                                                 default=str).encode()).hexdigest()
        with _lock:
            row = db.execute("SELECT v FROM r WHERE k=?", (key,)).fetchone()
        if row:
            d = json.loads(row[0])
            for i, x in enumerate(d.get("data", [])):          # rows written before the normalisation
                x["index"] = i if x.get("index") is None else x["index"]
                x.setdefault("object", "embedding")
            d["usage"] = d.get("usage") or {"prompt_tokens": 0, "total_tokens": 0}
            d.setdefault("object", "list")
            return CreateEmbeddingResponse.model_validate(d)
        B.check(SYSTEM)
        kw.setdefault("timeout", 60)          # a dropped TLS handshake otherwise hangs for the client default (600 s)
        for attempt in range(6):
            try:
                resp = orig(self, *args, **kw)
                break
            except Exception:  # noqa: BLE001
                if attempt == 5:
                    raise
                time.sleep(min(60, 2 ** attempt * 3))
        n = (resp.usage.prompt_tokens if resp.usage else 0) or sum(len(str(x)) // 4 for x in (kw.get("input") or []))
        norm = {"object": "list", "model": kw.get("model", ""),   # the Gemini endpoint omits index and usage
                "data": [{"object": "embedding", "index": i, "embedding": d.embedding} for i, d in enumerate(resp.data)],
                "usage": {"prompt_tokens": n, "total_tokens": n}}
        with _lock:
            db.execute("INSERT OR REPLACE INTO r VALUES (?, ?)", (key, json.dumps(norm)))
            db.commit()
        B.charge(SYSTEM, n * 0.15e-6, "build:embeddings")
        return resp

    Embeddings.create = create
    Embeddings._p12 = True


def _recent_messages_prompt_en(parsed_messages, context_messages):
    """English translation of TeleMem's get_recent_messages_prompt (events scope); JSON output for its parser."""
    user_prompt = f"""Below is one turn of a conversation and several turns before it. Summarize this turn in 20 to 100 words.
Keep the summary concise but clear, and cover the key nouns, times, and actions.
Turn to summarize {{
{parsed_messages}
}}
Earlier turns {{
{context_messages}
}}
Output a JSON list with one string: ["<summary>"]"""
    return "You are a helpful assistant.", user_prompt


def _person_prompt_en(parsed_messages, context_messages, target_character):
    """English translation of TeleMem's get_person_prompt (character scope); JSON output for its parser."""
    user_prompt = f"""Below is one new turn of a conversation and several turns before it.
Focus on the character "{target_character}" and extract four kinds of key information from this turn:
1. Relationships and interactions: their relationships, feelings, or concrete actions with others.
2. Events and details: specific events, tasks, decisions, or outcomes they take part in.
3. Traits and background: personality, preferences, abilities, habits, or identity.
4. Specific items and places: items or gifts they own, use, or that appear, and the places they are in.
Summarize the extracted information in one concise sentence that covers the key nouns, actions, and points.
Turn to analyse {{
{parsed_messages}
}}
Earlier turns {{
{context_messages}
}}
Target character: {target_character}
Output a JSON list with one string: ["<summary>"]"""
    return "You are a helpful assistant.", user_prompt


def make_memory(run_dir):
    import telemem.mem0 as tm
    tm.get_recent_messages_prompt = _recent_messages_prompt_en     # its built-in prompts are Chinese
    tm.get_person_prompt = _person_prompt_en
    if not getattr(tm.get_update_memory_prompt, "_p12", False):   # bug in 1.10.1: called with similar_mem_text=,
        _upd = tm.get_update_memory_prompt                         # defined as similary_mem_text
        fixed = lambda new_mem_text, similar_mem_text=None, similary_mem_text=None: _upd(
            new_mem_text, similar_mem_text if similar_mem_text is not None else similary_mem_text)
        fixed._p12 = True
        tm.get_update_memory_prompt = fixed
    from telemem import TeleMemory, TeleMemoryConfig
    key = C.gemini_key()
    cfg = TeleMemoryConfig(**{
        "llm": {"provider": "openai", "config": {"model": LLM_MODEL, "openai_base_url": C.GEMINI_OAI, "api_key": key,
                                                 "temperature": 0.1}},
        "embedder": {"provider": "openai", "config": {"model": EMB_MODEL, "openai_base_url": C.GEMINI_OAI,
                                                      "api_key": key, "embedding_dims": EMB_DIMS}},
        "vector_store": {"provider": "faiss", "config": {"collection_name": "telemem", "path": str(run_dir / "faiss"),
                                                          "embedding_model_dims": EMB_DIMS}},
        "history_db_path": str(run_dir / "history.db"),
    })
    os.environ.pop("OPENROUTER_API_KEY", None)   # mem0 routes every LLM call to OpenRouter when this is set (the MemLens
    return TeleMemory(cfg)                       # judge reads its key from the .env file instead)


def messages_of(ep, caps):
    out = []
    for r in ep.rounds:
        cap = " ".join(f"[Image: {caps.get(p, 'image')}]" for p in r.images)
        u = (r.user + (" " + cap if cap else "")).strip()
        if u:
            out.append({"role": "user", "content": f"[{r.date}] {u}" if r.date else u})
        if r.assistant:
            out.append({"role": "assistant", "content": f"[{r.date}] {r.assistant}" if r.date else r.assistant})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["memeye", "memlens", "visualmem", "camroll"])
    ap.add_argument("--readers", default="lite,strong")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit_eps", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="k/n: this process takes episodes with index % n == k")
    a = ap.parse_args()
    C.patch_openai(SYSTEM)
    patch_embeddings()
    import mm
    k, n = (int(x) for x in a.shard.split("/"))
    eps = [e for i, e in enumerate(C.load_bench(a.bench)[: a.limit_eps or None]) if i % n == k]
    caps = mm.build_captions(eps, "short")
    outp = C.out_path(a.bench, SYSTEM)
    done = C.done_keys(outp)
    readers = a.readers.split(",")
    from concurrent.futures import ThreadPoolExecutor
    for ei, ep in enumerate(eps):
        todo = [q for q in ep.qas if any((q.qid, rd) not in done for rd in readers)]
        if not todo:
            continue
        run_dir = ROOT / "baselines/runs/telemem" / a.bench / ep.name.replace("/", "_")
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True)
        t0 = time.time()
        s0 = B.spent(SYSTEM, "build")
        C.STATE["stage"] = "build"
        msgs = messages_of(ep, caps)
        items = [msgs[max(0, k - CONTEXT): k + 1] for k in range(len(msgs))]
        mem = make_memory(run_dir)
        for s in range(0, len(items), 64):                      # its buffer size; progress is visible per chunk
            mem.add_batch(items[s:s + 64], user_id="user")
        build_usd = B.spent(SYSTEM, "build") - s0
        build_s = time.time() - t0
        C.STATE["stage"] = "query"

        def one(qa):
            recs = []
            try:
                s1 = B.spent(SYSTEM, "query")
                t_r = time.perf_counter()
                res = mem.search(qa.question, user_id="user", limit=LIMIT)
                hits = res.get("results", []) if isinstance(res, dict) else res
                ctx = "\n".join(f"- {h.get('memory', '')}" for h in hits)
                t_ret = time.perf_counter() - t_r
                q_usd = B.spent(SYSTEM, "query") - s1
                for rd in readers:
                    if (qa.qid, rd) in done:
                        continue
                    pred, r_cost = C.read_context(SYSTEM, ep, qa, ctx, [], rd)
                    recs.append({"qid": qa.qid, "reader": rd, "system": SYSTEM, "pred": pred,
                                 "score": C.judge(SYSTEM, a.bench, qa, pred), "n_items": len(hits),
                                 "ctx_chars": len(ctx), "usd_query": q_usd, "usd_read": r_cost,
                                 "usd_build_share": build_usd / max(1, len(ep.qas)),
                                 "usd": q_usd + r_cost + build_usd / max(1, len(ep.qas)), "build_seconds": build_s,
                                 "t_retrieve_s": round(t_ret, 3), "t_read_s": round(getattr(C.TL, "read_s", 0.0), 3),
                                 "warm": qa is todo[0]})
            except B.BudgetExceeded as e:
                recs.append({"qid": qa.qid, "error": f"budget: {e}"})
            except Exception as e:  # noqa: BLE001
                recs.append({"qid": qa.qid, "error": repr(e)[:300]})
            return recs

        with ThreadPoolExecutor(a.workers) as ex, open(outp, "a") as f:
            for recs in ex.map(one, todo):
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                f.flush()
        print(f"  [{ei + 1}/{len(eps)}] {ep.name}: {len(msgs)} messages, build {build_s:.0f}s ${build_usd:.2f}, "
              f"spent ${B.spent(SYSTEM):.2f}", flush=True)


if __name__ == "__main__":
    import faulthandler
    faulthandler.dump_traceback_later(900, repeat=True)     # print all stacks every 15 min (hang diagnosis)
    try:
        main()
    except B.BudgetExceeded as e:
        print("STOPPED:", e, flush=True)
    sys.stdout.flush()
    os._exit(0)      # phase 16: a TeleMem/mem0 worker thread can keep the interpreter spinning after all records are written
