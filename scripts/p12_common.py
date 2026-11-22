"""Phase 12 shared code for the three additional 2026 baselines (V-Mem, MMA, EverMemOS).
Same protocol as phase 11 (scripts/p11_baselines.py): each system builds and searches its memory with its own code;
the retrieved memory goes to our readers, system prompt, and judges. Every fresh model call is charged to the
system's ledger and runs stop at the $100 cap (scripts/p12_budget.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import p12_budget as B  # noqa: E402

GEMINI_OAI = "https://generativelanguage.googleapis.com/v1beta/openai/"
READERS = {"lite": ("gemini-2.5-flash-lite", 0, "medium"), "strong": ("gemini-3.8-flash", "low", "low")}
STATE = {"system": None, "stage": "query"}
LATENCY = os.getenv("P13_LATENCY") == "1"     # phase 13 latency runs: sample questions, sequential, no judging
RETRIEVAL_ONLY = os.getenv("P13_RETRIEVAL_ONLY") == "1"   # phase 13: store each question's retrieved rounds, no reading
TL = threading.local()                          # TL.read_s: wall-clock of the last reader call in this thread
if LATENCY:                                     # latency runs: reader calls are always fresh (never the local cache)
    import gem as _gem
    _gem.FRESH_TAGS = ["bl:"]


def out_path(bench, system):
    base = os.getenv("P16_BASELINE_DIR") or ("results/p13/retrieval" if RETRIEVAL_ONLY else "results/p13/latency" if LATENCY else "results/baselines")
    p = ROOT / base / bench / f"{system}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
_db_lock = threading.Lock()
_seen_errors = set()


def gemini_key():
    from keys import gemini_key as k
    return k()


def patch_openai(system):
    """Cache by request content, log usage, low thinking for gemini-3, charge fresh calls, stop at the cap."""
    from openai.resources.chat.completions import Completions
    from openai.types.chat import ChatCompletion
    STATE["system"] = system
    if getattr(Completions, "_p12", False):
        return
    orig = Completions.create
    db = sqlite3.connect(str(ROOT / "cache/openai_compat.sqlite"), check_same_thread=False, timeout=60)
    db.execute("CREATE TABLE IF NOT EXISTS r (k TEXT PRIMARY KEY, v TEXT)")
    db.commit()

    def create(self, *args, **kw):
        kw.pop("stream", None)
        kw.pop("extra_body", None)
        model = kw.get("model", "")
        if model.startswith("gemini-3") and "reasoning_effort" not in kw:
            kw["reasoning_effort"] = "low"
        key = hashlib.sha256(json.dumps({k: kw.get(k) for k in sorted(kw) if k != "timeout"}, sort_keys=True,
                                        default=str).encode()).hexdigest()
        with _db_lock:
            row = db.execute("SELECT v FROM r WHERE k=?", (key,)).fetchone()
        if row:
            return ChatCompletion.model_validate_json(row[0])
        B.check(STATE["system"])
        kw.setdefault("timeout", 180)
        for attempt in range(6):
            try:
                resp = orig(self, *args, **kw)
                break
            except Exception as e:  # noqa: BLE001
                msg = repr(e)[:300]
                if msg not in _seen_errors:
                    _seen_errors.add(msg)
                    print(f"[p12 openai] {model}: {msg}", file=sys.stderr, flush=True)
                if attempt == 5 or "400" in msg[:60]:
                    raise
                time.sleep(min(60, 2 ** attempt * 3))
        with _db_lock:
            db.execute("INSERT OR REPLACE INTO r VALUES (?, ?)", (key, resp.model_dump_json()))
            db.commit()
        u = resp.usage
        th = 0
        if u is not None and getattr(u, "completion_tokens_details", None) is not None:
            th = getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0
        p, o = ((u.prompt_tokens or 0), (u.completion_tokens or 0)) if u else (0, 0)
        B.charge(STATE["system"], B.usd(model, p, 0, o, th), f"{STATE['stage']}:{model}")
        rec = {"model": model, "tag": f"bl:{STATE['system']}:{STATE['stage']}", "prompt": p, "cached": 0, "out": o,
               "thoughts": th, "img_tokens": 0, "t": time.time(), "oai": True, "pid": os.getpid()}
        with _db_lock, open(ROOT / "cache/usage.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        return resp

    Completions.create = create
    Completions._p12 = True


def load_bench(bench):
    """Episodes of a benchmark restricted to the question sets of Table 2."""
    from run_mm import episodes
    keep = set(json.load(open(os.environ["P13_QIDS"]))) if os.getenv("P13_QIDS") else None
    if bench == "memlens":
        qids = json.load(open(ROOT / os.getenv("P16_MEMLENS_QIDS", "results/p11_memlens_sub150.json")))
        qids = [q for q in qids if keep is None or q in keep]
        eps = episodes("memlens", "32k", qids)
        for e in eps:
            e.qas = [q for q in e.qas if q.qid in set(qids)]
        return _shard(eps)
    eps = episodes(bench, "all")
    if keep is not None:
        for e in eps:
            e.qas = [q for q in e.qas if q.qid in keep]
        eps = [e for e in eps if e.qas]
    return _shard(eps)


def _shard(eps):
    """P16_SHARD=k/n: this process takes the episodes with index % n == k (parallel runs over many histories)."""
    if os.getenv("P16_SHARD"):
        k, n = (int(x) for x in os.environ["P16_SHARD"].split("/"))
        eps = [e for i, e in enumerate(eps) if i % n == k]
    return eps


def read_rounds(system, ep, qa, idxs, reader, caps):
    """Read the selected raw rounds with their images, exactly as RAG@40 and VGS do (mm2.read)."""
    import mm2
    model, thinking, res = READERS[reader]
    B.check(system)
    t = time.perf_counter()
    pred, usages = mm2.read(ep, qa, sorted(set(idxs)), res, caps, model, thinking, tag=f"bl:{system}:read")
    TL.read_s = time.perf_counter() - t
    cost = sum(B.usd(model, u.get("prompt", 0), u.get("cached", 0), u.get("out", 0), u.get("thoughts", 0)) for u in usages)
    B.charge(system, cost, f"read:{reader}")          # charged even when served from the local cache (conservative)
    return pred, cost


def read_context(system, ep, qa, ctx, images, reader):
    """Read a text memory context (plus any images the system returns), as in phase 11."""
    import mm
    import mm2
    from gem import Img, generate
    model, thinking, res = READERS[reader]
    B.check(system)
    parts = [f"User profile: {json.dumps(ep.profile, ensure_ascii=False)[:1500]}\n"] if ep.profile else []
    parts.append(f"\n=== RETRIEVED MEMORY ===\n{ctx or '(no memory retrieved)'}\n")
    for p in images:
        parts += ["\n[Image from memory]", Img(p, res)]
    parts.append("\n=== END OF MEMORY ===\n")
    parts = mm._merge(parts + mm2._question_parts(qa) + ["\nAnswer:"])
    t = time.perf_counter()
    r = generate(parts, model=model, system=mm.system_prompt(ep.bench), thinking=thinking, tag=f"bl:{system}:read",
                 max_output_tokens=2048)
    TL.read_s = time.perf_counter() - t
    return r["text"].strip(), B.charge_gem(system, r, model, f"read:{reader}")


def judge(system, bench, qa, pred):
    import re
    import mm
    import mm2
    from gem import generate
    if LATENCY:
        return None
    if bench == "memlens":                                    # OpenRouter judge (Qwen3-VL), not on the Gemini ledger
        if os.getenv("P12_DEFER_MEMLENS_JUDGE") == "1" or STATE.get("or_dead"):
            return None                                       # judged later by scripts/p12_judge_pending.py
        try:
            return mm2.memlens_judge(qa, pred)["score"]
        except Exception as e:  # noqa: BLE001  (expired key: keep the prediction, judge later)
            if "401" in str(e) or "expired" in str(e).lower():
                STATE["or_dead"] = True
                return None
            raise
    if bench in ("visualmem", "camroll"):
        from data_visualmem import mc_score
        return mc_score(pred, qa.answer)
    pred = pred if len(pred) <= 2000 else pred[:2000] + " …[truncated]"
    p = mm._JUDGE_T.replace("{{question}}", qa.question).replace("{{ground_truth}}", qa.answer).replace("{{model_output}}", pred)
    for attempt in range(3):
        r = generate(p, model=mm.JUDGE, thinking="low", json_schema=mm._JUDGE_SCHEMA, tag=f"bl:{system}:judge",
                     seed=attempt or None)
        B.charge_gem(system, r, mm.JUDGE, "judge")
        try:
            return float(json.loads(r["text"])["score"])
        except Exception:  # noqa: BLE001
            m = re.search(r'"score"\s*:\s*([0-9.]+)', r["text"] or "")
            if m:
                return float(m.group(1))
    return 0.0


def done_keys(path):
    if not Path(path).exists():
        return set()
    out = set()
    for l in open(path):
        r = json.loads(l)
        if not r.get("error"):
            out.add((r["qid"], r["reader"]))
    return out
