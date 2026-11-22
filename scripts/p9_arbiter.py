"""Phase 9: dual-view reading with arbitration. Two stored readings of the same question -- the set view (QSS or VGS)
and the wide view (RAG@40) -- are combined: if their answers agree, the common answer is kept; otherwise a lightweight
arbiter reads a small evidence pool (the QSS set plus the cosine top-k rounds) with both candidate answers and gives the
final answer. Method cost = cost of both readings + arbiter calls (judge excluded, as everywhere).
python scripts/p9_arbiter.py --bench wma --qids results/p8/wma_dev_qids.json \
    --set FILE --wide FILE [--k 10] [--thinking 0] [--tag dev]
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm  # noqa: E402
import mm2  # noqa: E402
from analyze import boot_ci  # noqa: E402
from gem import generate, pmap  # noqa: E402
from run_mm import episodes  # noqa: E402
from run_mm2 import usd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARB_SYS = (
    "You check answers about a user's long-term conversation history with an assistant (text and shared images). You "
    "get some rounds of that history, a question, and two candidate answers produced by two different retrieval views "
    "of the history. One view may have missed evidence that the other found, or may have been distracted by irrelevant "
    "rounds. Using the rounds shown, decide which candidate is correct and complete, or correct it if both are wrong. "
    "If the history shown does not contain the information, say that it is not available. Reply with the final answer "
    "only, in the format the question asks for.")


def L(p):
    R = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("error"):
            raise SystemExit(f"{p}: failed records")
        if "usd" not in r:   # phase-1 records store the raw usage of one lightweight-reader call
            r["usd"] = usd([r["usage"]], "gemini-2.5-flash-lite")
        R[r["qid"]] = r
    return R


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split())


def agree(a, b):
    a, b = norm(a), norm(b)
    return bool(a) and (a == b or (len(a) > 3 and a in b) or (len(b) > 3 and b in a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--qids", default=None)
    ap.add_argument("--set", dest="set_file", required=True)
    ap.add_argument("--wide", required=True)
    ap.add_argument("--k", type=int, default=10, help="cosine top-k rounds added to the QSS set as arbiter evidence")
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--thinking", type=int, default=0)
    ap.add_argument("--res", default="medium")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    qids = json.load(open(a.qids)) if a.qids else None
    eps = episodes(a.bench, a.split if a.bench != "memlens" else "32k", qids if a.bench in ("memlens", "wma") else None)
    if qids:
        keep = set(qids)
        for e in eps:
            e.qas = [q for q in e.qas if q.qid in keep]
        eps = [e for e in eps if e.qas]
    key = "cap:short" if a.bench in ("memlens", "wma") else "cap:dense"
    caps = {s: mm.build_captions(eps, s) for s in (["short"] if a.bench in ("memlens", "wma") else ["short", "dense"])}
    if a.bench in ("memlens", "wma"):
        caps["dense"] = caps["short"]
    idx = {e.name: mm.RoundIndex(e, caps, [key]) for e in eps}
    S, W = L(a.set_file), L(a.wide)

    def one(job):
        e, qa = job
        s, w = S[qa.qid], W[qa.qid]
        base = {"qid": qa.qid, "task": qa.task, "qtype": qa.qtype, "x": qa.x, "y": qa.y, "question": qa.question,
                "gold": qa.answer, "set_score": s["score"], "wide_score": w["score"],
                "usd_views": s["usd"] + w["usd"]}
        if agree(s["pred"], w["pred"]):
            return {**base, "pred": s["pred"], "score": s["score"], "arbitrated": False, "usd": base["usd_views"]}
        r2i = e.rid2idx
        pool = [r2i[x] for x in s["retrieved"] if x in r2i]
        for i in idx[e.name].topk(qa, key, a.k):
            if i not in pool:
                pool.append(i)
        ctx = mm2._strip_q(mm2._context_parts(e, pool, a.res, caps, False))
        cands = [("set", s["pred"]), ("wide", w["pred"])]
        random.Random(qa.qid).shuffle(cands)
        tail = (f"\nQuestion: {qa.question}\n\nCandidate answer 1: {cands[0][1]}\nCandidate answer 2: {cands[1][1]}\n"
                f"\nFinal answer:")
        r = generate(mm._merge(ctx + [tail]), model=a.model, system=ARB_SYS, thinking=a.thinking, tag="p9:arbiter",
                     max_output_tokens=1024)
        pred = r["text"].strip()
        j = mm2.memlens_judge(qa, pred) if a.bench == "memlens" else mm.judge(qa.question, qa.answer, pred)
        if a.bench == "memlens" and not j.get("reason"):
            raise RuntimeError("empty MemLens judge response")
        return {**base, "pred": pred, "score": j["score"], "arbitrated": True, "n_pool": len(pool),
                "usd": base["usd_views"] + usd([r["usage"]], a.model)}

    jobs = [(e, qa) for e in eps for qa in e.qas if qa.qid in S and qa.qid in W]
    R = pmap(one, jobs, workers=8, desc="arbiter")
    out = ROOT / f"results/p9/arb_{a.bench}_{a.tag or 'run'}_k{a.k}_th{a.thinking}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for r in R:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sc = np.array([r["score"] for r in R]); ws = np.array([r["wide_score"] for r in R]); ss = np.array([r["set_score"] for r in R])
    d = sc - ws; lo, hi = boot_ci(d)
    arb = [r for r in R if r["arbitrated"]]
    print(json.dumps({"bench": a.bench, "tag": a.tag, "n": len(R), "wide": round(ws.mean(), 3), "set": round(ss.mean(), 3),
                      "fused": round(sc.mean(), 3), "fused_minus_wide": [round(d.mean(), 3), round(lo, 3), round(hi, 3)],
                      "oracle": round(np.maximum(ws, ss).mean(), 3), "share_arbitrated": round(len(arb) / len(R), 3),
                      "usd_1k_fused": round(1000 * np.mean([r["usd"] for r in R]), 2),
                      "usd_1k_wide": round(1000 * np.mean([W[r["qid"]]["usd"] for r in R]), 2)}))


if __name__ == "__main__":
    main()
