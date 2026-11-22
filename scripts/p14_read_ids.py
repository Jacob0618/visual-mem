"""Phase 14: answer from given round lists (ablation controls: model ranking cut at k, cosine top-k of the set size).
Reads the raw rounds with their images exactly as run_mm2 (mm2.read) and judges as run_mm2.
python scripts/p14_read_ids.py --bench memlens --ids results/p14/ids/memlens_rank8.json --out results/p14/runs/memlens/rank8.jsonl"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import mm  # noqa: E402
import mm2  # noqa: E402
from gem import pmap  # noqa: E402
from run_mm import episodes  # noqa: E402
from run_mm2 import usd  # noqa: E402

READERS = {"lite": ("gemini-2.5-flash-lite", 0, "medium"), "strong": ("gemini-3.8-flash", "low", "low")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--ids", required=True, help="JSON {qid: [round ids]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reader", default="lite")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    ids = json.load(open(a.ids))
    split = "32k" if a.bench == "memlens" else "all"
    eps = episodes(a.bench, split, sorted(ids) if a.bench in ("memlens", "visualmem") else None)
    jobs = [(e, qa) for e in eps for qa in e.qas if qa.qid in ids]
    model, thinking, res = READERS[a.reader]
    caps = {"short": {}, "dense": {}}

    def one(job):
        e, qa = job
        r2i = e.rid2idx
        idxs = [r2i[x] for x in ids[qa.qid] if x in r2i]
        try:
            pred, us = mm2.read(e, qa, sorted(set(idxs)), res, caps, model, thinking, tag=f"p14:{Path(a.out).stem}")
            if a.bench == "memlens":
                j = mm2.memlens_judge(qa, pred)
            elif a.bench in ("visualmem", "camroll"):
                from data_visualmem import mc_score
                j = {"score": mc_score(pred, qa.answer), "reason": "exact letter match"}
            else:
                j = mm.judge(qa.question, qa.answer, pred)
            return {"qid": qa.qid, "task": qa.task, "retrieved": [e.rounds[i].rid for i in idxs], "n_loaded": len(idxs),
                    **mm.clue_stats(e, qa, idxs), "gold": qa.answer, "pred": pred, "score": j["score"],
                    "usd": usd(us, model)}
        except Exception as ex:  # noqa: BLE001
            return {"qid": qa.qid, "error": f"{type(ex).__name__}: {str(ex)[:200]}", "score": 0.0}

    done = {}
    if Path(a.out).exists():                    # resume: keep finished records, re-run failed ones
        for l in open(a.out):
            r = json.loads(l)
            if not r.get("error"):
                done[r["qid"]] = r
    R = list(done.values()) + pmap(one, [j for j in jobs if j[1].qid not in done], workers=a.workers, desc=Path(a.out).stem)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in R:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in R if not r.get("error")]
    import numpy as np
    print(json.dumps({"out": a.out, "n": len(R), "errors": len(R) - len(ok), "score": round(float(np.mean([r["score"] for r in ok])), 4),
                      "full": round(float(np.mean([r["full"] for r in ok if r.get("n_gold")])), 3),
                      "rounds": round(float(np.mean([r["n_loaded"] for r in ok])), 1)}))


if __name__ == "__main__":
    main()
