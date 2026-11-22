"""Text-memory verification runs (LoCoMo / LongMemEval).

python scripts/run_text.py --bench locomo --cond cos:10 --cond cos:50 --cond full --cond oracle
python scripts/run_text.py --bench lme --which S --cond cos:10 --cond full
Writes results/text/<bench>[_<which>]/<cond>.jsonl and appends to results/text/summary.jsonl.
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import textmem  # noqa: E402
from data_text import lme_subset_ids, load_lme, load_locomo  # noqa: E402
from gem import pmap, spend  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="locomo")
    ap.add_argument("--which", default="S")
    ap.add_argument("--cond", action="append", required=True)
    ap.add_argument("--reader", default=textmem.READER)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--n_per_type", type=int, default=17)
    args = ap.parse_args()
    if args.bench == "locomo":
        eps, tag = load_locomo(), "locomo"
    else:
        eps, tag = load_lme(args.which, lme_subset_ids(args.n_per_type)), f"lme_{args.which}"
    outdir = ROOT / f"results/text/{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    if any(c.startswith("cos") for c in args.cond):
        pmap(textmem.unit_matrix, eps, workers=4, desc="index")
    for cond in args.cond:
        t0 = time.time()
        jobs = []
        for e in eps:
            for qa in e.qas:
                if cond == "full":
                    ids = list(range(len(e.units)))
                elif cond == "oracle":
                    ids = [i for i, u in enumerate(e.units) if u.uid in set(qa.evidence)] or list(range(len(e.units)))
                elif cond == "oracle_sess":  # all units of the evidence sessions
                    sids = {u.sid for u in e.units if u.uid in set(qa.evidence)}
                    ids = [i for i, u in enumerate(e.units) if u.sid in sids] or list(range(len(e.units)))
                else:
                    ids = textmem.topk(e, qa, int(cond.split(":")[1]))
                jobs.append((e, qa, ids))

        def run(job):
            e, qa, ids = job
            a = textmem.answer(e, qa, ids, "lme" if args.bench == "lme" else "locomo", tag=f"text:{tag}:{cond}",
                               reader=args.reader)
            s = textmem.judge(qa, a["pred"])
            return {"qid": qa.qid, "qtype": qa.qtype, "abs": qa.abstention, "cond": cond, "reader": args.reader,
                    "n_units": len(ids), **textmem.evidence_recall(e, qa, ids), "question": qa.question,
                    "gold": qa.answer, "pred": a["pred"], "score": s, "usage": a["usage"]}

        first = {}
        for i, (e, _, _) in enumerate(jobs):
            first.setdefault(e.name, i)
        fi = list(first.values())
        warm = pmap(run, [jobs[i] for i in fi], workers=args.workers)
        rest_i = [i for i in range(len(jobs)) if i not in set(fi)]
        rest = pmap(run, [jobs[i] for i in rest_i], workers=args.workers, desc=cond)
        recs = [None] * len(jobs)
        for i, r in zip(fi + rest_i, warm + rest):
            recs[i] = r
        rd = "" if args.reader == textmem.READER else f"__{args.reader}"
        with open(outdir / f"{cond.replace(':', '-')}{rd}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by = collections.defaultdict(list)
        for r in recs:
            by["abstention" if r["abs"] else r["qtype"]].append(r["score"])
        cov = [r["cov"] for r in recs if r.get("n_gold")]
        s = {"bench": tag, "cond": cond, "reader": args.reader, "n": len(recs),
             "acc": round(float(np.mean([r["score"] for r in recs])), 4),
             "by": {k: (round(float(np.mean(v)), 3), len(v)) for k, v in sorted(by.items())},
             "ev_cov": round(float(np.mean(cov)), 3) if cov else None,
             "prompt_tok": int(np.mean([r["usage"]["prompt"] for r in recs])),
             "cached_tok": int(np.mean([r["usage"]["cached"] for r in recs])),
             "secs": round(time.time() - t0, 1), "spend_total_usd": spend().get("TOTAL")}
        print(json.dumps(s))
        with open(ROOT / "results/text/summary.jsonl", "a") as f:
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
