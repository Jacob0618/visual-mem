"""Paired statistics + cost tables for any results directory of per-question jsonl files.

python scripts/analyze.py results/mm/memeye_dev --ref "fc__raw-low" [--group x|y|qtype|task]
- mean score with 95% bootstrap CI (question resampling, 10k reps)
- paired difference vs --ref with 95% bootstrap CI and two-sided sign-flip permutation p
- read cost in USD per 1k questions from logged usage (uncached input, cached input, output+thinking)
"""
import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gem import PRICES  # noqa: E402

P = PRICES["gemini-3.8-flash"]


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        R = [json.loads(l) for l in open(f)]
        out[os.path.basename(f)[:-6]] = {r["qid"]: r for r in R}
    return out


def boot_ci(x, reps=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(reps, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def perm_p(d, reps=10000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    s = rng.choice([-1.0, 1.0], size=(reps, len(d)))
    return float((np.abs((s * d).mean(1)) >= obs - 1e-12).mean())


def usd_per_1k(recs):
    u = [r["usage"] for r in recs]
    cost = [((x["prompt"] - x["cached"]) * P["in"] + x["cached"] * P["cached"] + (x["out"] + x["thoughts"]) * P["out"]) / 1e6
            for x in u]
    return 1000 * float(np.mean(cost))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--group", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--score", default="score")
    args = ap.parse_args()
    D = load(args.dir)
    names = [n for n in D if (args.only is None or n in args.only)]
    qids = sorted(set.intersection(*[set(D[n]) for n in names]))
    ref = D.get(args.ref)
    rows = []
    for n in names:
        R = [D[n][q] for q in qids]
        s = np.array([r[args.score] for r in R], float)
        lo, hi = boot_ci(s)
        row = {"cond": n, "n": len(R), "mean": s.mean(), "ci": (lo, hi),
               "prompt_tok": np.mean([r["usage"]["prompt"] for r in R]),
               "cached_frac": np.sum([r["usage"]["cached"] for r in R]) / max(1, np.sum([r["usage"]["prompt"] for r in R])),
               "usd_1k": usd_per_1k(R)}
        if ref is not None and n != args.ref:
            d = s - np.array([ref[q][args.score] for q in qids], float)
            row["diff"] = d.mean()
            row["diff_ci"] = boot_ci(d)
            row["p"] = perm_p(d)
        if args.group:
            g = collections.defaultdict(list)
            for r in R:
                g[str(r.get(args.group))].append(r[args.score])
            row["groups"] = {k: (round(float(np.mean(v)), 3), len(v)) for k, v in sorted(g.items())}
        rows.append(row)
    rows.sort(key=lambda r: -r["mean"])
    print(f"n common questions = {len(qids)}; ref = {args.ref}")
    print(f"| condition | mean [95% CI] | Δ vs ref [95% CI], p | prompt tok | cached | read $/1k q |")
    print("|---|---|---|---:|---:|---:|")
    for r in rows:
        dd = (f"{r['diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}], p={r['p']:.3g}"
              if "diff" in r else "—")
        print(f"| {r['cond']} | {r['mean']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | {dd} | {r['prompt_tok']:,.0f} | "
              f"{r['cached_frac']:.0%} | {r['usd_1k']:.2f} |")
    if args.group:
        keys = sorted({k for r in rows for k in r["groups"]})
        print("\n| condition | " + " | ".join(f"{k} (n={rows[0]['groups'].get(k, (0, 0))[1]})" for k in keys) + " |")
        print("|---|" + "---:|" * len(keys))
        for r in rows:
            print(f"| {r['cond']} | " + " | ".join(f"{r['groups'].get(k, (float('nan'),))[0]:.3f}" for k in keys) + " |")


if __name__ == "__main__":
    main()
