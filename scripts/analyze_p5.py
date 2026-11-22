"""Phase-5 analysis (no API calls): EVA and baselines on a MemLens split or on MemEye, by question type, with paired
bootstrap differences and measured cost.  python scripts/analyze_p5.py --bench memlens --split dev
"""
import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import boot_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_dir(d, reader, keep=None):
    D = {}
    for f in glob.glob(str(ROOT / d / f"*__{reader}*.jsonl")):
        parts = os.path.basename(f)[:-6].split("__")
        name = parts[0].replace("-", ":") + "|" + parts[1].replace("-", ":", 1) + "".join(f" +{x}" for x in parts[3:])
        R = {}
        for l in open(f):
            r = json.loads(l)
            if keep is None or r["qid"] in keep:
                R[r["qid"]] = r
        if R:
            D.setdefault(name, {}).update(R)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="memlens")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--reader", default="gemini-2.5-flash-lite")
    ap.add_argument("--ref", default="rag:40|read:medium")
    a = ap.parse_args()
    if a.bench == "memlens":
        keep = set(json.load(open(ROOT / f"results/memlens_{a.split}.json")))
        dirs = ["results/mm2/memlens_32k", f"results/mm2/memlens_32k_memlens_{a.split}"]
        tkey = "y"
    else:
        keep, dirs, tkey = None, ["results/mm2/memeye_all"], "y"
    D = {}
    for d in dirs:
        for k, v in load_dir(d, a.reader, keep).items():
            D.setdefault(k, {}).update(v)
    if a.bench == "memeye":
        sys.path.insert(0, str(ROOT / "scripts"))
        from analyze_p2 import load_all
        D.update({k: v for k, v in load_all(a.reader).items() if k not in D})
    ref = D.get(a.ref)
    types = sorted({r[tkey] for R in D.values() for r in R.values() if r.get(tkey)})
    print(f"\n### {a.bench} {a.split} reader {a.reader} (paired differences vs {a.ref})\n")
    print("| condition | n | acc | Δ vs ref [95% CI] | rounds | relevant units | $/1k | " + " | ".join(t[:10] for t in types) + " |")
    print("|---|---:|---:|---|---:|---:|---:|" + "---:|" * len(types))
    for name in sorted(D, key=lambda n: -np.mean([r["score"] for r in D[n].values()])):
        R = D[name]
        if len(R) < 0.9 * (len(keep) if keep else 371):
            continue
        s = np.mean([r["score"] for r in R.values()])
        if ref and name != a.ref:
            q = sorted(set(R) & set(ref)); d = np.array([R[x]["score"] - ref[x]["score"] for x in q]); lo, hi = boot_ci(d)
            dd = f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
        else:
            dd = "—"
        by = collections.defaultdict(list)
        for r in R.values():
            by[r.get(tkey)].append(r["score"])
        ru = np.mean([r.get("n_relevant_units", np.nan) for r in R.values()])
        print(f"| {name} | {len(R)} | {s:.3f} | {dd} | {np.mean([r.get('n_loaded') or 0 for r in R.values()]):.1f} | "
              f"{ru:.1f} | {1000 * np.mean([r.get('usd', 0) for r in R.values()]):.2f} | "
              + " | ".join(f"{np.mean(by[t]):.2f}" if by[t] else "–" for t in types) + " |")


if __name__ == "__main__":
    main()
