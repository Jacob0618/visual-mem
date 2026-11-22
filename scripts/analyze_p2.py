"""Phase-2 comparison table: every condition on MemEye-371 for a given reader, with total $ per 1k questions
(all calls incl. selection, map and zoom stages), mean rounds loaded, full-set recall, and paired Δ vs references.

python scripts/analyze_p2.py --reader gemini-2.5-flash-lite --ref "rag:20|read:medium" --ref "fc|read:medium"
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import boot_ci  # noqa: E402
from gem import PRICES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P1_NAME = {"fc__raw-{r}": "fc|read:{r}", "oracle__raw-{r}": "oracle|read:{r}", "cap-dense-{k}__raw-{r}": "rag:{k}|read:{r}",
           "fc__cap-dense": "fc|cap:dense", "oracle__cap-dense": "oracle|cap:dense", "fc__cap-short": "fc|cap:short",
           "cap-dense-20__cap-dense": "rag:20|cap:dense", "fc__both-dense-medium": "fc|both:dense:medium"}


def p1_usd(r, model):
    p = PRICES[model]
    u = r["usage"]
    return ((u["prompt"] - u["cached"]) * p["in"] + u["cached"] * p["cached"] + (u["out"] + u["thoughts"]) * p["out"]) / 1e6


def load_all(reader):
    D = {}
    # phase-1 style runs (single call): 2.5 runs live in memeye_all; 3.8 runs are the dev+test files
    if reader == "gemini-3.8-flash":
        srcs = [("results/mm/memeye_dev", ""), ("results/mm/memeye_test", "")]
    else:
        srcs = [("results/mm/memeye_all", f"__{reader}")]
    for d, sfx in srcs:
        for f in glob.glob(str(ROOT / d / f"*{sfx}.jsonl")):
            base = os.path.basename(f)[:-6]
            if sfx and not base.endswith(sfx):
                continue
            stem = base[: len(base) - len(sfx)] if sfx else base
            if "__seed" in stem or "__gemini" in stem:
                continue
            name = None
            for k in ["10", "20", "40"]:
                for r in ["low", "medium", "high"]:
                    if stem == f"cap-dense-{k}__raw-{r}":
                        name = f"rag:{k}|read:{r}"
            for r in ["low", "medium", "high"]:
                if stem == f"fc__raw-{r}":
                    name = f"fc|read:{r}"
                if stem == f"oracle__raw-{r}":
                    name = f"oracle|read:{r}"
            name = name or P1_NAME.get(stem)
            if not name:
                continue
            for l in open(f):
                r = json.loads(l)
                r["usd"] = p1_usd(r, reader)
                r["n_loaded"] = len(r.get("retrieved", [])) or None
                D.setdefault(name, {})[r["qid"]] = r
    for f in glob.glob(str(ROOT / "results/mm2/memeye_all" / f"*__{reader}*.jsonl")):
        stem = os.path.basename(f)[:-6].replace(f"__{reader}", "")
        parts = stem.split("__")
        sel, rd, extra = parts[0], parts[1], parts[2:]
        if any(x.startswith("seed") for x in extra):
            continue
        name = sel.replace("-", ":") + "|" + rd.replace("-", ":", 1) + "".join(f" +{x}" for x in extra)
        for l in open(f):
            r = json.loads(l)
            D.setdefault(name, {})[r["qid"]] = r
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="gemini-2.5-flash-lite")
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--min_n", type=int, default=371)
    a = ap.parse_args()
    D = load_all(a.reader)
    D = {k: v for k, v in D.items() if len(v) >= a.min_n}
    q = sorted(set.intersection(*[set(v) for v in D.values()]))
    rows = []
    for name, R in D.items():
        s = np.array([R[x]["score"] for x in q])
        lo, hi = boot_ci(s)
        full = [R[x]["full"] for x in q if R[x].get("n_gold") and R[x].get("full") is not None]
        nl = [R[x].get("n_loaded") for x in q if R[x].get("n_loaded")]
        row = {"name": name, "mean": s.mean(), "ci": (lo, hi), "usd1k": 1000 * np.mean([R[x].get("usd", 0) for x in q]),
               "full": np.mean(full) if full else float("nan"), "loaded": np.mean(nl) if nl else float("nan"),
               "err": sum(1 for x in q if R[x].get("error"))}
        for ref in a.ref:
            if ref in D and ref != name:
                d = s - np.array([D[ref][x]["score"] for x in q])
                l2, h2 = boot_ci(d)
                row[ref] = f"{d.mean():+.3f} [{l2:+.3f}, {h2:+.3f}]"
        rows.append(row)
    rows.sort(key=lambda r: -r["mean"])
    print(f"reader={a.reader}  n={len(q)} questions (MemEye, all 8 tasks)\n")
    hdr = "| condition | score [95% CI] | $ / 1k q | rounds loaded | full-set recall | " + " | ".join(f"Δ vs {r}" for r in a.ref) + " |"
    print(hdr)
    print("|---|---|---:|---:|---:|" + "---|" * len(a.ref))
    for r in rows:
        print(f"| {r['name']} | {r['mean']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | {r['usd1k']:.2f} | "
              f"{r['loaded']:.1f} | {r['full']:.3f} | " + " | ".join(r.get(ref, "—") for ref in a.ref) + " |"
              + (f" errors={r['err']}" if r["err"] else ""))


if __name__ == "__main__":
    main()
