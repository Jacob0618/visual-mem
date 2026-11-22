"""Phase 6: offline evaluation of set-aware routing (no API calls). A question classified as multi-item (results/p6/
setness.json) goes to the set path, a single-item question to a cheap path. Reports accuracy and measured cost.
python scripts/route_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import boot_ci  # noqa: E402
from analyze_p2 import load_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
M = json.load(open(ROOT / "results/p6/setness.json"))
V = json.load(open(ROOT / "results/p5/visibility_hat.json"))


def L(p):
    p = ROOT / p
    if not p.exists():
        return {}
    D = {json.loads(l)["qid"]: json.loads(l) for l in open(p)}
    bad = [q for q, r in D.items() if r.get("error")]
    if bad:
        raise SystemExit(f"{p}: {len(bad)} failed records (e.g. {D[bad[0]]['error'][:80]}); re-run before analysis")
    return D


def table(title, conds, ref, routes):
    qs = sorted(set.intersection(*[set(v) for v in conds.values()]))
    multi = np.array([M.get(q, False) for q in qs])
    print(f"\n### {title} ({len(qs)} q; multi-item {multi.mean():.3f})")
    print("| method | all | single | multi | Δ vs " + ref + " [95% CI] | $/1k |")
    print("|---|---:|---:|---:|---|---:|")
    R = dict(conds)
    for name, (single, mult) in routes.items():
        R[name] = {q: (conds[mult][q] if M.get(q, False) else conds[single][q]) for q in qs}
    for name, D in R.items():
        s = np.array([D[q]["score"] for q in qs]); d = s - np.array([conds[ref][q]["score"] for q in qs]); lo, hi = boot_ci(d)
        print(f"| {name} | {s.mean():.3f} | {s[~multi].mean():.3f} | {s[multi].mean():.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | "
              f"{1000 * np.mean([D[q].get('usd', 0) for q in qs]):.2f} |")


def main():
    D5, D8 = load_all("gemini-2.5-flash-lite"), load_all("gemini-3.8-flash")
    table("MemEye, lightweight reader", {"RAG@10": D5["rag:10|read:medium"], "RAG@20": D5["rag:20|read:medium"],
          "RAG@40": D5["rag:40|read:medium"], "QSS": D5["toc38:40|read:medium"]}, "RAG@40",
          {"route: single RAG@10, multi QSS": ("RAG@10", "QSS"), "route: single RAG@20, multi QSS": ("RAG@20", "QSS"),
           "route: single RAG@40, multi QSS": ("RAG@40", "QSS")})
    table("MemEye, strong reader", {"RAG@10": D8["rag:10|read:low"], "RAG@40": D8["rag:40|read:low"],
          "QSS": D8["toc38:40|read:low"], "QSS + index": D8["toc38:40|read+toc:low"]}, "RAG@40",
          {"route: single RAG@10, multi QSS + index": ("RAG@10", "QSS + index"),
           "route: single RAG@40, multi QSS + index": ("RAG@40", "QSS + index")})
    base, keep = "results/mm2/memlens_32k/", None
    f = lambda n: L(base + n)
    QLN = {**L("results/mm2/memlens_32k_memlens_dev/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl"),
           **L("results/mm2/memlens_32k_memlens_test/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl")}
    Q = f("toc38-40__read-medium__gemini-2.5-flash-lite.jsonl")
    VGS = {q: (Q[q] if V[q] >= 0.7 else QLN[q]) for q in QLN if q in Q}
    lite = {"RAG@10": f("rag-10__read-medium__gemini-2.5-flash-lite.jsonl"), "RAG@20": f("rag-20__read-medium__gemini-2.5-flash-lite.jsonl"),
            "RAG@40": f("rag-40__read-medium__gemini-2.5-flash-lite.jsonl"), "VGS": VGS}
    table("MemLens-32k (dev + test), lightweight reader", lite, "RAG@40",
          {"route: single RAG@10, multi VGS": ("RAG@10", "VGS"), "route: single RAG@20, multi VGS": ("RAG@20", "VGS"),
           "route: single RAG@40, multi VGS": ("RAG@40", "VGS")})
    for split in ["dev", "test"]:
        sub = f"results/mm2/memlens_32k_memlens_{split}/"
        S = {"RAG@40": L(sub + "rag-40__read-low__gemini-3.8-flash.jsonl"), "QSS-long": L(sub + "toc38-40__read-low__gemini-3.8-flash__snip600.jsonl")}
        if all(S.values()):
            table(f"MemLens-32k {split}, strong reader", S, "RAG@40", {"route: single QSS-long, multi RAG@40": ("QSS-long", "RAG@40")})
    g = "results/mm2/memgal_all/"
    table("Mem-Gallery, lightweight reader", {"RAG@10": L(g + "rag-10__read-medium__gemini-2.5-flash-lite.jsonl"),
          "RAG@40": L(g + "rag-40__read-medium__gemini-2.5-flash-lite.jsonl"), "QSS": L(g + "toc38-40__read-medium__gemini-2.5-flash-lite.jsonl")},
          "RAG@40", {"route: single RAG@10, multi QSS": ("RAG@10", "QSS")})


if __name__ == "__main__":
    main()
