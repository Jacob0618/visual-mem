"""Phase 6: MemEye cost-accuracy table with measured explicit-cache costs (no API calls)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import boot_ci  # noqa: E402
from analyze_p2 import load_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
M = json.load(open(ROOT / "results/p6/setness.json"))


def L(n):
    return {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "results/mm2/memeye_all" / n)}


def main():
    per_q_cache = {"lite": 0.28 / 1000, "strong": 0.52 / 1000}   # explicit-cache creation + storage (results/p6/xcache_overheads.json)
    D5, D8 = load_all("gemini-2.5-flash-lite"), load_all("gemini-3.8-flash")
    lite = {"RAG@40": D5["rag:40|read:medium"], "RAG@40 + thinking": D5["rag:40|read:medium +think1024"],
            "QSS (uncached index; draw 1)": D5["toc38:40|read:medium"], "QSS (uncached; draw 2)": L("toc38-40__read-medium__gemini-2.5-flash-lite__draw2.jsonl"),
            "QSS (cached index)": L("toc38-40__read-medium__gemini-2.5-flash-lite__xcache.jsonl"),
            "QSS (cached index; draw 2)": L("toc38-40__read-medium__gemini-2.5-flash-lite__xcache__draw3.jsonl"),
            "QSS + thinking (uncached)": D5["toc38:40|read:medium +think1024"],
            "QSS + thinking (cached index)": L("toc38-40__read-medium__gemini-2.5-flash-lite__think1024__xcache.jsonl")}
    strong = {"RAG@10": L("rag-10__read-low__gemini-3.8-flash.jsonl"), "RAG@40": L("rag-40__read-low__gemini-3.8-flash.jsonl"),
              "QSS + index (uncached)": D8["toc38:40|read+toc:low"],
              "QSS + index (cached index)": L("toc38-40__read+toc-low__gemini-3.8-flash__xcache.jsonl")}
    def route(single, multi):
        return {q: (multi[q] if M.get(q) else single[q]) for q in multi}
    lite["route: single RAG@40 + thinking, multi QSS + thinking (cached)"] = route(lite["RAG@40 + thinking"], lite["QSS + thinking (cached index)"])
    lite["route: single RAG@40, multi QSS + thinking (cached)"] = route(lite["RAG@40"], lite["QSS + thinking (cached index)"])
    strong["route: single RAG@40, multi QSS + index (cached)"] = route(strong["RAG@40"], strong["QSS + index (cached index)"])
    cached_names = {k for k in list(lite) + list(strong) if "cached" in k and "uncached" not in k}
    for title, T, ref in [("lightweight reader", lite, "RAG@40"), ("strong reader", strong, "RAG@40")]:
        qs = sorted(set.intersection(*[set(v) for v in T.values()]))
        pq = per_q_cache["strong" if "strong" in title else "lite"]
        print(f"\n### MemEye ({len(qs)} q), {title}; cached runs include explicit-cache creation and storage (${1000 * pq:.2f}/1k)\n")
        print("| method | accuracy | Δ vs RAG@40 [95% CI] | $/1k |"); print("|---|---:|---|---:|")
        for k, D in T.items():
            s = np.array([D[q]["score"] for q in qs]); d = s - np.array([T[ref][q]["score"] for q in qs]); lo, hi = boot_ci(d)
            extra = 0.0
            if k in cached_names:
                frac = np.mean([M.get(q, False) for q in qs]) if k.startswith("route") else 1.0
                extra = pq * frac
            cost = 1000 * (np.mean([D[q]["usd"] for q in qs]) + extra)
            print(f"| {k} | {s.mean():.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {cost:.2f} |")


if __name__ == "__main__":
    main()
