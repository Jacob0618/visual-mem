"""Phase 8: registered external test on WorldMemArena lifelong-personal (RESEARCH_LOG P8-D1/D1a). No API calls.
python scripts/p8_wma_analyze.py > results/p8/wma_results.md
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import boot_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/mm2/wma_all_wma_personal_qids"
V = json.load(open(ROOT / "results/p8/wma_visibility_hat.json"))


def L(name):
    R = {}
    for l in open(D / name):
        r = json.loads(l)
        if r.get("error"):
            raise SystemExit(f"{name}: failed records ({r['error'][:80]})")
        R[r["qid"]] = r
    return R


def cache_overhead(R, base=None):
    """Explicit-cache creation + one hour of storage, attributed analytically: one cache per history whose size is the
    cached prefix of its calls (median over the questions of the history; minus the cached tokens of a reused base call,
    e.g. the selection call that the strong pipeline shares with the QSS run)."""
    by = defaultdict(list)
    for q, r in R.items():
        c = r["cached_tok"] - (base[q]["cached_tok"] if base else 0)
        by[r["task"]].append(max(0, c))
    return sum(float(np.median(v)) for v in by.values()) * (0.75 + 0.50) / 1e6


def row(name, D_, ref, qs, extra=0.0):
    s = np.array([D_[q]["score"] for q in qs]); d = s - np.array([ref[q]["score"] for q in qs]); lo, hi = boot_ci(d)
    cost = 1000 * (np.mean([D_[q]["usd"] for q in qs]) + extra / len(qs))
    return f"| {name} | {s.mean():.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {cost:.2f} |"


def main():
    R40 = L("rag-40__read-medium__gemini-2.5-flash-lite.jsonl")
    Q2 = L("toc38-40__read-medium__gemini-2.5-flash-lite__xcache.jsonl")
    QN = L("toc38-40__readn-medium__gemini-2.5-flash-lite__snip600__xcache.jsonl")
    qs = sorted(set(R40) & set(Q2) & set(QN))
    hi = [q for q in qs if V[q] >= 0.7]; lo = [q for q in qs if V[q] < 0.7]
    VGS = {q: (Q2[q] if V[q] >= 0.7 else QN[q]) for q in qs}
    ov_q2, ov_qn = cache_overhead(Q2), cache_overhead(QN)
    print(f"### WorldMemArena lifelong-personal ({len(qs)} questions; {len(hi)} with v_hat >= 0.7), lightweight reader\n")
    print("| method | accuracy | Δ vs RAG@40 [95% CI] | $/1k |\n|---|---:|---|---:|")
    print(row("RAG@40", R40, R40, qs))
    print(row("QSS (cached index)", Q2, R40, qs, ov_q2))
    print(row("QSS 600 + notes (cached index)", QN, R40, qs, ov_qn))
    print(row("VGS (frozen gate)", VGS, R40, qs, ov_q2 * len(hi) / len(qs) + ov_qn * len(lo) / len(qs)))
    print("\nW3, notes path minus QSS-200 by side of the threshold:")
    gains = {}
    for name, sub in [("v_hat >= 0.7", hi), ("v_hat < 0.7", lo)]:
        d = np.array([QN[q]["score"] - Q2[q]["score"] for q in sub]); l_, h_ = boot_ci(d); gains[name] = d
        print(f"  {name} (n={len(sub)}): {d.mean():+.3f} [{l_:+.3f}, {h_:+.3f}]")
    rng = np.random.default_rng(0)
    a, b = gains["v_hat < 0.7"], gains["v_hat >= 0.7"]
    if len(a) and len(b):
        bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(10000)]
        print(f"  difference (low minus high): {a.mean() - b.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    by = defaultdict(list)
    for q in qs:
        by[R40[q]["qtype"]].append(q)
    print("\nBy question type (VGS minus RAG@40):")
    for t, sub in sorted(by.items()):
        d = np.array([VGS[q]["score"] - R40[q]["score"] for q in sub])
        print(f"  {t} (n={len(sub)}): RAG@40 {np.mean([R40[q]['score'] for q in sub]):.3f}, VGS {np.mean([VGS[q]['score'] for q in sub]):.3f}, diff {d.mean():+.3f}")
    try:
        S40 = L("rag-40__read-low__gemini-3.8-flash.jsonl")
        SQ = L("toc38-40__read+toc-low__gemini-3.8-flash__xcache.jsonl")
    except FileNotFoundError:
        return
    qs2 = sorted(set(S40) & set(SQ))
    print(f"\n### Strong reader ({len(qs2)} questions)\n\n| method | accuracy | Δ vs RAG@40 [95% CI] | $/1k |\n|---|---:|---|---:|")
    print(row("RAG@40", S40, S40, qs2))
    print(row("QSS + index (cached index)", SQ, S40, qs2, cache_overhead(Q2) + cache_overhead(SQ, Q2)))


if __name__ == "__main__":
    main()
