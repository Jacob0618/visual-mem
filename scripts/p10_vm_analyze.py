"""Phase 10: VisualMem, frozen unified method (RESEARCH_LOG P10-D5). No API calls.
python scripts/p10_vm_analyze.py > results/p10/vm_results.md
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyze import boot_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/mm2/visualmem_all"
TAU = 0.7
ABST = re.compile(r"(not (stated|mentioned|specified|provided|available|contain|include|recorded)|absent|no (information|record|mention)|"
                  r"cannot (be )?(determine|answer|find)|unable to|does not (say|state|mention|specify|contain|include)|isn't (stated|mentioned)|"
                  r"unknown|insufficient)", re.I)
T0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0   # start time of the P10-D5 runs (for cache creation costs)


def L(name):
    p = D / name if (D / name).exists() else Path(str(D) + "_vm_lovis") / name
    if not p.exists():
        p = Path(str(D) + "_vm_hivis") / name
    out = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("error"):
            raise SystemExit(f"{name}: failed record {r['qid']}: {r['error'][:80]}")
        out[r["qid"]] = r
    return out


def row(name, X, ref, qs, extra=0.0):
    s = np.array([X[q]["score"] for q in qs]); d = s - np.array([ref[q]["score"] for q in qs]); lo, hi = boot_ci(d)
    usd = 1000 * np.mean([X[q].get("usd", 0.0) for q in qs]) + extra
    return f"| {name} | {s.mean():.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {usd:.2f} |"


def cache_overhead(model):
    """Creation (cached-input price for the first write is billed as input) + storage of explicit caches since T0."""
    price = {"gemini-3.8-flash": 0.75, "gemini-2.5-flash-lite": 0.10}[model]
    tot = 0.0
    for l in open(ROOT / "cache/usage.jsonl"):
        if '"xcache:create"' not in l:
            continue
        r = json.loads(l)
        if r["t"] >= T0 and r["model"] == model:
            tot += r["prompt"] * price / 1e6 + r.get("storage_usd", 0.0)
    return tot


def main():
    V = json.load(open(ROOT / "results/p10/vm_visibility_hat.json"))
    R = L("rag-40__read-medium__gemini-2.5-flash-lite.jsonl")
    S = L("toc38-40__read-medium__gemini-2.5-flash-lite__xcache.jsonl")
    W = L("toc38-40__readn-medium__gemini-2.5-flash-lite__snip600__xcache.jsonl")
    qs = sorted(q for q in R if q in S and (V[q] >= TAU or q in W))
    G = {q: (S[q] if V[q] >= TAU else W[q]) for q in qs}
    over = cache_overhead("gemini-3.8-flash")
    print(f"### VisualMem (frozen unified method; {len(qs)} questions; 10 persistent histories of 1,017 rounds)\n")
    print(f"share on the standard path: {np.mean([V[q] >= TAU for q in qs]):.3f}; cache creation + storage since T0: ${over:.2f} "
          f"(${1000 * over / len(qs):.2f} per 1k questions, all runs together)\n")
    print("| method | accuracy | Δ vs RAG@40 [95% CI] | $/1k |\n|---|---:|---|---:|")
    print(row("lite RAG@40", R, R, qs))
    print(row("lite QSS (standard path for all)", S, R, qs))
    print(row("lite VGS (primary V1)", G, R, qs))
    lo = [q for q in qs if V[q] < TAU]; hi = [q for q in qs if V[q] >= TAU]
    print(row(f"  low-visibility ({len(lo)}): widened path", W, R, lo))
    print(row(f"  low-visibility ({len(lo)}): standard path", S, R, lo))
    print(row(f"  high-visibility ({len(hi)}): standard path", S, R, hi))
    sR = L("rag-40__read-low__gemini-3.8-flash.jsonl")
    sS = L("toc38-40__read+toc-low__gemini-3.8-flash__xcache.jsonl")
    sW = {q: ({**R[q], "usd": r["usd"] + R[q]["usd"]} if ABST.search(r["pred"] or "") and len(r["pred"] or "") < 400 else r) for q, r in sR.items()}
    qs2 = sorted(q for q in qs if q in sR and (V[q] < TAU or q in sS))
    sG = {q: (sS[q] if V[q] >= TAU else sW[q]) for q in qs2}
    print(row("strong RAG@40", sR, sR, qs2))
    print(row("strong VGS (secondary V2)", sG, sR, qs2))
    hi2 = [q for q in qs2 if V[q] >= TAU]
    print(row(f"  high-visibility ({len(hi2)}): QSS + cached index", sS, sR, hi2))
    print(row(f"  high-visibility ({len(hi2)}): RAG@40 + deferral", sW, sR, hi2))
    print(row(f"  all: RAG@40 + deferral", sW, sR, qs2))
    print("\nfull-set recall of the opened rounds (gold = image rounds of the question's event):")
    for name, X in [("RAG@40", R), ("QSS 200-character index", S), ("QSS 600-character index (low-visibility questions)", W)]:
        f = [X[q]["full"] for q in X if X[q].get("full") is not None]
        n = [X[q].get("n_loaded", 0) for q in X]
        print(f"  {name}: FSR {np.mean(f):.3f} over {len(f)} q, rounds {np.mean(n):.1f}")


if __name__ == "__main__":
    main()
