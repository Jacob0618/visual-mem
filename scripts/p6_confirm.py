"""Phase 6 confirmation (registered in RESEARCH_LOG P6-D4); no API calls.
(A) MemLens-64k test-split questions; (B) Mem-Gallery strong reader with the explicit index cache."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import boot_ci  # noqa: E402
from analyze_p2 import p1_usd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
M = json.load(open(ROOT / "results/p6/setness.json"))
ABST = re.compile(r"(not (stated|mentioned|specified|provided|available|contain|include|recorded)|absent|no (information|record|mention)|"
                  r"cannot (be )?(determine|answer|find)|unable to|does not (say|state|mention|specify|contain|include)|isn't (stated|mentioned)|"
                  r"unknown|insufficient)", re.I)


def abst(p):
    return bool(ABST.search(p or "")) and len(p or "") < 400


def L(p, reader=None):
    p = ROOT / p
    if not p.exists():
        return {}
    D = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("error"):
            raise SystemExit(f"{p}: contains failed records ({r['error'][:80]}); re-run the condition before analysis")
        if reader and "usd" not in r:
            r["usd"] = p1_usd(r, reader)
        D[r["qid"]] = r
    return D


def table(title, T, ref):
    qs = sorted(set.intersection(*[set(v) for v in T.values()]))
    print(f"\n### {title} ({len(qs)} q)\n\n| method | accuracy | Δ vs {ref} [95% CI] | $/1k |\n|---|---:|---|---:|")
    for k, D in T.items():
        s = np.array([D[q]["score"] for q in qs]); d = s - np.array([T[ref][q]["score"] for q in qs]); lo, hi = boot_ci(d)
        print(f"| {k} | {s.mean():.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {1000 * np.mean([D[q]['usd'] for q in qs]):.2f} |")


def main():
    V = json.load(open(ROOT / "results/p6/visibility_hat_64k.json"))
    d = "results/mm2/memlens_64k_memlens_test/"
    R40 = L(d + "rag-40__read-medium__gemini-2.5-flash-lite.jsonl")
    TH = L(d + "rag-40__read-medium__gemini-2.5-flash-lite__think1024.jsonl")
    QLN = L("results/mm2/memlens_64k_memlens64k_test_lovis/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl")
    QH = L("results/mm2/memlens_64k_memlens64k_test_hivis/toc38-40__read-medium__gemini-2.5-flash-lite.jsonl")
    VGS = {q: (QH[q] if V[q] >= 0.7 else QLN[q]) for q in R40 if (q in QH if V[q] >= 0.7 else q in QLN)}
    lite = {"RAG@40": R40, "RAG@40 + thinking": TH, "VGS": VGS,
            "routed thinking (single RAG@40, multi RAG@40 + thinking)": {q: (TH[q] if M[q] else R40[q]) for q in TH},
            "routed VGS (single RAG@40, multi VGS)": {q: (VGS[q] if M[q] else R40[q]) for q in VGS}}
    table("MemLens-64k test-split questions, lightweight reader", lite, "RAG@40")
    S40 = L(d + "rag-40__read-low__gemini-3.8-flash.jsonl")
    fb = {}
    for q, r in S40.items():
        if q in R40 and abst(r["pred"]):
            fb[q] = {**R40[q], "usd": r["usd"] + R40[q]["usd"]}
        else:
            fb[q] = r
    print(f"\nstrong-reader abstentions at 64k: {np.mean([abst(r['pred']) for r in S40.values()]):.3f}")
    table("MemLens-64k test-split questions, strong reader", {"RAG@40": S40, "RAG@40 + abstention fallback": fb}, "RAG@40")
    G40 = L("results/mm/memgal_heldout/cap-dense-40__raw-low.jsonl", "gemini-3.8-flash")
    GQ = L("results/mm2/memgal_all/toc38-40__read+toc-low__gemini-3.8-flash__xcache.jsonl")
    GQu = L("results/mm2/memgal_all/toc38-40__read+toc-low__gemini-3.8-flash.jsonl")
    if GQ:
        c = [json.loads(l) for l in open(ROOT / "cache/usage.jsonl") if '"xcache:create"' in l]
        table("Mem-Gallery, strong reader (cached index costs exclude cache creation and storage)",
              {"RAG@40": G40, "QSS + index (uncached)": GQu, "QSS + index (cached index)": GQ}, "RAG@40")


if __name__ == "__main__":
    main()
