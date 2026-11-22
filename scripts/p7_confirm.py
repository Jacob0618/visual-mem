"""Phase 7 confirmation (registered P7-D1): compressed VGS on MemLens-64k test-split questions. No API calls."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import boot_ci  # noqa: E402
from p6_confirm import L, M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    V = json.load(open(ROOT / "results/p6/visibility_hat_64k.json"))
    d = "results/mm2/memlens_64k_memlens_test/"
    lo_d, hi_d = "results/mm2/memlens_64k_memlens64k_test_lovis/", "results/mm2/memlens_64k_memlens64k_test_hivis/"
    R40 = L(d + "rag-40__read-medium__gemini-2.5-flash-lite.jsonl")
    F_lo, F_hi = L(lo_d + "toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl"), L(hi_d + "toc38-40__read-medium__gemini-2.5-flash-lite.jsonl")
    C_lo = L(lo_d + "toc38-40__readn-medium__gemini-2.5-flash-lite__snip600__cnoassist+shortid+telegraph.jsonl")
    C_hi = L(hi_d + "toc38-40__read-medium__gemini-2.5-flash-lite__cnoassist+shortid+telegraph.jsonl")
    N_lo = L(lo_d + "toc@3.8nt-40__readn-medium__gemini-2.5-flash-lite__snip600__cnoassist+shortid+telegraph.jsonl")
    N_hi = L(hi_d + "toc@3.8nt-40__read-medium__gemini-2.5-flash-lite__cnoassist+shortid+telegraph.jsonl")
    VGS = {q: (F_hi[q] if V[q] >= 0.7 else F_lo[q]) for q in R40}
    cVGS = {q: (C_hi[q] if V[q] >= 0.7 else C_lo[q]) for q in R40}
    nVGS = {q: (N_hi[q] if V[q] >= 0.7 else N_lo[q]) for q in R40}
    T = {"RAG@40": R40, "VGS (frozen index)": VGS, "VGS (compressed index)": cVGS,
         "VGS (compressed index, selector without thinking)": nVGS,
         "routed VGS (compressed, selector without thinking)": {q: (nVGS[q] if M[q] else R40[q]) for q in R40},
         "routed VGS (frozen)": {q: (VGS[q] if M[q] else R40[q]) for q in R40},
         "routed VGS (compressed)": {q: (cVGS[q] if M[q] else R40[q]) for q in R40}}
    qs = sorted(R40)
    print(f"### MemLens-64k test-split questions ({len(qs)}), lightweight reader\n")
    print("| method | accuracy | Δ vs RAG@40 [95% CI] | Δ vs frozen VGS [95% CI] | $/1k |\n|---|---:|---|---|---:|")
    d4 = np.array([nVGS[q]["score"] - cVGS[q]["score"] for q in qs]); a4, b4 = boot_ci(d4)
    print(f"C4: no-thinking minus thinking selector (compressed VGS): {d4.mean():+.3f} [{a4:+.3f}, {b4:+.3f}]\n")
    for k, D in T.items():
        s = np.array([D[q]["score"] for q in qs])
        d1 = s - np.array([R40[q]["score"] for q in qs]); d2 = s - np.array([VGS[q]["score"] for q in qs])
        a1, b1 = boot_ci(d1); a2, b2 = boot_ci(d2)
        print(f"| {k} | {s.mean():.3f} | {d1.mean():+.3f} [{a1:+.3f}, {b1:+.3f}] | {d2.mean():+.3f} [{a2:+.3f}, {b2:+.3f}] | "
              f"{1000 * np.mean([D[q]['usd'] for q in qs]):.2f} |")


if __name__ == "__main__":
    main()
