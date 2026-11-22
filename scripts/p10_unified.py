"""Phase 10: the unified gated method (one rule for every benchmark) from stored per-question outputs. No API calls.
Gate: v_hat(q) >= 0.7 -> standard path, else widened path (tau frozen in P5).
  lightweight reader: standard = QSS over the 200-character index; widened = 600-character index + per-member notes.
  strong reader:      standard = QSS (+ the index as context when the index is cached, i.e. persistent memory);
                      widened  = RAG@40, deferring an abstention to the lightweight reader on the same rounds.
python scripts/p10_unified.py > results/p10_unified.md
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import boot_ci  # noqa: E402
from analyze_p2 import load_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "results/mm2"
TAU = 0.7
ABST = re.compile(r"(not (stated|mentioned|specified|provided|available|contain|include|recorded)|absent|no (information|record|mention)|"
                  r"cannot (be )?(determine|answer|find)|unable to|does not (say|state|mention|specify|contain|include)|isn't (stated|mentioned)|"
                  r"unknown|insufficient)", re.I)


def abst(p):
    return bool(ABST.search(p or "")) and len(p or "") < 400


def L(p):
    D = {}
    for l in open(M / p):
        r = json.loads(l)
        if r.get("error"):
            raise SystemExit(f"{p}: failed records")
        D[r["qid"]] = r
    return D


DEFER = False   # phase 14: the strong widened path is plain RAG@40; set True to reproduce the phase-10 deferral


def defer(strong, lite):
    if not DEFER:
        return strong
    return {q: ({**lite[q], "usd": r["usd"] + lite[q]["usd"], "deferred": True} if abst(r["pred"]) and q in lite else r)
            for q, r in strong.items()}


def gate(V, std, wid, qs):
    return {q: (std[q] if V[q] >= TAU else wid[q]) for q in qs}


def row(name, X, ref, qs, extra_usd=0.0):
    s = np.array([X[q]["score"] for q in qs]); d = s - np.array([ref[q]["score"] for q in qs]); lo, hi = boot_ci(d)
    usd = 1000 * np.mean([X[q].get("usd", 0.0) for q in qs]) + extra_usd
    return f"| {name} | {s.mean():.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {usd:.2f} |"


def main():
    V32 = json.load(open(ROOT / "results/p5/visibility_hat.json"))
    V64 = json.load(open(ROOT / "results/p6/visibility_hat_64k.json"))
    test = set(json.load(open(ROOT / "results/memlens_test.json")))
    P2 = load_all("gemini-2.5-flash-lite")
    from analyze_p2 import p1_usd
    for k in ("rag:40|read:medium", "toc38:40|read:medium"):
        for r in P2[k].values():
            r.setdefault("usd", p1_usd(r, "gemini-2.5-flash-lite") if "usage" in r else 0.0)

    print("### MemEye (development; 371 q; persistent histories)\n")
    me_r40 = P2["rag:40|read:medium"]; me_std = P2["toc38:40|read:medium"]
    me_wid = L("memeye_all/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl")
    qs = sorted(set(me_r40) & set(me_std) & set(me_wid))
    print(f"share on the standard path: {np.mean([V32[q] >= TAU for q in qs]):.3f}\n")
    print("| method | accuracy | Δ vs RAG@40 [95% CI] | $/1k |\n|---|---:|---|---:|")
    print(row("lite RAG@40", me_r40, me_r40, qs))
    print(row("lite VGS (uncached index)", gate(V32, me_std, me_wid, qs), me_r40, qs))
    s_r40 = L("memeye_all/rag-40__read-low__gemini-3.8-flash.jsonl")
    s_std = L("memeye_all/toc38-40__read+toc-low__gemini-3.8-flash__cnoassist+telegraph__xcache.jsonl")
    s_wid = defer(s_r40, me_r40)
    qs2 = sorted(set(qs) & set(s_r40) & set(s_std))
    print(row("strong RAG@40", s_r40, s_r40, qs2))
    print(row("strong VGS (compact cached index; + $0.38/1k cache creation and storage)", gate(V32, s_std, s_wid, qs2), s_r40, qs2, 0.38))

    print("\n### MemLens-32k test part (held-out; 466 q; one history per question, nothing cached)\n")
    r40 = {q: r for q, r in L("memlens_32k/rag-40__read-medium__gemini-2.5-flash-lite.jsonl").items() if q in test}
    std = L("memlens_32k/toc38-40__read-medium__gemini-2.5-flash-lite.jsonl")
    wid = L("memlens_32k_memlens_test/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl")
    qs = sorted(q for q in r40 if q in std and q in wid)
    print(f"share on the standard path: {np.mean([V32[q] >= TAU for q in qs]):.3f}\n")
    print("| method | accuracy | Δ vs RAG@40 [95% CI] | $/1k |\n|---|---:|---|---:|")
    print(row("lite RAG@40", r40, r40, qs))
    print(row("lite VGS", gate(V32, std, wid, qs), r40, qs))
    sr40 = L("memlens_32k_memlens_test/rag-40__read-low__gemini-3.8-flash.jsonl")
    sstd = L("memlens_32k_memlens_test/toc38-40__read-low__gemini-3.8-flash.jsonl")
    swid = defer(sr40, r40)
    qs2 = sorted(set(qs) & set(sr40) & set(sstd))
    print(row("strong RAG@40", sr40, sr40, qs2))
    print(row("strong VGS (post hoc composition: deferral was motivated on this split)", gate(V32, sstd, swid, qs2), sr40, qs2))
    print(row("strong widened path alone (RAG@40 + deferral)", swid, sr40, qs2))

    print("\n### MemLens-64k test part (held-out, histories twice as long; 466 q)\n")
    d = "memlens_64k_memlens_test/"
    r40 = L(d + "rag-40__read-medium__gemini-2.5-flash-lite.jsonl")
    std = L("memlens_64k_memlens64k_test_hivis/toc38-40__read-medium__gemini-2.5-flash-lite.jsonl")
    wid = L("memlens_64k_memlens64k_test_lovis/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl")
    stdc = L("memlens_64k_memlens64k_test_hivis/toc38-40__read-medium__gemini-2.5-flash-lite__cnoassist+shortid+telegraph.jsonl")
    widc = L("memlens_64k_memlens64k_test_lovis/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600__cnoassist+shortid+telegraph.jsonl")
    qs = sorted(q for q in r40 if (q in std if V64[q] >= TAU else q in wid))
    print(f"share on the standard path: {np.mean([V64[q] >= TAU for q in qs]):.3f}\n")
    print("| method | accuracy | Δ vs RAG@40 [95% CI] | $/1k |\n|---|---:|---|---:|")
    print(row("lite RAG@40", r40, r40, qs))
    print(row("lite VGS", gate(V64, std, wid, qs), r40, qs))
    qsc = [q for q in qs if (q in stdc if V64[q] >= TAU else q in widc)]
    print(row("lite VGS, compact index", gate(V64, stdc, widc, qsc), r40, qsc))
    sr40 = L(d + "rag-40__read-low__gemini-3.8-flash.jsonl")
    print(row("strong RAG@40", sr40, sr40, sorted(sr40)))
    print(row("strong widened path (RAG@40 + deferral; registered)", defer(sr40, r40), sr40, sorted(sr40)))
    sstd = L("memlens_64k_memlens64k_test_hivis/toc38-40__read-low__gemini-3.8-flash.jsonl")
    swid = defer(sr40, r40)
    qs2 = sorted(q for q in sr40 if (q in sstd if V64[q] >= TAU else q in swid))
    hi2 = [q for q in qs2 if V64[q] >= TAU]
    print(row("strong VGS (standard path QSS on the high-visibility questions; deferral registered at 64k)", gate(V64, sstd, swid, qs2), sr40, qs2))
    print(row(f"  high-visibility subset ({len(hi2)} q): QSS", sstd, sr40, hi2))
    print(row(f"  high-visibility subset ({len(hi2)} q): RAG@40 + deferral", swid, sr40, hi2))


if __name__ == "__main__":
    main()
