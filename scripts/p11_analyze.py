"""Phase 11: external baselines vs RAG@40 and VGS on the same questions (paired bootstrap). No API calls.
python scripts/p11_analyze.py > results/p11_baselines.md"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import boot_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "results/mm2"
TAU = 0.7
STRONG_DEFER = False   # phase 14: the strong widened path is plain RAG@40 (no deferral to the lightweight reader)


def L(p):
    p = Path(p)
    if not p.exists():
        return {}
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p) if not json.loads(l).get("error")}


def bl(bench, system, reader):
    p = ROOT / f"results/baselines/{bench}/{system}.jsonl"
    if not p.exists():
        return {}
    return {r["qid"]: r for r in (json.loads(l) for l in open(p)) if not r.get("error") and r.get("reader") == reader}


def ours(bench):
    """RAG@40 and VGS per reader, as composed in scripts/p10_unified.py / p10_vm_analyze.py."""
    import re
    ABST = re.compile(r"(not (stated|mentioned|specified|provided|available|contain|include|recorded)|absent|no (information|record|mention)|"
                      r"cannot (be )?(determine|answer|find)|unable to|does not (say|state|mention|specify|contain|include)|isn't (stated|mentioned)|"
                      r"unknown|insufficient)", re.I)
    defer = lambda S, Lt: ({q: (Lt[q] if ABST.search(r["pred"] or "") and len(r["pred"] or "") < 400 and q in Lt else r)
                            for q, r in S.items()} if STRONG_DEFER else S)
    out = {}
    if bench == "memeye":
        from analyze_p2 import load_all
        P2 = load_all("gemini-2.5-flash-lite")
        V = json.load(open(ROOT / "results/p5/visibility_hat.json"))
        r40, std = P2["rag:40|read:medium"], P2["toc38:40|read:medium"]
        wid = L(M / "memeye_all/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl")
        out["lite"] = (r40, {q: (std[q] if V[q] >= TAU else wid[q]) for q in std if q in wid})
        s40 = L(M / "memeye_all/rag-40__read-low__gemini-3.8-flash.jsonl")
        sstd = L(M / "memeye_all/toc38-40__read+toc-low__gemini-3.8-flash__cnoassist+telegraph__xcache.jsonl")
        sw = defer(s40, r40)
        out["strong"] = (s40, {q: (sstd[q] if V[q] >= TAU else sw[q]) for q in sstd if q in sw})
    elif bench == "memlens":
        V = json.load(open(ROOT / "results/p5/visibility_hat.json"))
        test = set(json.load(open(ROOT / "results/memlens_test.json")))
        r40 = {q: r for q, r in L(M / "memlens_32k/rag-40__read-medium__gemini-2.5-flash-lite.jsonl").items() if q in test}
        std = L(M / "memlens_32k/toc38-40__read-medium__gemini-2.5-flash-lite.jsonl")
        wid = L(M / "memlens_32k_memlens_test/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl")
        out["lite"] = (r40, {q: (std[q] if V[q] >= TAU else wid[q]) for q in r40 if q in std and q in wid})
        s40 = L(M / "memlens_32k_memlens_test/rag-40__read-low__gemini-3.8-flash.jsonl")
        sstd = L(M / "memlens_32k_memlens_test/toc38-40__read-low__gemini-3.8-flash.jsonl")
        sw = defer(s40, r40)
        out["strong"] = (s40, {q: (sstd[q] if V[q] >= TAU else sw[q]) for q in s40 if q in sstd})
    else:
        V = json.load(open(ROOT / "results/p10/vm_visibility_hat.json"))
        r40 = L(M / "visualmem_all/rag-40__read-medium__gemini-2.5-flash-lite.jsonl")
        std = L(M / "visualmem_all/toc38-40__read-medium__gemini-2.5-flash-lite__xcache.jsonl")
        wid = L(M / "visualmem_all_vm_lovis/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600__xcache.jsonl")
        out["lite"] = (r40, {q: (std[q] if V[q] >= TAU else wid[q]) for q in r40 if q in std and (V[q] >= TAU or q in wid)})
        s40 = L(M / "visualmem_all/rag-40__read-low__gemini-3.8-flash.jsonl")
        sstd = L(M / "visualmem_all_vm_hivis/toc38-40__read+toc-low__gemini-3.8-flash__xcache.jsonl")
        out["strong"] = (s40, {q: (sstd[q] if V[q] >= TAU else s40[q]) for q in s40 if V[q] < TAU or q in sstd})
    return out


def main():
    for bench in ["memeye", "memlens", "visualmem"]:
        O = ours(bench)
        for reader in ["lite", "strong"]:
            R40, VGS = O[reader]
            systems = {s: bl(bench, s, reader) for s in ["simplemem", "omni", "mirix"]}
            systems = {s: X for s, X in systems.items() if X}
            if not systems:
                continue
            print(f"\n### {bench}, {reader} reader\n")
            print("| method | n | accuracy | Δ vs RAG@40 [95% CI] | Δ vs VGS [95% CI] | $/1k |\n|---|---:|---:|---|---|---:|")
            for name, X in [("RAG@40", R40), *systems.items(), ("VGS", VGS)]:
                qs = sorted(q for q in X if q in R40 and q in VGS)
                s = np.array([X[q]["score"] for q in qs])
                d1 = s - np.array([R40[q]["score"] for q in qs]); d2 = s - np.array([VGS[q]["score"] for q in qs])
                l1, h1 = boot_ci(d1); l2, h2 = boot_ci(d2)
                usd = 1000 * np.mean([X[q].get("usd", 0.0) for q in qs])
                print(f"| {name} | {len(qs)} | {s.mean():.3f} | {d1.mean():+.3f} [{l1:+.3f}, {h1:+.3f}] | {d2.mean():+.3f} [{l2:+.3f}, {h2:+.3f}] | {usd:.2f} |")


if __name__ == "__main__":
    main()
