"""Phase 16: VGS with the compact index (no assistant snippet, short ids under session headers, telegraphed user text),
composed offline from per-question outputs (v2 image descriptions). Gate: history-level visibility H >= 0.45 (P16-D9);
P16_GATE=question restores the per-question estimate (v_hat on the compact lines, tau = 0.7).
No API calls.  python scripts/p16_compose.py"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze import boot_ci  # noqa: E402

TAU = 0.7
LITE, STRONG = "gemini-2.5-flash-lite", "gemini-3.8-flash"
C, C600, CS = "__cnoassist+shortid+telegraph", "__snip600__cnoassist+shortid+telegraph", "__cnoassist+telegraph"
P14, P16 = ROOT / "results/p14/runs", ROOT / "results/p16/final"
SRC = {
    "memeye": {"rag_l": P14 / f"memeye/rag-40__read-medium__{LITE}.jsonl", "rag_s": P14 / f"memeye/rag-40__read-low__{STRONG}.jsonl",
               "std_l": P16 / f"memeye/toc38-40__read-medium__{LITE}{C}__xcache.jsonl",
               "wid_l": P16 / f"memeye/toc38-40__readn-medium__{LITE}{C600}__xcache.jsonl",
               "std_s": P14 / f"memeye/toc38-40__read+toc-low__{STRONG}{CS}__xcache.jsonl"},
    "memlens": {"rag_l": P16 / f"memlens/rag-40__read-medium__{LITE}.jsonl", "rag_s": P16 / f"memlens/rag-40__read-low__{STRONG}.jsonl",
                "std_l": P16 / f"memlens/toc38-40__read-medium__{LITE}{C}.jsonl",
                "wid_l": P16 / f"memlens/toc38-40__readn-medium__{LITE}{C600}.jsonl",
                "std_s": P16 / f"memlens/toc38-40__read-low__{STRONG}{C}.jsonl"},
    "visualmem": {"rag_l": P14 / f"visualmem/rag-40__read-medium__{LITE}.jsonl", "rag_s": P14 / f"visualmem/rag-40__read-low__{STRONG}.jsonl",
                  "std_l": P14 / f"visualmem/toc38-40__read-medium__{LITE}{C}__xcache.jsonl",
                  "wid_l": P14 / f"visualmem/toc38-40__readn-medium__{LITE}{C600}__xcache.jsonl",
                  "std_s": P16 / f"visualmem/toc38-40__read+toc-low__{STRONG}{CS}__xcache.jsonl"}}
SRC["memeye"]["uni_s"] = P16 / f"memeye/tocu-40-40__read+toc-low__{STRONG}{CS}__xcache.jsonl"
SRC["visualmem"]["uni_s"] = P16 / f"visualmem/tocu-40-40__read+toc-low__{STRONG}{CS}__xcache.jsonl"
SRC["memlens"]["uni_s"] = P16 / f"memlens/tocu-40-40__read-low__{STRONG}{C}.jsonl"
SRC["memlens64"] = {k: Path(str(v).replace("/final/memlens/", "/final/memlens64/")) for k, v in SRC["memlens"].items()}
VHAT = {"memeye": "vhat_compact_memeye.json", "memlens": "vhat_compact_memlens_test.json", "visualmem": "vhat_compact_visualmem.json",
        "memlens64": "vhat_compact_memlens_test64.json"}
# MemEye re-read after removing the example phrase from the reader prompt (P16-D6)
P16B = ROOT / "results/p16/final2/memeye"
SRC["memeye"] = {"rag_l": P16B / f"rag-40__read-medium__{LITE}.jsonl", "rag_s": P16B / f"rag-40__read-low__{STRONG}.jsonl",
                 "std_l": P16B / f"toc38-40__read-medium__{LITE}{C}__xcache.jsonl",
                 "wid_l": P16B / f"toc38-40__readn-medium__{LITE}{C600}__xcache.jsonl",
                 "std_s": P16B / f"toc38-40__read+toc-low__{STRONG}{CS}__xcache.jsonl",
                 "uni_s": P16B / f"tocu-40-40__read+toc-low__{STRONG}{CS}__xcache.jsonl"}
# explicit-cache creation + storage per 1,000 questions (persistent histories), measured from the usage log
CACHE_USD = {("memeye", "lite"): 0.2, ("memeye", "strong"): 0.3, ("visualmem", "lite"): 0.71, ("visualmem", "strong"): 3.13}


def L(p):
    if not Path(p).exists():
        return None
    R = {}
    for line in open(p):
        r = json.loads(line)
        R[r["qid"]] = r
    bad = [q for q, r in R.items() if r.get("error")]
    if bad:
        raise SystemExit(f"{p}: {len(bad)} failed records")
    return R


# Gate of the paper (P16-D9): history-level visibility, computed at write time and independent of the question.
# H(h) = mean over the rounds of a history of the share of the round's content words that its compact index line shows
# (scripts/p16_hvis.py). TAU_H is the midpoint of the gap on the development data (MemLens dev max .353, MemEye min .552).
GATE = os.getenv("P16_GATE", "history")          # "history" (paper) | "question" (per-question estimate v_hat >= TAU)
TAU_H = 0.45
HVIS = {"memeye": "hvis_memeye.json", "memlens": "hvis_memlens.json", "memlens64": "hvis_memlens64.json",
        "visualmem": "hvis_visualmem.json"}


def hvis(bench):
    return json.load(open(ROOT / "results/p16" / HVIS[bench]))


def routes(bench, V=None):
    """qid -> True when the question's history takes the standard path."""
    if GATE == "history":
        return {q: h["H200"] >= TAU_H for q, h in hvis(bench).items()}
    V = V if V is not None else json.load(open(ROOT / "results/p16" / VHAT[bench]))
    return {q: v >= TAU for q, v in V.items()}


def gate(route, std, wid):
    """route: qid -> True for the standard path (a missing question takes the standard path)."""
    return {q: (std[q] if route.get(q, True) else wid[q]) for q in std if q in wid}


def qgate(V, std, wid):
    """The per-question estimate as a router (analysis only): standard path when v_hat(q) >= TAU."""
    return {q: (std[q] if V.get(q, 1.0) >= TAU else wid[q]) for q in std if q in wid}


def compose(bench, qids=None):
    S = {k: L(v) for k, v in SRC[bench].items()}
    V = json.load(open(ROOT / "results/p16" / VHAT[bench]))
    out = {k: S.get(k) for k in ("rag_l", "rag_s", "std_l", "wid_l", "std_s", "uni_s")}
    R = routes(bench, V)
    if S["std_l"] and S["wid_l"]:
        out["vgs_l"] = gate(R, S["std_l"], S["wid_l"])
    if S["std_s"] and S.get("uni_s"):      # strong widened path: selected rounds U the 40 most similar rounds
        out["vgs_s"] = gate(R, S["std_s"], S["uni_s"])
    if qids is not None:
        out = {k: ({q: r for q, r in D.items() if q in qids} if D else None) for k, D in out.items()}
    out["route"] = R
    return out, V


def row(D, base, qs):
    a = np.array([D[q]["score"] for q in qs]); d = a - np.array([base[q]["score"] for q in qs]); lo, hi = boot_ci(d)
    usd = 1000 * float(np.mean([D[q].get("usd", 0.0) for q in qs]))
    return a.mean(), d.mean(), lo, hi, usd


def main():
    sub150 = set(json.load(open(ROOT / "results/p11_memlens_sub150.json")))
    for bench, qids in [("memeye", None), ("memlens", None), ("memlens", sub150), ("memlens64", None), ("visualmem", None)]:
        if not all(Path(v).exists() for v in SRC[bench].values()) or not (ROOT / "results/p16" / VHAT[bench]).exists():
            print(f"{bench}: runs incomplete"); continue
        out, V = compose(bench, qids)
        name = bench + (" (150 subset)" if qids else "")
        for reader, base_k, keys in [("lite", "rag_l", ["std_l", "wid_l", "vgs_l"]), ("strong", "rag_s", ["std_s", "uni_s", "vgs_s"])]:
            base = out[base_k]
            if not base:
                print(f"{name:22s} {reader}: RAG@40 run missing"); continue
            qs = sorted(set(base).intersection(*[set(out[k]) for k in keys if out.get(k)]))
            line = f"{name:22s} {reader:6s} n={len(qs):3d} RAG@40 {np.mean([base[q]['score'] for q in qs]):.3f} (${1000*np.mean([base[q].get('usd',0) for q in qs]):.2f})"
            for k in keys:
                if out.get(k):
                    a, d, lo, hi, usd = row(out[k], base, qs)
                    line += f" | {k} {a:.3f} {d:+.3f} [{lo:+.3f},{hi:+.3f}] ${usd:.2f}"
            share = np.mean([out["route"].get(q, True) for q in qs])
            print(line + f" | standard share {share:.2f}")


if __name__ == "__main__":
    main()
