"""Phase 13: online latency per question (wall-clock from receiving the question to the reader's answer, memory
already built, one question at a time) on fixed samples: 30 MemEye questions (Home_Renovation_Interior_Design, 120
rounds), 10 MemLens-32k questions, 30 VisualMem questions (persona 0, 1,013 rounds). No API calls; reads
results/p13/latency/. The first question of a persistent history is a warm-up (cache creation) and is excluded.
A selection or embedding request that an earlier condition had already issued is served by the local response cache in
0 s; such records take the latency of the same question's fresh request.
VGS composes the gated path per question as in Table 2; the strong widened path adds the lightweight RAG@40 call when
the strong reader abstains (deferral).
python scripts/p13_latency.py -> results/p13/latency_summary.json, results/p13/latency_fig.json"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / "results/p13/latency"
TAU = 0.7
ABST = re.compile(r"(not (stated|mentioned|specified|provided|available|contain|include|recorded)|absent|no (information|record|mention)|"
                  r"cannot (be )?(determine|answer|find)|unable to|does not (say|state|mention|specify|contain|include)|isn't (stated|mentioned)|"
                  r"unknown|insufficient)", re.I)
HIT = 0.05      # seconds: a selection/embedding faster than this was served from the local cache


def load(p):
    p = Path(p)
    if not p.exists():
        return {}
    out = {}
    for l in open(p):
        r = json.loads(l)
        if not r.get("error"):
            out[(r["qid"], r.get("reader", ""))] = r
    return out


def ours(bench):
    o = L / bench / "ours"
    f = lambda cond, reader, extra="": load(o / f"{cond}__{reader}{extra}.jsonl")
    xc = "__xcache" if bench == "visualmem" else ""
    runs = {"rag_l": f("rag-40__read-medium", "gemini-2.5-flash-lite"),
            "std_l": f("toc38-40__read-medium", "gemini-2.5-flash-lite", xc),
            "wid_l": f("toc38-40__readn-medium", "gemini-2.5-flash-lite", "__snip600" + xc),
            "rag_s": f("rag-40__read-low", "gemini-3.8-flash"),
            "std_s": f("toc38-40__read+toc-low", "gemini-3.8-flash", "__cnoassist+telegraph__xcache") if bench == "memeye"
            else f("toc38-40__read-low", "gemini-3.8-flash") if bench == "memlens"
            else f("toc38-40__read+toc-low", "gemini-3.8-flash", "__xcache")}
    runs = {k: {q: r for (q, _), r in v.items()} for k, v in runs.items()}
    fresh_sel = {}                                   # question -> fresh selection / embedding latency by type
    for k in ("rag_l", "rag_s"):
        for q, r in runs[k].items():
            if r["t_select_s"] >= HIT:
                fresh_sel.setdefault(("emb", q), r["t_select_s"])
    for k in ("std_l", "wid_l", "std_s"):
        for q, r in runs[k].items():
            if r["t_select_s"] >= HIT:
                fresh_sel.setdefault(("sel", q), r["t_select_s"])

    def lat(k, q):
        r = runs[k].get(q)
        if r is None:
            return None
        ts = r["t_select_s"]
        if ts < HIT:
            ts = fresh_sel.get(("emb" if k.startswith("rag") else "sel", q), ts)
        return ts + r["t_read_s"], ts, r

    V = json.load(open(ROOT / ("results/p10/vm_visibility_hat.json" if bench == "visualmem" else "results/p5/visibility_hat.json")))
    out = {"rag": {}, "vgs": {}}
    for reader, rk, sk, wk in (("lite", "rag_l", "std_l", "wid_l"), ("strong", "rag_s", "std_s", None)):
        for q in runs[rk]:
            a = lat(rk, q)
            if a is None or (runs[rk][q].get("warm") and bench != "memlens"):   # MemLens: one history per question
                continue
            out["rag"][(q, reader)] = (a[0], a[1])
            hi = V.get(q, 0) >= TAU
            if reader == "lite":
                b = lat(sk if hi else wk, q)
                if b:
                    out["vgs"][(q, reader)] = (b[0], b[1])
            else:
                if hi:
                    b = lat(sk, q)
                    if b:
                        out["vgs"][(q, reader)] = (b[0], b[1])
                else:                                   # wide retrieval; defer to the light RAG@40 answer on abstention
                    t = a[0]
                    if ABST.search(runs[rk][q]["pred"] or "") and len(runs[rk][q]["pred"] or "") < 400:
                        t += lat("rag_l", q)[0]
                    out["vgs"][(q, reader)] = (t, a[1])
    return out


def baseline(bench, system):
    name = {"mirix": "mirix.jsonl"}.get(system, f"{system}.jsonl")
    R = load(L / bench / name)
    out = {}
    for (q, reader), r in R.items():
        if r.get("warm") and bench != "memlens":
            continue
        if "t_retrieve_s" in r and "t_read_s" in r:
            out[(q, reader)] = (r["t_retrieve_s"] + r["t_read_s"], r["t_retrieve_s"])
    return out


def main():
    S, FIG = {}, {}
    sys.path.insert(0, str(ROOT / "scripts"))
    for bench in ["memeye", "memlens", "visualmem"]:
        D = ours(bench)
        for s in ["simplemem", "omni", "mirix", "vmem", "mma", "telemem"]:
            D[s] = baseline(bench, s)
        for s, X in D.items():
            for reader in ("lite", "strong"):
                v = [t for (q, rd), t in X.items() if rd == reader]
                if not v:
                    continue
                tot, ret = np.array([t[0] for t in v]), np.array([t[1] for t in v])
                S[f"{bench}|{reader}|{s}"] = {"n": len(v), "median": float(np.median(tot)), "mean": float(np.mean(tot)),
                                              "p90": float(np.percentile(tot, 90)), "retrieve_median": float(np.median(ret)),
                                              "read_median": float(np.median(tot - ret))}
                print(f"{bench:9s} {reader:6s} {s:9s} n={len(v):2d} median {np.median(tot):6.2f}s  mean {np.mean(tot):6.2f}s  "
                      f"(retrieval {np.median(ret):5.2f}s)")
    json.dump(S, open(ROOT / "results/p13/latency_summary.json", "w"), indent=1)
    from p11_table import cells                         # accuracy of Table 2 next to the MemEye latency (Figure 5d)
    for reader in ("lite", "strong"):
        C = cells("memeye", reader)
        for s in ["rag", "simplemem", "omni", "mirix", "vmem", "mma", "telemem", "vgs"]:
            k = f"memeye|{reader}|{s}"
            if k in S and C.get(s):
                FIG[f"{s}|{reader}"] = {"lat": S[k]["median"], "acc": C[s]["acc"]}
    json.dump(FIG, open(ROOT / "results/p13/latency_fig.json", "w"), indent=1)


if __name__ == "__main__":
    main()
