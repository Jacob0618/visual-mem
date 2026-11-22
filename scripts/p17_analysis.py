"""Phase 17 analyses (no API calls): long histories, visibility manipulation, selector models, open-weight selectors.
python scripts/p17_analysis.py [long|vis|sel|open|all] -> results/p17/*.json"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p16_compose as P  # noqa: E402
import p16_paper as PP  # noqa: E402
from analyze import boot_ci  # noqa: E402

OUT = ROOT / "results/p17"
P22 = ROOT / "results/p22"
LITE, STRONG = P.LITE, P.STRONG


def L(p):
    R = {}
    for line in open(p):
        r = json.loads(line)
        R[r["qid"]] = r
    if any(r.get("error") for r in R.values()):
        raise SystemExit(f"{p}: failed records")
    return R


def g1(pat):
    fs = sorted(glob.glob(str(pat)))
    return L(fs[0]) if fs else None


def acc(D, qs=None):
    return float(np.mean([D[q]["score"] for q in (qs or sorted(D))]))


def diff(A, B):
    qs = sorted(set(A) & set(B))
    d = np.array([A[q]["score"] - B[q]["score"] for q in qs])
    lo, hi = boot_ci(d)
    return [float(d.mean()), lo, hi, PP.perm_p(d), len(qs)]


def long():
    """MemEye histories grown with distractor sessions: RAG@40 and the standard path of VGS (whole index)."""
    out, _ = P.compose("memeye")
    res = {"0": {"rounds": None, "rag_l": acc(out["rag_l"]), "vgs_l": acc(out["std_l"]), "rag_s": acc(out["rag_s"]), "vgs_s": acc(out["std_s"]),
                 "d_l": diff(out["std_l"], out["rag_l"]), "d_s": diff(out["std_s"], out["rag_s"])}}
    for g in (250, 500, 1000):
        d = OUT / f"long/dx{g}"
        F = {"rag_l": g1(d / f"rag-40__read-medium__{LITE}.jsonl"), "vgs_l": g1(d / f"toc38-40__read-medium__{LITE}*.jsonl"),
             "rag_s": g1(d / f"rag-40__read-low__{STRONG}.jsonl"), "vgs_s": g1(d / f"toc38-40__read+toc-low__{STRONG}*.jsonl")}
        r = {k: (acc(v) if v and len(v) == 371 else None) for k, v in F.items()}
        for rd in ("l", "s"):
            if r[f"rag_{rd}"] is not None and r[f"vgs_{rd}"] is not None:
                r[f"d_{rd}"] = diff(F[f"vgs_{rd}"], F[f"rag_{rd}"])
                r[f"fsr_{rd}"] = [float(np.mean([x["full"] for x in F[k].values() if x.get("n_gold")])) for k in (f"rag_{rd}", f"vgs_{rd}")]
                r[f"usd_{rd}"] = [1000 * float(np.mean([x.get("usd", 0) for x in F[k].values()])) for k in (f"rag_{rd}", f"vgs_{rd}")]
                r[f"tok_{rd}"] = float(np.mean([x.get("sel_in_tok", x.get("in_tok", 0)) for x in F[f"vgs_{rd}"].values()]))
        res[str(g)] = r
    try:
        sys.path.insert(0, str(ROOT / "results/p16"))
        from load import load
        import scale
        eps = load("memeye")
        for g in ("0", "250", "500", "1000"):
            G = [scale.grow(e, int(g)) for e in eps]      # averaged over questions, as the history sizes of the setup section
            res[g]["rounds"] = float(sum(len(e.rounds) * len(e.qas) for e in G) / sum(len(e.qas) for e in G))
    except Exception as ex:  # noqa: BLE001
        print("rounds not computed:", repr(ex)[:100])
    json.dump(res, open(OUT / "long.json", "w"), indent=1)
    for g, r in res.items():
        print("long +%s images" % g, {k: ([round(x, 3) if isinstance(x, float) else x for x in v] if isinstance(v, list) else (round(v, 3) if isinstance(v, float) else v)) for k, v in r.items()})
    return res


def vis():
    """Index visibility manipulated by the line length (lightweight reader)."""
    sys.path.insert(0, str(ROOT / "results/p16"))
    from load import captions, load
    import mm2
    import p16_hvis as HV
    res = {}
    for bench, name, snips, pat in [("memeye", "memeye", (30, 60, 200), OUT / "vis/memeye"), ("memlens", "memlens_test", (200, 600, 2000), OUT / "vis/memlens")]:
        out, _ = P.compose(bench)
        eps = load(name)
        caps = captions(eps)[0]
        mm2.NO_ASSIST = mm2.SHORT_IDS = mm2.TELEGRAPH = True
        rows = {}
        for s in snips:
            vals = []
            for e in eps:
                mm2.SNIPPET = s
                v = []
                for r in e.rounds:
                    c = {p: caps.get(p, "") for p in r.images}
                    full = set(mm2._words(r.user + " " + (r.assistant or "") + " " + " ".join(c.values())))
                    if full:
                        v.append(len(full & set(mm2._words(mm2.toc_line(r, c, e)))) / len(full))
                vals.append(float(np.mean(v)) if v else 1.0)
            D = out["std_l"] if s == 200 else g1(pat / f"toc38-40__read-medium__{LITE}__snip{s}__*.jsonl")
            rows[str(s)] = {"H": float(np.mean(vals)), "acc": acc(D) if D else None,
                            "fsr": float(np.mean([x["full"] for x in D.values() if x.get("n_gold")])) if D else None,
                            "rounds": float(np.mean([x["n_loaded"] for x in D.values()])) if D else None}
        mm2.NO_ASSIST = mm2.SHORT_IDS = mm2.TELEGRAPH = False
        mm2.SNIPPET = 200
        rows["rag"] = acc(out["rag_l"]); rows["widened"] = acc(out["wid_l"])
        if bench == "memlens":
            D = g1(pat / f"toc38-40__readn-medium__{LITE}__cnoassist*.jsonl")
            rows["notes200"] = acc(D) if D else None
        res[bench] = rows
        print(bench, json.dumps({k: ({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()} if isinstance(v, dict) else (round(v, 3) if isinstance(v, float) else v)) for k, v in rows.items()}))
    json.dump(res, open(OUT / "vis.json", "w"), indent=1)
    return res


SELECTORS = [("gemini-3.8-flash", "toc38-40", "Gemini"), ("gemini-2.5-flash", "toc@2.5f-40", "Gemini"), ("gemini-2.5-flash-lite", "toc@2.5fl-40", "Gemini"),
             ("GPT-5-mini", "toc@gpt5m-40", "OpenAI"), ("GPT-4.1-mini", "toc@gpt41m-40", "OpenAI"), ("DeepSeek-V3.1", "toc@ds-40", "open weights"),
             ("Qwen3-235B (thinking)", "n150_toc@qwent-40", "open weights"), ("Qwen3-235B", "toc@qwen-40", "open weights")]


def sel():
    """Selector models on the compact index (MemEye, selection only)."""
    res = {}
    for name, key, fam in SELECTORS:
        f = ROOT / f"results/mm2/select_eval_{key}_cnoassist+shortid+telegraph.json"
        if key == "toc38-40":                       # the selections of the paper's standard path (regenerated descriptions)
            D = P.compose("memeye")[0]["std_l"]
            R = [{"full": r["full"], "n": r["n_loaded"], "n_gold": r.get("n_gold"), "usd": r.get("sel_usd", 0.0)} for r in D.values()]
            RK = PP.ranks("memeye")                 # cosine retrieval with as many rounds as each selection (as in Table 2)
            cos = float(np.mean([all(rk <= D[q]["n_loaded"] for rk in RK[q]["gold"].values()) for q in D if RK[q]["gold"]]))
        elif key in ("toc@qwen-40", "n150_toc@qwent-40") and (key == "toc@qwen-40" or g1(P22 / "sel/toc@qwent-40__read-medium__*.jsonl")):
            D = g1(ROOT / "results/p16/open/memeye/*.jsonl") if key == "toc@qwen-40" else g1(P22 / "sel/toc@qwent-40__read-medium__*.jsonl")
            if len(D) < 371:                         # end-to-end run not complete yet
                continue
            R = [{"full": r["full"], "n": r["n_loaded"], "n_gold": r.get("n_gold"), "usd": r.get("usd_select") or 0.0} for r in D.values()]
            RK = PP.ranks("memeye")
            cos = float(np.mean([all(rk <= D[q]["n_loaded"] for rk in RK[q]["gold"].values()) for q in D if RK[q]["gold"]]))
        elif f.exists():
            R = json.load(open(f))["per_q"]
            cos = float(np.mean([r["rag_full_same_k"] for r in R if r.get("n_gold")]))
        else:
            continue
        g = [r for r in R if r.get("n_gold")]
        res[name] = {"family": fam, "n": len(R), "fsr": float(np.mean([r["full"] for r in g])), "rounds": float(np.mean([r["n"] for r in R])),
                     "cos_same_k": cos, "empty": int(sum(1 for r in R if r["n"] == 0)), "usd_1k": 1000 * float(np.mean([r.get("usd", 0.0) for r in R]))}
        print(name.ljust(24), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items()})
    json.dump(res, open(OUT / "selectors.json", "w"), indent=1)
    return res


def open_weight():
    """End-to-end lightweight-reader accuracy with open-weight selectors on the path chosen by the gate."""
    res = {}
    for tag, base in (("Qwen3-235B", ROOT / "results/p16/open"), ("DeepSeek-V3.1", OUT / "open_ds")):
        for b in ("memeye", "memlens", "visualmem"):
            D = g1(base / b / "*.jsonl")
            out, _ = P.compose(b)
            if not D or len(D) != len(out["rag_l"]):
                continue
            res[f"{tag}|{b}"] = {"acc": acc(D), "rag": acc(out["rag_l"]), "vgs": acc(out["vgs_l"]), "d": diff(D, out["rag_l"]),
                                 "fsr": float(np.mean([x["full"] for x in D.values() if x.get("n_gold")])), "rounds": float(np.mean([x["n_loaded"] for x in D.values()]))}
            print(tag, b, {k: ([round(x, 3) if isinstance(x, float) else x for x in v] if isinstance(v, list) else round(v, 3)) for k, v in res[f"{tag}|{b}"].items()})
    json.dump(res, open(OUT / "open_weight.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("long", long), ("vis", vis), ("sel", sel), ("open", open_weight)):
        if what in (name, "all"):
            fn()
