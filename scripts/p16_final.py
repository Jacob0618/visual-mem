"""Phase 16 (final revision): tables and figures of the ACL paper that depend on the rebuilt memory-system baselines, the
history-level gate and the added controls. No API calls; every input is a stored per-question file.
python scripts/p16_final.py [check|main|figs|appendix|all]"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p16_compose as P  # noqa: E402
import p16_paper as PP  # noqa: E402
from analyze import boot_ci  # noqa: E402

OUT = ROOT / "results/p16"
POL = ROOT / "paper_acl/sections_polished"
FIGP = ROOT / "paper_acl/figures_polished"
BENCH = ["memeye", "memlens", "visualmem"]
NQ = {"memeye": 371, "memlens": 466, "visualmem": 696}
SYSTEMS = [("SimpleMem", "simplemem"), ("Omni-SimpleMem", "omni"), ("TeleMem", "telemem"), ("V-Mem", "vmem"), ("MMA", "mma")]
LITE, STRONG = "gemini-2.5-flash-lite", "gemini-3.8-flash"
C = "cnoassist+shortid+telegraph"
RANK = {"memeye": f"select_eval_rankr38-100_{C}.json", "visualmem": f"select_eval_visualmem_rankr38-100_{C}.json",
        "memlens": f"select_eval_memlens32k_memlens_test_rankr38-100_{C}.json"}
f3 = PP.f3


def L(p):
    p = Path(p)
    if not p.exists():
        return None
    R = {}
    for line in open(p):
        r = json.loads(line)
        R[r["qid"]] = r
    if any(r.get("error") for r in R.values()):
        raise SystemExit(f"{p}: failed records")
    return R


# ------------------------------------------------------------------------------------------------ inputs
def bl(bench, system, reader):
    """Rebuilt memory-system baseline: {qid: record} of one reader, or None while incomplete."""
    p = OUT / "baselines" / bench / f"{system}.jsonl"
    if not p.exists():
        return None
    R = {}
    for line in open(p):
        r = json.loads(line)
        if r.get("error") or r.get("score") is None:
            return None
        if r["reader"] == reader:
            R[r["qid"]] = r
    return R if len(R) == NQ[bench] else None


def ledger(bench, system):
    """Construction and query-time spend of a baseline on a benchmark (USD), from its ledger (fresh calls only; calls of
    the MemLens smoke tests were issued once and are added)."""
    tot = 0.0
    for p in [OUT / "budget" / bench / f"{system}.jsonl"] + ([OUT / "smoke/budget" / f"{system}.jsonl"] if bench == "memlens" else []):
        if p.exists():
            for line in open(p):
                r = json.loads(line)
                if r["what"].split(":")[0] in ("build", "query"):
                    tot += r["usd"]
    return tot


def fc(bench, reader):
    if bench == "memeye":
        return L(OUT / f"final2/memeye/fc__read-{'medium' if reader == 'lite' else 'low'}__{LITE if reader == 'lite' else STRONG}.jsonl")
    if bench == "visualmem":
        return L(ROOT / f"results/p14/runs/visualmem/fc__read-{'medium' if reader == 'lite' else 'low'}__{LITE if reader == 'lite' else STRONG}.jsonl")
    test = set(json.load(open(ROOT / "results/memlens_test.json")))
    if reader == "lite":
        D = L(ROOT / f"results/mm2/memlens_32k/fc__read-medium__{LITE}.jsonl")
        return {q: r for q, r in D.items() if q in test}
    A = L(ROOT / f"results/p14/runs/memlens_fc/fc__read-low__{STRONG}.jsonl") or {}
    B = L(OUT / f"final/memlens_fc/fc__read-low__{STRONG}.jsonl") or {}
    D = {**A, **B}
    return {q: r for q, r in D.items() if q in test}


def rerank(bench, reader):
    D = L(OUT / f"final/{bench}/ret_rerank40_{reader}.jsonl")
    if not D:
        return None
    usd = {r["qid"]: r["usd"] for r in json.load(open(ROOT / "results/mm2" / RANK[bench]))["per_q"]}
    return {q: dict(r, usd=r.get("usd", 0.0) + usd.get(q, 0.0)) for q, r in D.items()}


def cells(bench, reader):
    """method -> {acc, usd, d, lo, hi, p, n, D} on the benchmark's full question set (None while a run is incomplete)."""
    out, _ = P.compose(bench)
    rk, vk = ("rag_l", "vgs_l") if reader == "lite" else ("rag_s", "vgs_s")
    rag = out[rk]
    add = P.CACHE_USD.get((bench, reader), 0.0) / 1000
    X = {"rag": rag, "rerank": rerank(bench, reader), "fc": fc(bench, reader),
         "vgs": {q: dict(r, usd=r.get("usd", 0.0) + add) for q, r in out[vk].items()}}
    for _, s in SYSTEMS:
        D = bl(bench, s, reader)
        if D:
            share = ledger(bench, s) / NQ[bench]
            D = {q: dict(r, usd=r.get("usd_read", 0.0) + share) for q, r in D.items()}
        X[s] = D
    res = {}
    for k, D in X.items():
        if not D or len(D) != len(rag) or set(D) != set(rag):
            res[k] = None
            continue
        qs = sorted(rag)
        a = np.array([D[q]["score"] for q in qs]); d = a - np.array([rag[q]["score"] for q in qs])
        lo, hi = boot_ci(d)
        res[k] = {"acc": float(a.mean()), "usd": 1000 * float(np.mean([D[q].get("usd", 0.0) for q in qs])), "d": float(d.mean()),
                  "lo": lo, "hi": hi, "p": PP.perm_p(d) if k != "rag" else 1.0, "n": len(qs), "D": D}
    return res


ROWS = [[("RAG@40", "rag"), ("\\quad + LLM rerank", "rerank"), ("Full context", "fc")],
        [("SimpleMem", "simplemem"), ("Omni-SimpleMem", "omni"), ("TeleMem", "telemem")],
        [("V-Mem", "vmem"), ("MMA", "mma")], [("\\textbf{\\vgs{} (ours)}", "vgs")]]


def check():
    """Completeness of every input (run before writing tables)."""
    ok = True
    for b in BENCH:
        for r in ("lite", "strong"):
            Cc = cells(b, r)
            miss = [k for k, v in Cc.items() if v is None]
            ok &= not miss
            print(f"{b:9s} {r:6s} " + " ".join(f"{k}={v['acc']:.3f}" for k, v in Cc.items() if v) + (f" | MISSING: {miss}" if miss else ""))
    return ok


def money(a, b):
    f = lambda x: "--" if x is None else (f"{x:.0f}" if x >= 100 else f"{x:.1f}")  # noqa: E731
    return f"{f(a)}\\,/\\,{f(b)}"


def main_table():
    T = {(b, r): cells(b, r) for b in BENCH for r in ("lite", "strong")}
    H = {"memeye": ("hME", "MemEye"), "memlens": ("hML", "MemLens-32k"), "visualmem": ("hVM", "VisualMem")}
    Ls = ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4.6pt}", "\\renewcommand{\\arraystretch}{1.06}",
          "\\begin{tabularx}{\\linewidth}{l|ccY|ccY|ccY}", "\\toprule",
          "\\multirow{2}{*}{\\textbf{Method}} & " + " & ".join(
              f"\\multicolumn{{3}}{{c{'|' if b != 'visualmem' else ''}}}{{\\cellcolor{{{H[b][0]}}}\\textbf{{{H[b][1]}}}}}" for b in BENCH) + " \\\\",
          " & " + " & ".join(f"\\cellcolor{{{H[b][0]}}}Light & \\cellcolor{{{H[b][0]}}}Strong & \\cellcolor{{{H[b][0]}}}\\$/1k" for b in BENCH) + " \\\\",
          "\\midrule"]
    best = {k: max(v["acc"] for v in T[k].values() if v) for k in T}
    for gi, g in enumerate(ROWS):
        if gi:
            Ls.append("\\midrule")
        for name, key in g:
            cs = []
            for b in BENCH:
                for r in ("lite", "strong"):
                    c = T[(b, r)][key]
                    if c is None:
                        cs.append("--"); continue
                    t = f3(c["acc"])
                    t = f"\\textbf{{{t}}}" if abs(c["acc"] - best[(b, r)]) < 5e-4 else t
                    cs.append(t)
                cs.append(money(*(T[(b, r)][key]["usd"] if T[(b, r)][key] else None for r in ("lite", "strong"))))
            Ls.append(("\\oursrow " if key == "vgs" else "") + name + " & " + " & ".join(cs) + " \\\\")
    Ls += ["\\bottomrule", "\\end{tabularx}",
           "\\caption{\\textbf{Main results.} Accuracy and total cost in USD per 1,000 questions (lightweight\\,/\\,strong reader), "
           "including memory construction and caches. Every method is evaluated on the same questions (371, 466, and 696) with the "
           "same readers, prompts, and judges.}",
           "\\label{tab:main}", "\\end{table*}"]
    (POL / "tab_main.tex").write_text("\n".join(Ls) + "\n")
    json.dump({f"{b}|{r}|{k}": {kk: vv for kk, vv in v.items() if kk != "D"} for (b, r), Cc in T.items() for k, v in Cc.items() if v},
              open(OUT / "main_cells.json", "w"), indent=1)
    print("main table written")
    return T


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    if what == "check":
        check()
    if what in ("main", "all"):
        main_table()


# ------------------------------------------------------------------------------------------------ appendix tables
def _g(pattern):
    import glob
    fs = sorted(glob.glob(str(pattern)))
    return L(fs[0]) if fs else None


def _acc(D, qs=None):
    qs = qs or sorted(D)
    return float(np.mean([D[q]["score"] for q in qs]))


def gate_table(S):
    g = ["\\begin{table*}[t]", "\\centering", "\\small", "\\begin{tabularx}{\\linewidth}{X|ccc|ccc}", "\\toprule",
         "\\rowcolor{hHead} & \\multicolumn{3}{c|}{\\textbf{Lightweight reader}} & \\multicolumn{3}{c}{\\textbf{Strong reader}} \\\\",
         "\\rowcolor{hHead} \\textbf{Policy} & MemEye & MemLens & VisualMem & MemEye & MemLens & VisualMem \\\\", "\\midrule"]
    row = lambda lab, k, fmt="{:.3f}": lab + " & " + " & ".join(fmt.format(S[b][r][k]) for r in ("lite", "strong") for b in BENCH) + " \\\\"  # noqa: E731
    g += [row("Standard path for every history", "std"), row("Widened path for every history", "wid"),
          "\\oursrow " + row("\\vgs{}: history visibility $H(h)\\ge0.45$", "vgs"), "\\midrule",
          row("Per-question estimate $\\hat v(q)\\ge0.7$", "qgate"), row("Per-question message length", "length"),
          row("Per-question random routing", "random"), row("Per-question oracle", "oracle"),
          row("AUC of $\\hat v(q)$", "auc", "{:.2f}"), "\\bottomrule", "\\end{tabularx}",
          "\\caption{\\textbf{Routing policies.} Accuracy when every history takes one path, with the history-level gate of "
          "\\vgs{}, and with per-question routers. Random routing uses the path shares of the per-question estimate; the oracle "
          "takes the better path for each question; AUC measures how well $\\hat v(q)$ ranks the questions that gain from widening.}",
          "\\label{tab:gate}", "\\end{table*}"]
    (POL / "tab_gate.tex").write_text("\n".join(g) + "\n")


def swap_table(S):
    rows = {}
    for b in BENCH:
        out, _ = P.compose(b)
        sn = _g(OUT / f"swap/{b}/toc38-40__readn-low__{STRONG}*.jsonl"); lu = _g(OUT / f"swap/{b}/tocu38-40-40__read-medium__{LITE}*.jsonl")
        rows[b] = {"l_notes": _acc(out["wid_l"]), "l_union": _acc(lu), "s_union": _acc(out["uni_s"]), "s_notes": _acc(sn),
                   "l_std": _acc(out["std_l"]), "s_std": _acc(out["std_s"])}
    json.dump(rows, open(OUT / "swap.json", "w"), indent=1)
    b2 = lambda x, y: (f"\\textbf{{{f3(x)}}}" if x >= y else f3(x))  # noqa: E731
    t = ["\\begin{table}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{2.2pt}", "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{llccc}", "\\toprule",
         "\\rowcolor{hHead} Reader & Widened path & MemEye & MemLens & VisualMem \\\\", "\\midrule",
         "\\multirow{2}{*}{Light} & notes + selected rounds & " + " & ".join(b2(rows[b]["l_notes"], rows[b]["l_union"]) for b in BENCH) + " \\\\",
         " & selected $\\cup$ top-40 rounds & " + " & ".join(b2(rows[b]["l_union"], rows[b]["l_notes"]) for b in BENCH) + " \\\\", "\\midrule",
         "\\multirow{2}{*}{Strong} & selected $\\cup$ top-40 rounds & " + " & ".join(b2(rows[b]["s_union"], rows[b]["s_notes"]) for b in BENCH) + " \\\\",
         " & notes + selected rounds & " + " & ".join(b2(rows[b]["s_notes"], rows[b]["s_union"]) for b in BENCH) + " \\\\",
         "\\bottomrule", "\\end{tabular}}",
         "\\caption{\\textbf{Swapping the widened paths.} Accuracy when each reader uses its own widened path (first row of each "
         "block) or the other reader's, applied to every question.}", "\\label{tab:swap}", "\\end{table}"]
    (POL / "tab_swap.tex").write_text("\n".join(t) + "\n")
    return rows


def retrieval_table(T):
    R = {}
    for b in BENCH:
        out, _ = P.compose(b)
        R[b] = {f"rag{k}_l": _acc(_g(OUT / f"swap/{b}/rag-{k}__read-medium__{LITE}.jsonl")) for k in (10, 20, 30)}
        R[b]["rag40_l"], R[b]["rag40_s"] = T[(b, "lite")]["rag"]["acc"], T[(b, "strong")]["rag"]["acc"]
        R[b]["rag30_s"] = _acc(L(OUT / f"fig1/{b}_rag30_strong.jsonl"))
        R[b]["hyb_l"] = _acc(L(OUT / f"final/{b}/ret_hybrid40_lite.jsonl"))
        R[b]["rer_l"], R[b]["rer_s"] = T[(b, "lite")]["rerank"]["acc"], T[(b, "strong")]["rerank"]["acc"]
        R[b]["vgs_l"], R[b]["vgs_s"] = T[(b, "lite")]["vgs"]["acc"], T[(b, "strong")]["vgs"]["acc"]
    json.dump(R, open(OUT / "retrieval_baselines.json", "w"), indent=1)
    def line(lab, kl, ks, ours=False):
        cs = []
        for b in BENCH:
            cs += [f3(R[b][kl]) if kl else "--", f3(R[b][ks]) if ks else "--"]
        return ("\\oursrow " if ours else "") + lab + " & " + " & ".join(cs) + " \\\\"
    t = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\columnwidth}{!}{%",
         "\\begin{tabular}{l|cc|cc|cc}", "\\toprule",
         "\\multirow{2}{*}{\\textbf{Retrieval}} & \\multicolumn{2}{c|}{\\cellcolor{hME}\\textbf{MemEye}} & \\multicolumn{2}{c|}{\\cellcolor{hML}\\textbf{MemLens}} & \\multicolumn{2}{c}{\\cellcolor{hVM}\\textbf{VisualMem}} \\\\",
         " & \\cellcolor{hME}L & \\cellcolor{hME}S & \\cellcolor{hML}L & \\cellcolor{hML}S & \\cellcolor{hVM}L & \\cellcolor{hVM}S \\\\", "\\midrule",
         line("Dense top-10", "rag10_l", None), line("Dense top-20", "rag20_l", None), line("Dense top-30", "rag30_l", "rag30_s"),
         line("Dense top-40 (RAG@40)", "rag40_l", "rag40_s"), line("BM25 + dense (RRF), top-40", "hyb_l", None),
         line("Dense top-100 $\\rightarrow$ LLM rerank, top-40", "rer_l", "rer_s"), "\\midrule", line("\\vgs{}", "vgs_l", "vgs_s", True),
         "\\bottomrule", "\\end{tabular}}",
         "\\caption{\\textbf{Retrieval baselines.} Accuracy with the lightweight (L) and strong (S) reader. The reranker is the "
         "selector model with the selector instruction, applied to the 100 most similar rounds.}", "\\label{tab:retrieval}", "\\end{table}"]
    (POL / "tab_retrieval.tex").write_text("\n".join(t) + "\n")
    return R


def std_table(T):
    from p13_metrics import f1, bleu1, rank_metrics, ids, gold_rounds
    names = [("RAG@40", "rag"), *SYSTEMS, ("\\vgs{} (ours)", "vgs")]
    M = collections.defaultdict(dict)
    for b in ("memeye", "memlens"):
        gold = {q: r["gold"] for q, r in T[(b, "lite")]["rag"]["D"].items()}
        for _, k in names:
            for r in ("lite", "strong"):
                c = T[(b, r)][k]
                if c:
                    M[k][(b, r, "f1")] = float(np.mean([f1(c["D"][q]["pred"], str(gold[q])) for q in gold]))
                    M[k][(b, r, "bleu")] = float(np.mean([bleu1(c["D"][q]["pred"], str(gold[q])) for q in gold]))
    for b in BENCH:
        G = gold_rounds(b)
        for k in ("rag", "vmem", "mma", "vgs"):
            c = T[(b, "lite")][k]
            if c:
                ms = [rank_metrics(ids(c["D"][q]), G[q]) for q in c["D"] if G.get(q)]
                ms = [m for m in ms if m]
                M[k][(b, "r10")] = float(np.mean([m["r10"] for m in ms])); M[k][(b, "ndcg")] = float(np.mean([m["ndcg10"] for m in ms]))
    json.dump({k: {"|".join(kk): v for kk, v in d.items()} for k, d in M.items()}, open(OUT / "std_metrics.json", "w"), indent=1)
    def best(keys, cell):
        vals = [M[k][cell] for _, k in keys if cell in M[k]]
        return max(vals) if vals else None
    A = ["\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{2.4pt}", "\\begin{tabular}[t]{l|cc|cc}", "\\toprule",
         "\\rowcolor{hHead} & \\multicolumn{2}{c|}{\\textbf{Token F1}} & \\multicolumn{2}{c}{\\textbf{BLEU-1}} \\\\",
         "\\rowcolor{hHead} \\textbf{Method} & MemEye & MemLens & MemEye & MemLens \\\\", "\\midrule"]
    for name, k in names:
        if not M[k]:
            continue
        cs = []
        for m in ("f1", "bleu"):
            for b in ("memeye", "memlens"):
                pr = []
                for r in ("lite", "strong"):
                    v = M[k].get((b, r, m)); bb = best(names, (b, r, m))
                    pr.append("--" if v is None else (f"\\textbf{{{f3(v)}}}" if abs(v - bb) < 5e-4 else f3(v)))
                cs.append(" / ".join(pr))
        A.append(("\\midrule\n\\oursrow " if k == "vgs" else "") + name + " & " + " & ".join(cs) + " \\\\")
    A += ["\\bottomrule", "\\end{tabular}\\hfill", "\\begin{tabular}[t]{l|ccc|ccc}", "\\toprule",
          "\\rowcolor{hHead} & \\multicolumn{3}{c|}{\\textbf{Recall@10}} & \\multicolumn{3}{c}{\\textbf{nDCG@10}} \\\\",
          "\\rowcolor{hHead} \\textbf{Method} & ME & ML & VM & ME & ML & VM \\\\", "\\midrule"]
    ev = [("RAG@40", "rag"), ("V-Mem", "vmem"), ("MMA", "mma"), ("\\vgs{} (ours)", "vgs")]
    for name, k in ev:
        if not M[k]:
            continue
        cs = []
        for m in ("r10", "ndcg"):
            for b in BENCH:
                v = M[k].get((b, m)); bb = best(ev, (b, m))
                cs.append("--" if v is None else (f"\\textbf{{{f3(v)}}}" if abs(v - bb) < 5e-4 else f3(v)))
        A.append(("\\midrule\n\\oursrow " if k == "vgs" else "") + name + " & " + " & ".join(cs) + " \\\\")
    A += ["\\bottomrule", "\\end{tabular}",
          "\\caption{\\textbf{Standard metrics on MemEye, MemLens, and VisualMem.} Left, answer metrics for lightweight / strong "
          "readers. Right, evidence metrics for methods returning raw rounds, using the lightweight path for \\vgs{}. Definitions are in "
          "Appendix~\\ref{app:latency}.}", "\\label{tab:std}", "\\end{table*}"]
    (POL / "tab_std.tex").write_text("\n".join(A) + "\n")
    return M


def latency():
    """Median online latency: VGS on the path chosen by the history gate; V-Mem / MMA at the budgets of this paper."""
    import glob
    res = {}
    for b in BENCH:
        std_route = next(iter(P.routes(b).values()))
        pat = {("lite", True): f"toc38-40__read-medium__{LITE}*", ("lite", False): f"toc38-40__readn-medium__{LITE}*",
               ("strong", True): f"toc38-40__read*{STRONG}*", ("strong", False): f"tocu-40-40__read*{STRONG}*"}
        for r in ("lite", "strong"):
            D = _g(OUT / f"latency/{b}/{pat[(r, std_route)]}.jsonl")
            xs = [x for x in D.values() if x.get("warm", True)]
            res[f"{b}|{r}|vgs"] = [float(np.median([x["t_select_s"] + x["t_read_s"] for x in xs])), float(np.median([x["t_select_s"] for x in xs])), len(xs)]
        bench_is_memlens = b == "memlens"
        for s in ("vmem", "mma"):
            p = OUT / f"latency_bl/{b}/{s}.jsonl"
            if not p.exists():
                continue
            R = [json.loads(x) for x in open(p)]
            for r in ("lite", "strong"):
                # every MemLens question has its own history, so there is no cache-creating first question to exclude
                xs = [x for x in R if x.get("reader") == r and not x.get("error") and (bench_is_memlens or not x.get("warm"))]
                if xs:
                    res[f"{b}|{r}|{s}"] = [float(np.median([x["t_retrieve_s"] + x["t_read_s"] for x in xs])), float(np.median([x["t_retrieve_s"] for x in xs])), len(xs)]
    json.dump(res, open(OUT / "latency_final.json", "w"), indent=1)
    t = (POL / "tab_latency.tex").read_text().split("\n")
    out = []
    for l in t:
        if l.startswith("MIRIX"):
            continue
        for lab, s in (("V-Mem &", "vmem"), ("MMA &", "mma"), ("\\oursrow \\vgs{} (ours) &", "vgs")):
            if l.startswith(lab) and all(f"{b}|lite|{s}" in res for b in BENCH):
                c = l.split("&")
                v = [f"{res[f'{b}|lite|{s}'][0]:.1f} / {res[f'{b}|strong|{s}'][0]:.1f} & {res[f'{b}|lite|{s}'][1]:.1f}" for b in BENCH]
                l = lab + " " + " & ".join(v) + " &" + "&".join(c[7:])
        out.append(l)
    (POL / "tab_latency.tex").write_text("\n".join(out))
    return res


def systems_table():
    rows = [("SimpleMem (0.3.0)", "factual statements from windows of 40 turns, with resolved references and times", "planning, reflection, hybrid search", "--", "punctuation removed from keyword queries"),
            ("Omni-SimpleMem (0.3.0)", "multimodal atomic units, knowledge graph; images as on-demand units", "hybrid search, graph expansion (20 units)", "unit summaries (2.5-flash-lite)", "no modality filter for questions about images"),
            ("TeleMem (1.10.1)", "one memory per message, for the user profile and for events", "top 30 per scope", "extraction (2.5-flash-lite)", "English extraction prompts"),
            ("V-Mem", "raw user, assistant, and image units of every round", "modality routing (20 rounds and neighbours)", "anchors (2.5-flash-lite)", "--"),
            ("MMA", "raw turns with a confidence score", "similarity $\\times$ confidence (80 turns)", "--", "--")]
    t = ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3.4pt}",
         "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}p{2.6cm}|>{\\raggedright\\arraybackslash}p{3.8cm}|>{\\raggedright\\arraybackslash}p{3.0cm}|>{\\raggedright\\arraybackslash}p{2.5cm}|>{\\raggedright\\arraybackslash}X}",
         "\\toprule", "\\rowcolor{hHead} \\textbf{System} & \\textbf{Memory written at ingestion} & \\textbf{Retrieval (budget)} & \\textbf{Other models} & \\textbf{Adaptation} \\\\", "\\midrule"]
    t += [" & ".join(r) + " \\\\" for r in rows]
    t += ["\\bottomrule", "\\end{tabularx}",
          "\\caption{\\textbf{External memory systems.} ``Other models'' lists roles that do not use \\texttt{gemini-3.8-flash}; ``Adaptation'' lists the changes needed to run each release on our data.}",
          "\\label{tab:systems}", "\\end{table*}"]
    return "\n".join(t)


# ------------------------------------------------------------------------------------------------ figures
def figures(T):
    import shutil
    import matplotlib.pyplot as plt
    import make_figures_p13 as M
    from matplotlib.lines import Line2D
    METH = [("rag", "RAG@40", M.BLUE, "o"), ("rerank", "+ LLM rerank", M.BLUE, "d"), ("fc", "Full context", M.DARK, "s"),
            ("simplemem", "SimpleMem", M.ORANGE, "^"), ("omni", "Omni-SimpleMem", M.OLIVE, "D"), ("telemem", "TeleMem", M.GRAY, "h"),
            ("vmem", "V-Mem", M.TEAL, "v"), ("mma", "MMA", M.GREEN, "P"), ("vgs", "VGS (ours)", M.RED, "*")]
    col = {"lite": M.BLUE, "strong": M.RED}
    # ---- accuracy vs cost (a-c) and latency on MemEye (d)
    lat_old = json.load(open(ROOT / "results/p13/latency_fig.json")); LF = latency()
    fig, ax = M.row(4, h=1.32)
    for a, (b, title) in zip(ax, [("memeye", "(a) MemEye"), ("memlens", "(b) MemLens-32k"), ("visualmem", "(c) VisualMem")]):
        for r in ("lite", "strong"):
            for k, _, _, m in METH:
                v = T[(b, r)][k]
                if v:
                    a.scatter(v["usd"], v["acc"], color=col[r], marker=m, s=46 if k == "vgs" else 17, edgecolor="black", linewidth=0.45, zorder=3)
        a.set_xscale("log"); a.set_xlim(0.2, 200); a.set_xticks([1, 10, 100], ["1", "10", "100"]); a.minorticks_off()
        a.set_xlabel("USD per 1k questions"); a.set_title(title)
    ax[0].set_ylabel("Accuracy")
    a = ax[3]
    for r in ("lite", "strong"):
        for k, _, _, m in METH:
            v = T[("memeye", r)][k]
            lt = LF.get(f"memeye|{r}|{k}", [None])[0] if k in ("vgs", "vmem", "mma") else (lat_old.get(f"{k}|{r}") or {}).get("lat")
            if v and lt:
                a.scatter(lt, v["acc"], color=col[r], marker=m, s=46 if k == "vgs" else 17, edgecolor="black", linewidth=0.45, zorder=3)
    a.set_xscale("log"); a.minorticks_off(); a.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    a.set_xlabel("Seconds per question (median)"); a.set_title("(d) Latency (MemEye)")
    hs = [Line2D([], [], color="#bbbbbb", marker=m, linestyle="", markersize=5.5 if k == "vgs" else 4, label=lab, **M.EDGE) for k, lab, _, m in METH]
    hs += [Line2D([], [], color=col[r], marker="s", linestyle="", markersize=4, label=lab, **M.EDGE) for r, lab in (("lite", "light reader"), ("strong", "strong reader"))]
    M.finish(fig, "cost", hs, ncol=11, legend_h=0.12); shutil.copy(M.FIG / "cost.pdf", FIGP / "cost.pdf"); plt.close(fig)
    # ---- paired differences of every method from RAG@40
    fig, ax = M.row(6, h=1.75)
    order = [m for m in METH if m[0] != "rag"]
    for a, (b, r, title) in zip(ax, [(b, r, f"{t}, {rl}") for b, t in [("memeye", "MemEye"), ("memlens", "MemLens"), ("visualmem", "VisualMem")] for r, rl in [("lite", "light"), ("strong", "strong")]]):
        y = np.arange(len(order))[::-1]
        for yi, (k, lab, c, mk) in zip(y, order):
            v = T[(b, r)][k]
            if v:
                a.plot([v["lo"], v["hi"]], [yi, yi], color=c, linewidth=1.2)
                a.plot([v["d"]], [yi], marker=mk, color=c, markersize=4.4 if k == "vgs" else 3.4, **M.EDGE)
        a.axvline(0, color="black", linewidth=0.6)
        a.set_yticks(y, [o[1] for o in order] if a is ax[0] else [], fontsize=5.6); a.set_title(title, fontsize=6.6)
        a.tick_params(axis="x", labelsize=5.4); a.set_xlim(-0.6, 0.25); a.set_xticks([-0.4, -0.2, 0, 0.2])
    fig.supxlabel("Accuracy minus RAG@40 with paired 95% bootstrap interval", fontsize=6.4, y=0.02)
    M.finish(fig, "app_baselines"); shutil.copy(M.FIG / "app_baselines.pdf", FIGP / "app_baselines.pdf"); plt.close(fig)
    print("figures written (cost, app_baselines)")


def appendix(T=None):
    T = T or {(b, r): cells(b, r) for b in BENCH for r in ("lite", "strong")}
    S = json.load(open(OUT / "paper_stats.json"))
    gate_table(S)
    sw = swap_table(S)
    R = retrieval_table(T)
    std_table(T)
    (POL / "tab_systems.tex").write_text(systems_table() + "\n")
    print("appendix tables written;", "swap", {b: {k: round(v, 3) for k, v in d.items()} for b, d in sw.items()})
    print("retrieval", {b: {k: round(v, 3) for k, v in d.items()} for b, d in R.items()})
    return T


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("appendix", "figs", "all"):
    T_ = {(b, r): cells(b, r) for b in BENCH for r in ("lite", "strong")}
    if sys.argv[1] in ("appendix", "all"):
        appendix(T_)
    if sys.argv[1] in ("figs", "all"):
        figures(T_)


def memlens_table():
    """Table 2: MemLens test split (466 questions) with 32k and 64k histories; * = permutation p < 0.05."""
    S = json.load(open(OUT / "paper_stats.json"))
    rows = []
    for reader, lab in (("lite", "Light"), ("strong", "Strong")):
        for i, (b, h) in enumerate((("memlens", "32k"), ("memlens64", "64k"))):
            r = S[b][reader]
            bold = lambda x, y: (f"\\textbf{{{f3(x)}}}" if x > y else f3(x))  # noqa: E731
            star = "$^{*}$" if r["p"] < 0.05 else ""
            rows.append(f"{'\\multirow{2}{*}{' + lab + '}' if i == 0 else ''} & {h} & {bold(r['rag'], r['vgs'])} & {bold(r['vgs'], r['rag'])} & ${100 * r['diff'][0]:+.1f}${star} \\\\")
        rows.append("\\midrule")
    t = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{2.5pt}", "\\renewcommand{\\arraystretch}{1.12}",
         "\\begin{tabularx}{\\linewidth}{llYYc}", "\\toprule", "\\rowcolor{hHead} Reader & History & RAG@40 & \\vgs{} & $\\Delta$ (points) \\\\", "\\midrule",
         *rows[:-1], "\\bottomrule", "\\end{tabularx}",
         "\\caption{\\textbf{MemLens with longer histories (466 test questions).} Accuracy and paired gains over RAG@40 in points; $^{*}$ marks $p<0.05$ in a paired permutation test. Both history lengths use the same questions.}",
         "\\label{tab:memlens_full}", "\\end{table}"]
    (POL / "tab_memlens_full.tex").write_text("\n".join(t) + "\n")
    print("Table 2 written")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("memlens", "all"):
    memlens_table()
