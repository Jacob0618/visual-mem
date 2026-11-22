"""Phase 17: paper generators with the fourth benchmark (CamRoll) and the restructured figures. No API calls.
python scripts/p17_final.py [check|stats|main|figs|appendix|all]
Run order for a full regeneration: scripts/p16_paper.py ablation (three-benchmark ablation tables), then this script."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p16_compose as P  # noqa: E402
import p16_paper as PP  # noqa: E402
import p16_final as F  # noqa: E402

OUT, O17 = ROOT / "results/p16", ROOT / "results/p17"
POL, FIGP = F.POL, F.FIGP
LITE, STRONG = P.LITE, P.STRONG
C, C600, CS = "__cnoassist+shortid+telegraph", "__snip600__cnoassist+shortid+telegraph", "__cnoassist+telegraph"
CR = O17 / "camroll"
P.SRC["camroll"] = {"rag_l": CR / f"rag-40__read-medium__{LITE}.jsonl", "rag_s": CR / f"rag-40__read-low__{STRONG}.jsonl",
                    "std_l": CR / f"toc38-40__read-medium__{LITE}{C}__xcache.jsonl",
                    "wid_l": CR / f"toc38-40__readn-medium__{LITE}{C600}__xcache.jsonl",
                    "std_s": CR / f"toc38-40__read+toc-low__{STRONG}{CS}__xcache.jsonl",
                    "uni_s": CR / f"tocu-40-40__read+toc-low__{STRONG}{CS}__xcache.jsonl"}
P.VHAT["camroll"] = "vhat_compact_camroll.json"
P.HVIS["camroll"] = "hvis_camroll.json"
HAVE_CR = all(Path(v).exists() for v in P.SRC["camroll"].values()) and (OUT / P.HVIS["camroll"]).exists() and (OUT / P.VHAT["camroll"]).exists()
BENCH = ["memeye", "memlens", "visualmem"] + (["camroll"] if HAVE_CR else [])
NAME = {"memeye": "MemEye", "memlens": "MemLens", "visualmem": "VisualMem", "camroll": "CamRoll"}
HCOL = {"memeye": "hME", "memlens": "hML", "visualmem": "hVM", "camroll": "hCR"}
PP.BENCH = BENCH
PP.SHORT["camroll"] = "CR"
F.BENCH = BENCH
f3 = PP.f3


def n_questions(bench):
    return len(P.L(P.SRC[bench]["rag_l"]))


def cache_usd():
    """Explicit-cache creation and storage of the CamRoll standard-path runs per 1,000 questions, measured from the usage
    log inside the wall-clock window of each run (its log file); falls back to the values measured on 2026-11-21."""
    if not HAVE_CR:
        return
    res = {}
    for reader, log, fallback in (("lite", "camroll_std_lite.log", 1.02), ("strong", "camroll_std_strong.log", 2.81)):
        f = O17 / "logs" / log
        tot = None
        if f.exists():
            st = f.stat()
            tot = 0.0
            for line in open(ROOT / "cache/usage.jsonl"):
                if '"xcache:create"' in line:
                    r = json.loads(line)
                    if st.st_birthtime <= r["t"] <= st.st_mtime:
                        tot += r["prompt"] * 0.75e-6 + r.get("storage_usd", 0.0)
        n = len(P.L(P.SRC["camroll"]["std_l"]))
        res[reader] = 1000 * tot / n if tot else fallback
        P.CACHE_USD[("camroll", reader)] = res[reader]
    return res


def fc(bench, reader):
    if bench != "camroll":
        return F._fc_orig(bench, reader)
    p = CR / f"fc__read-medium__{LITE}.jsonl"
    return F.L(p) if (reader == "lite" and p.exists()) else None


def rerank(bench, reader):
    if bench != "camroll":
        return F._rerank_orig(bench, reader)
    p = CR / f"ret_rerank40_{reader}.jsonl"
    f = ROOT / "results/mm2/select_eval_camroll_rankr38-100_cnoassist+shortid+telegraph.json"
    if not p.exists() or not f.exists():
        return None
    usd = {r["qid"]: r["usd"] for r in json.load(open(f))["per_q"]}
    return {q: dict(r, usd=r.get("usd", 0.0) + usd.get(q, 0.0)) for q, r in F.L(p).items()}


F._fc_orig, F._rerank_orig = F.fc, F.rerank
F.fc, F.rerank = fc, rerank


def cells_all():
    for b in BENCH:
        F.NQ[b] = n_questions(b)
    cache_usd()
    return {(b, r): F.cells(b, r) for b in BENCH for r in ("lite", "strong")}


def check(T=None):
    T = T or cells_all()
    for (b, r), Cc in T.items():
        miss = [k for k, v in Cc.items() if v is None]
        print(f"{b:9s} {r:6s} " + " ".join(f"{k}={v['acc']:.3f}" for k, v in Cc.items() if v) + (f" | MISSING: {miss}" if miss else ""))
    return T


MAIN_ROWS = [[("RAG@40", "rag"), ("Full context", "fc")],
             [("SimpleMem", "simplemem"), ("Omni-SimpleMem", "omni"), ("TeleMem", "telemem")],
             [("V-Mem", "vmem"), ("MMA", "mma")], [("\\textbf{\\vgs{} (ours)}", "vgs")]]


def main_table(T):
    """Table 1: accuracy (Light / Strong) on the four benchmarks. The VGS row carries its relative change over RAG@40 as a
    subscript. Costs and the re-ranked retrieval baseline are in the appendix (tab:cost, tab:rerank)."""
    keys = [k for g in MAIN_ROWS for _, k in g]
    best = {k: max(T[k][m]["acc"] for m in keys if T[k][m]) for k in T}
    ncol = "".join("|cc" for _ in BENCH)
    Ls = ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3.4pt}", "\\renewcommand{\\arraystretch}{1.06}",
          f"\\begin{{tabular}}{{l{ncol}}}", "\\toprule",
          "\\multirow{2}{*}{\\textbf{Method}} & " + " & ".join(
              f"\\multicolumn{{2}}{{c{'|' if b != BENCH[-1] else ''}}}{{\\cellcolor{{{HCOL[b]}}}\\textbf{{{NAME[b]}{'-32k' if b == 'memlens' else ''}}}}}" for b in BENCH) + " \\\\",
          " & " + " & ".join(f"\\cellcolor{{{HCOL[b]}}}Light & \\cellcolor{{{HCOL[b]}}}Strong" for b in BENCH) + " \\\\", "\\midrule"]
    rel = {}
    for gi, g in enumerate(MAIN_ROWS):
        if gi:
            Ls.append("\\midrule")
        for name, key in g:
            cs = []
            for b in BENCH:
                for r in ("lite", "strong"):
                    c = T[(b, r)][key]
                    if c is None:
                        cs.append("--"); continue
                    t = f"\\textbf{{{f3(c['acc'])}}}" if abs(c["acc"] - best[(b, r)]) < 5e-4 else f3(c["acc"])
                    if key == "vgs":
                        rel[(b, r)] = 100 * (c["acc"] - T[(b, r)]["rag"]["acc"]) / T[(b, r)]["rag"]["acc"]
                        t += f"\\,{{\\fontsize{{6.2}}{{7}}\\selectfont$_{{{rel[(b, r)]:+.1f}\\%}}$}}"
                    cs.append(t)
            Ls.append(("\\oursrow " if key == "vgs" else "") + name + " & " + " & ".join(cs) + " \\\\")
    nq = ", ".join(str(F.NQ[b]) for b in BENCH[:-1]) + f", and {F.NQ[BENCH[-1]]}"
    Ls += ["\\bottomrule", "\\end{tabular}",
           "\\caption{\\textbf{Main results.} Accuracy with the lightweight (Light) and the strong reader. Every method is evaluated on the "
           f"same questions ({nq}) with the same readers, prompts, and judges. Subscripts give the relative change of \\vgs{{}} over RAG@40; "
           "the best result per column is in bold. Costs are in Table~\\ref{tab:cost}, and retrieval re-ranked by the selector model is compared in "
           "Table~\\ref{tab:rerank}." + (" Full context is not run on CamRoll, where a history holds about 800 photos." if "camroll" in BENCH else "") + "}",
           "\\label{tab:main}", "\\end{table*}"]
    (POL / "tab_main.tex").write_text("\n".join(Ls) + "\n")
    json.dump({f"{b}|{r}": v for (b, r), v in rel.items()}, open(O17 / "relative_gain.json", "w"), indent=1)
    # appendix: re-ranked retrieval, accuracy and cost side by side
    H3 = "".join("|ccY" for _ in BENCH)
    rr = ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{2.3pt}", f"\\begin{{tabularx}}{{\\linewidth}}{{l{H3}}}", "\\toprule",
          "\\multirow{2}{*}{\\textbf{Method}} & " + " & ".join(
              f"\\multicolumn{{3}}{{c{'|' if b != BENCH[-1] else ''}}}{{\\cellcolor{{{HCOL[b]}}}\\textbf{{{NAME[b]}}}}}" for b in BENCH) + " \\\\",
          " & " + " & ".join(f"\\cellcolor{{{HCOL[b]}}}Light & \\cellcolor{{{HCOL[b]}}}Strong & \\cellcolor{{{HCOL[b]}}}\\$/1k" for b in BENCH) + " \\\\", "\\midrule"]
    for name, key in (("RAG@40", "rag"), ("RAG@40 + LLM rerank", "rerank"), ("\\textbf{\\vgs{} (ours)}", "vgs")):
        cs = []
        for b in BENCH:
            bb = {r: max(T[(b, r)][m]["acc"] for m in ("rag", "rerank", "vgs")) for r in ("lite", "strong")}
            for r in ("lite", "strong"):
                c = T[(b, r)][key]
                cs.append(f"\\textbf{{{f3(c['acc'])}}}" if abs(c["acc"] - bb[r]) < 5e-4 else f3(c["acc"]))
            cs.append(F.money(T[(b, "lite")][key]["usd"], T[(b, "strong")][key]["usd"]))
        rr.append(("\\oursrow " if key == "vgs" else "") + name + " & " + " & ".join(cs) + " \\\\")
    rr += ["\\bottomrule", "\\end{tabularx}",
           "\\caption{\\textbf{Re-ranked retrieval.} Accuracy with the lightweight and the strong reader and cost in USD per 1,000 questions "
           "(lightweight\\,/\\,strong). The selector model re-ranks the 100 most similar rounds and the reader receives the top 40; the cost includes the ranking call.}",
           "\\label{tab:rerank}", "\\end{table*}"]
    (POL / "tab_rerank.tex").write_text("\n".join(rr) + "\n")
    # appendix: cost per 1,000 questions
    t = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\columnwidth}{!}{%",
         "\\begin{tabular}{l" + "c" * len(BENCH) + "}", "\\toprule",
         "\\rowcolor{hHead} \\textbf{Method} & " + " & ".join(NAME[b] for b in BENCH) + " \\\\", "\\midrule"]
    for gi, g in enumerate(F.ROWS):
        if gi:
            t.append("\\midrule")
        for name, key in g:
            t.append(("\\oursrow " if key == "vgs" else "") + name + " & " + " & ".join(
                F.money(*(T[(b, r)][key]["usd"] if T[(b, r)][key] else None for r in ("lite", "strong"))) for b in BENCH) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}}",
          "\\caption{\\textbf{Cost.} USD per 1,000 questions with the lightweight\\,/\\,strong reader, including memory construction and caches.}",
          "\\label{tab:cost}", "\\end{table}"]
    (POL / "tab_cost.tex").write_text("\n".join(t) + "\n")
    json.dump({f"{b}|{r}|{k}": {kk: vv for kk, vv in v.items() if kk != "D"} for (b, r), Cc in T.items() for k, v in Cc.items() if v},
              open(OUT / "main_cells.json", "w"), indent=1)
    print("Table 1 (accuracy) and the cost table written for", BENCH)


# ------------------------------------------------------------------------------------------------ figures
def rank_shares(bench):
    R = PP.ranks(bench)
    c = [0, 0, 0]
    for q in R.values():
        for rk in q["gold"].values():
            c[0 if rk <= 10 else (1 if rk <= 40 else 2)] += 1
    return [x / sum(c) for x in c]


def figures(T, S):
    import shutil
    import matplotlib.pyplot as plt
    import make_figures_p13 as M
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    B = [b for b in BENCH if b in S and "story" in S[b]]
    lab = [PP.SHORT[b] for b in B]
    x = np.arange(len(B))
    # ---------------- Figure 3: diagnosis
    fig, ax = M.row(4)
    vd = [("ME", 0.810, 0.216), ("VM", 0.609, 0.297), ("MG", 0.909, 0.892)]
    for i, (_, w, wo) in enumerate(vd):
        ax[0].bar(i - 0.19, w, 0.36, color=M.BLUE, **M.BAR); ax[0].bar(i + 0.19, wo, 0.36, color="#d9d9d9", **M.BAR)
    ax[0].set_xticks(range(3), [v[0] for v in vd], fontsize=5.8); ax[0].set_ylabel("Accuracy"); ax[0].set_ylim(0, 1.22)
    ax[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax[0].set_title("(a) Removing visual input")
    ax[0].legend(handles=[Patch(color=M.BLUE, label="with images"), Patch(color="#d9d9d9", label="without")], frameon=False,
                 loc="upper center", fontsize=5.8, handlelength=1.0, ncol=2, columnspacing=0.6)
    sh = [rank_shares(b) for b in B]
    bot = np.zeros(len(B))
    for k, (c, l_) in enumerate([("#bcd2e8", "top 10"), (M.BLUE, "11–40"), (M.RED, "beyond 40")]):
        v = np.array([s[k] for s in sh]); ax[1].bar(x, v, 0.6, bottom=bot, color=c, label=l_, **M.BAR); bot += v
    ax[1].set_xticks(x, lab, fontsize=5.8); ax[1].set_ylim(0, 1.3); ax[1].set_yticks([0, 0.5, 1.0]); ax[1].set_ylabel("Share of gold rounds")
    ax[1].set_title("(b) Similarity rank of evidence"); ax[1].legend(frameon=False, fontsize=5.4, ncol=3, loc="upper center", handlelength=0.9, columnspacing=0.5, handletextpad=0.3)
    ax[2].bar(x - 0.19, [S[b]["story"]["rag_complete"] for b in B], 0.36, color="#bcd2e8", label="complete set", **M.BAR)
    ax[2].bar(x + 0.19, [S[b]["story"]["rag_incomplete"] for b in B], 0.36, color=M.BLUE, label="incomplete set", **M.BAR)
    ax[2].set_xticks(x, lab, fontsize=5.8); ax[2].set_ylim(0, 0.95); ax[2].set_ylabel("RAG@40 accuracy"); ax[2].set_title("(c) When evidence is missed")
    ax[2].legend(frameon=False, fontsize=5.4, ncol=2, loc="upper center", handlelength=0.9, columnspacing=0.6)
    rows = json.load(open(OUT / "set_analysis_rows_v2.json"))
    qss = {r["qid"]: r for r in json.load(open(OUT / "select_eval_compact_std.json"))["per_q"]}
    Bk = [(1, 1, "1"), (2, 3, "2–3"), (4, 5, "4–5"), (6, 99, "≥6")]
    xb = np.arange(len(Bk))
    mean = lambda f: [float(np.mean([f(r) for r in rows if lo <= r["g"] <= hi])) for lo, hi, _ in Bk]  # noqa: E731
    for l_, f, c, m, ls in [("cosine top-10", lambda r: r["cos10"], "#bcd2e8", "o", "--"), ("cosine top-40", lambda r: r["cos40"], M.BLUE, "s", "--"),
                            ("set selection", lambda r: qss[r["qid"]]["full"], M.RED, "D", "-")]:
        ax[3].plot(xb, mean(f), marker=m, color=c, linestyle=ls, label=l_, **M.EDGE)
    ax[3].set_xticks(xb, [b_[2] for b_ in Bk]); ax[3].set_xlabel("Gold rounds per question"); ax[3].set_ylabel("Full-set recall"); ax[3].set_ylim(0, 1.04)
    ax[3].set_title("(d) Complete evidence sets (MemEye)"); ax[3].legend(frameon=False, loc="lower left", fontsize=5.2, handlelength=1.4, labelspacing=0.15, bbox_to_anchor=(-0.03, -0.03))
    M.finish(fig, "problem"); shutil.copy(M.FIG / "problem.pdf", FIGP / "problem.pdf"); plt.close(fig)
    # ---------------- Figure 4: what VGS changes
    fig, ax = M.row(4, h=1.36, wr=[0.95, 0.95, 1.6, 1.15])
    a = ax[0]
    for k, (c, l_) in enumerate([("#bcd2e8", "top 10"), (M.BLUE, "11–40"), (M.RED, "beyond 40")]):
        a.bar(x + (k - 1) * 0.27, [S[b]["story"]["vgs_opens"][k] for b in B], 0.26, color=c, label=l_, **M.BAR)
    a.set_xticks(x, lab, fontsize=5.8); a.set_ylim(0, 1.3); a.set_yticks([0, 0.5, 1.0]); a.set_ylabel("Opened by VGS"); a.set_title("(a) Evidence by similarity rank")
    a.set_ylim(0, 1.42)
    a.legend(frameon=False, fontsize=5.0, ncol=3, loc="upper center", handlelength=0.7, columnspacing=0.35, handletextpad=0.25, borderpad=0.1)
    a = ax[1]
    a.bar(x - 0.19, [S[b]["story"]["vgs_on_rag_incomplete"] - S[b]["story"]["rag_incomplete"] for b in B], 0.36, color=M.RED, label="RAG@40 set incomplete", **M.BAR)
    a.bar(x + 0.19, [S[b]["story"]["vgs_on_rag_complete"] - S[b]["story"]["rag_complete"] for b in B], 0.36, color="#e8b9b9", label="complete", **M.BAR)
    a.axhline(0, color="black", linewidth=0.5); a.set_xticks(x, lab, fontsize=5.8); a.set_ylabel("VGS − RAG@40"); a.set_title("(b) Where the gain arises")
    ymax = max(S[b]["story"]["vgs_on_rag_incomplete"] - S[b]["story"]["rag_incomplete"] for b in B)
    a.set_ylim(min(-0.02, a.get_ylim()[0]), ymax * 1.45); a.legend(frameon=False, fontsize=5.0, loc="upper center", handlelength=0.9, labelspacing=0.15)
    a = ax[2]
    groups = [(b, r) for r in ("lite", "strong") for b in BENCH if b in S]
    xg = np.arange(len(groups))
    for j, (k, c) in enumerate([("std", "#f3d37a"), ("wid", "#9ec3e6"), ("vgs", "#7b1f1f")]):
        a.bar(xg + (j - 1) * 0.26, [S[b][r][k] - S[b][r]["rag"] for b, r in groups], 0.26, color=c, **M.BAR)
    a.axhline(0, color="black", linewidth=0.6)
    a.set_xticks(xg, [PP.SHORT[b] for b, r in groups], fontsize=5.4)
    nb = len([g_ for g_ in groups if g_[1] == "lite"])
    a.axvline(nb - 0.5, color=M.GRAY, linewidth=0.6, linestyle=":")
    lo_, hi_ = a.get_ylim(); a.set_ylim(lo_, hi_ + 0.22 * (hi_ - lo_))
    a.text((nb - 1) / 2, a.get_ylim()[1] * 0.985, "light reader", ha="center", va="top", fontsize=5.6)
    a.text(nb + (len(groups) - nb - 1) / 2, a.get_ylim()[1] * 0.985, "strong reader", ha="center", va="top", fontsize=5.6)
    a.set_ylabel("Minus RAG@40"); a.set_title("(c) Reading paths and the gate")
    a = ax[3]
    col = {"memeye": M.RED, "memlens": M.BLUE, "visualmem": M.TEAL, "camroll": M.ORANGE}
    for b in BENCH:
        pts = S[b]["calibration"]["lite"]
        a.scatter([p_[0] for p_ in pts], [p_[2] for p_ in pts], s=[max(5.0, p_[1] / 5.0) for p_ in pts], color=col[b], edgecolor="black", linewidth=0.35, zorder=3, label=PP.SHORT[b])
    a.axvline(P.TAU_H, color=M.GRAY, linewidth=0.7, linestyle=":"); a.axhline(0, color="black", linewidth=0.5)
    a.set_xlim(0.15, 1.05); a.set_xlabel("History visibility"); a.set_ylabel("Widened − standard"); a.set_title("(d) Visibility and widening")
    lo_, hi_ = a.get_ylim(); a.set_ylim(lo_, hi_ + 0.28 * (hi_ - lo_))
    lg = a.legend(frameon=False, fontsize=5.0, loc="upper right", ncol=len(BENCH), handletextpad=0.0, borderpad=0.05, columnspacing=0.3, bbox_to_anchor=(1.03, 1.02))
    for h_ in lg.legend_handles:
        h_.set_sizes([9.0])
    hs = [Patch(color="#f3d37a", label="standard path for all"), Patch(color="#9ec3e6", label="widened path for all"), Patch(color="#7b1f1f", label="VGS gate")]
    M.finish(fig, "story_gate", handles=hs, ncol=3); shutil.copy(M.FIG / "story_gate.pdf", FIGP / "story_gate.pdf"); plt.close(fig)
    # ---------------- Figure 5: accuracy against cost on every benchmark
    METH = [("rag", "RAG@40", "o"), ("rerank", "+ LLM rerank", "d"), ("fc", "Full context", "s"), ("simplemem", "SimpleMem", "^"), ("omni", "Omni-SimpleMem", "D"),
            ("telemem", "TeleMem", "h"), ("vmem", "V-Mem", "v"), ("mma", "MMA", "P"), ("vgs", "VGS (ours)", "*")]
    colr = {"lite": M.BLUE, "strong": M.RED}
    fig, ax = M.row(len(BENCH), h=1.32)
    for a, b, t in zip(ax, BENCH, "abcd"):
        for r in ("lite", "strong"):
            for k, _, m in METH:
                v = T[(b, r)][k]
                if v:
                    a.scatter(v["usd"], v["acc"], color=colr[r], marker=m, s=46 if k == "vgs" else 17, edgecolor="black", linewidth=0.45, zorder=3)
        a.set_xscale("log"); a.set_xlim(0.2, 200); a.set_xticks([1, 10, 100], ["1", "10", "100"]); a.minorticks_off()
        a.set_xlabel("USD per 1k questions"); a.set_title(f"({t}) {NAME[b]}")
    ax[0].set_ylabel("Accuracy")
    hs = [Line2D([], [], color="#bbbbbb", marker=m, linestyle="", markersize=5.5 if k == "vgs" else 4, label=l_, **M.EDGE) for k, l_, m in METH]
    hs += [Line2D([], [], color=colr[r], marker="s", linestyle="", markersize=4, label=l_, **M.EDGE) for r, l_ in (("lite", "light reader"), ("strong", "strong reader"))]
    M.finish(fig, "cost", hs, ncol=11, legend_h=0.12); shutil.copy(M.FIG / "cost.pdf", FIGP / "cost.pdf"); plt.close(fig)
    print("figures written: problem, story_gate, cost")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    print("benchmarks:", BENCH)
    T = cells_all()
    if what == "check":
        check(T)
    S = PP.stats() if what in ("stats", "figs", "all") else None
    if what in ("main", "all"):
        main_table(T)
    if what in ("figs", "all"):
        figures(T, S)


# ------------------------------------------------------------------------------------------------ appendix
def selectors_table():
    S = json.load(open(O17 / "selectors.json"))
    E = {"gemini-3.8-flash": float(np.mean([r["score"] for r in P.compose("memeye")[0]["std_l"].values()]))}
    for name, d in (("GPT-5-mini", "open_gpt5m"), ("DeepSeek-V3.1", "open_ds")):
        fs = sorted((O17 / d / "memeye").glob("*.jsonl"))
        if fs:
            E[name] = float(np.mean([json.loads(l)["score"] for l in open(fs[0])]))
    fs = sorted((OUT / "open/memeye").glob("*.jsonl"))
    if fs:
        E["Qwen3-235B"] = float(np.mean([json.loads(l)["score"] for l in open(fs[0])]))
    for name, alias in (("gemini-2.5-flash", "2.5f"), ("gemini-2.5-flash-lite", "2.5fl"), ("GPT-4.1-mini", "gpt41m"), ("Qwen3-235B (thinking)", "qwent")):
        fs = sorted((ROOT / "results/p22/sel").glob(f"toc@{alias}-40__read-medium__*.jsonl"))
        if fs:
            R_ = [json.loads(l) for l in open(fs[0])]
            if len(R_) == 371 and not any(r.get("error") for r in R_):
                E[name] = float(np.mean([r["score"] for r in R_]))
    rag = float(np.mean([r["score"] for r in P.compose("memeye")[0]["rag_l"].values()]))
    order = sorted(S, key=lambda k: -S[k]["fsr"])
    t = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3.2pt}", "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lccc|c}", "\\toprule",
         "\\rowcolor{hHead} \\textbf{Selector} & FSR & Rounds & Cosine & Accuracy \\\\", "\\midrule"]
    for k in order:
        r = S[k]
        nm = k + ("$^{\\ddagger}$" if r["n"] < 371 else "")
        t.append(("\\oursrow " if k == "gemini-3.8-flash" else "") + f"{nm} & {f3(r['fsr'])} & {r['rounds']:.1f} & {f3(r['cos_same_k']) if r['cos_same_k'] is not None else '--'} & {f3(E[k]) if k in E else '--'} \\\\")
    t += ["\\midrule", f"RAG@40 & .914 & 40 & -- & {f3(rag)} \\\\", "\\bottomrule", "\\end{tabular}}",
          "\\caption{\\textbf{Selector models} under the same instruction and compact index on MemEye. FSR and opened rounds of the selection; "
          "\\emph{Cosine} is the full-set recall of cosine retrieval with as many rounds as each selection; \\emph{Accuracy} is the lightweight reader's on the standard path. "
          "$^{\\ddagger}$A fixed random subset of 150 questions.}", "\\label{tab:selectors}", "\\end{table}"]
    (POL / "tab_selectors.tex").write_text("\n".join(t) + "\n")
    return S, E


def retrieval_table4(T):
    """Retrieval baselines on every benchmark with both readers (appendix)."""
    P22 = ROOT / "results/p22/ret"

    def a(p):
        p = Path(p)
        if not p.exists():
            return None
        R = [json.loads(l) for l in open(p)]
        return float(np.mean([r["score"] for r in R])) if R and not any(r.get("error") for r in R) else None

    def cellv(b, r, key):
        lite = r == "lite"
        if key in ("rag", "rerank", "vgs"):
            return T[(b, r)][key]["acc"] if T[(b, r)][key] else None
        if key == "hyb":
            return a((O17 / f"camroll/ret_hybrid40_{r}.jsonl") if b == "camroll" else (OUT / f"final/{b}/ret_hybrid40_{r}.jsonl"))
        k = int(key)
        name = f"rag-{k}__read-{'medium' if lite else 'low'}__{LITE if lite else STRONG}.jsonl"
        cands = [P22 / b / name]
        if lite and b != "camroll":
            cands.insert(0, OUT / f"swap/{b}/{name}")
        if not lite and k == 30 and b != "camroll":
            cands.insert(0, OUT / f"fig1/{b}_rag30_strong.jsonl")
        if not lite and k == 10 and b == "memeye":
            cands.insert(0, OUT / f"sec2/{name}")
        for c in cands:
            v = a(c)
            if v is not None:
                return v
        return None

    rows = [("Dense top-10", "10"), ("Dense top-20", "20"), ("Dense top-30", "30"), ("Dense top-40 (RAG@40)", "rag"), ("BM25 + dense (RRF), top-40", "hyb"),
            ("Dense top-100 $\\rightarrow$ LLM rerank, top-40", "rerank")]
    R = {f"{b}|{r}|{k}": cellv(b, r, k) for b in BENCH for r in ("lite", "strong") for _, k in rows + [("", "vgs")]}
    json.dump(R, open(O17 / "retrieval_baselines4.json", "w"), indent=1)
    t = ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{5pt}", "\\begin{tabular}{l" + "|cc" * len(BENCH) + "}", "\\toprule",
         "\\multirow{2}{*}{\\textbf{Retrieval}} & " + " & ".join(f"\\multicolumn{{2}}{{c{'|' if b != BENCH[-1] else ''}}}{{\\cellcolor{{{HCOL[b]}}}\\textbf{{{NAME[b]}}}}}" for b in BENCH) + " \\\\",
         " & " + " & ".join(f"\\cellcolor{{{HCOL[b]}}}Light & \\cellcolor{{{HCOL[b]}}}Strong" for b in BENCH) + " \\\\", "\\midrule"]
    for lab, k in rows:
        t.append(lab + " & " + " & ".join(f3(R[f"{b}|{r}|{k}"]) if R[f"{b}|{r}|{k}"] is not None else "--" for b in BENCH for r in ("lite", "strong")) + " \\\\")
    t += ["\\midrule", "\\oursrow \\vgs{} & " + " & ".join(f3(R[f"{b}|{r}|vgs"]) for b in BENCH for r in ("lite", "strong")) + " \\\\", "\\bottomrule", "\\end{tabular}",
          "\\caption{\\textbf{Retrieval baselines.} Accuracy with the lightweight and the strong reader. The reranker is the selector model with the selector instruction, "
          "applied to the 100 most similar rounds.}", "\\label{tab:retrieval}", "\\end{table*}"]
    (POL / "tab_retrieval.tex").write_text("\n".join(t) + "\n")
    miss = [k for k, v in R.items() if v is None]
    print("retrieval table (four benchmarks): missing cells:", miss)
    return R


def long_table():
    Lg = json.load(open(O17 / "long.json"))
    t = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabularx}{\\linewidth}{lYYYY}", "\\toprule",
         "\\rowcolor{hHead} & \\multicolumn{2}{c}{\\textbf{Light}} & \\multicolumn{2}{c}{\\textbf{Strong}} \\\\",
         "\\rowcolor{hHead} \\textbf{Rounds} & RAG@40 & \\vgs{} & RAG@40 & \\vgs{} \\\\", "\\midrule"]
    for g in ("0", "250", "500", "1000"):
        r = Lg.get(g)
        if not r or any(r.get(k) is None for k in ("rag_l", "vgs_l", "rag_s", "vgs_s")):
            continue
        b = lambda x, y: (f"\\textbf{{{f3(x)}}}" if x > y else f3(x))  # noqa: E731
        t.append(f"{int(r['rounds'] + 0.5):,} & {b(r['rag_l'], r['vgs_l'])} & {b(r['vgs_l'], r['rag_l'])} & {b(r['rag_s'], r['vgs_s'])} & {b(r['vgs_s'], r['rag_s'])} \\\\")
    t += ["\\bottomrule", "\\end{tabularx}",
          "\\caption{\\textbf{Longer histories.} MemEye histories enlarged with image chats of other users (mean rounds per history); the 371 questions and their evidence are unchanged. "
          "\\vgs{} reads the whole index on its standard path.}", "\\label{tab:long}", "\\end{table}"]
    (POL / "tab_long.tex").write_text("\n".join(t) + "\n")
    return Lg


def baselines_figure(T):
    import shutil
    import matplotlib.pyplot as plt
    import make_figures_p13 as M
    METH = [("rerank", "+ LLM rerank", M.BLUE, "d"), ("fc", "Full context", M.DARK, "s"), ("simplemem", "SimpleMem", M.ORANGE, "^"), ("omni", "Omni-SimpleMem", M.OLIVE, "D"),
            ("telemem", "TeleMem", M.GRAY, "h"), ("vmem", "V-Mem", M.TEAL, "v"), ("mma", "MMA", M.GREEN, "P"), ("vgs", "VGS (ours)", M.RED, "*")]
    panels = [(b, r, f"{NAME[b]}, {rl}") for b in BENCH for r, rl in (("lite", "light"), ("strong", "strong"))]
    fig, ax = M.row(len(panels), h=1.75)
    for a, (b, r, title) in zip(ax, panels):
        y = np.arange(len(METH))[::-1]
        for yi, (k, _, c, mk) in zip(y, METH):
            v = T[(b, r)][k]
            if v:
                a.plot([v["lo"], v["hi"]], [yi, yi], color=c, linewidth=1.2)
                a.plot([v["d"]], [yi], marker=mk, color=c, markersize=4.4 if k == "vgs" else 3.4, **M.EDGE)
        a.axvline(0, color="black", linewidth=0.6)
        a.set_yticks(y, [o[1] for o in METH] if a is ax[0] else [], fontsize=5.6); a.set_title(title, fontsize=6.0)
        a.tick_params(axis="x", labelsize=5.0); a.set_xlim(-0.6, 0.3); a.set_xticks([-0.4, 0, 0.2])
    fig.supxlabel("Accuracy minus RAG@40 with paired 95% bootstrap interval", fontsize=6.4, y=0.02)
    M.finish(fig, "app_baselines"); shutil.copy(M.FIG / "app_baselines.pdf", FIGP / "app_baselines.pdf"); plt.close(fig)


def gate_table(S):
    """Routing policies on every benchmark (history-level gate of the paper and per-question routers)."""
    nb = len(BENCH)
    g = ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabularx}{\\linewidth}{X|" + "c" * nb + "|" + "c" * nb + "}", "\\toprule",
         f"\\rowcolor{{hHead}} & \\multicolumn{{{nb}}}{{c|}}{{\\textbf{{Lightweight reader}}}} & \\multicolumn{{{nb}}}{{c}}{{\\textbf{{Strong reader}}}} \\\\",
         "\\rowcolor{hHead} \\textbf{Policy} & " + " & ".join(NAME[b] for b in BENCH) + " & " + " & ".join(NAME[b] for b in BENCH) + " \\\\", "\\midrule"]

    def row(lab, k, fmt="{:.3f}"):
        return lab + " & " + " & ".join(fmt.format(S[b][r][k]) if S[b][r].get(k) is not None else "--" for r in ("lite", "strong") for b in BENCH) + " \\\\"
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


def appendix(T):
    S = json.load(open(OUT / "paper_stats.json"))
    gate_table(S)                        # routing policies on every benchmark of BENCH
    keep = F.BENCH
    F.BENCH = ["memeye", "memlens", "visualmem"]   # these analyses were run on the three development-era benchmarks
    try:
        F.swap_table(S); F.std_table(T)
    finally:
        F.BENCH = keep
    (POL / "tab_systems.tex").write_text(F.systems_table() + "\n")
    F.memlens_table()
    retrieval_table4(T)
    if (O17 / "selectors.json").exists():
        selectors_table()
    if (O17 / "long.json").exists():
        long_table()
    baselines_figure(T)
    print("appendix tables and the baseline-interval figure written")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("appendix", "all"):
    appendix(T)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("ablation", "all"):
    PP.ablation()      # Table 3 and the full ablation table on every benchmark of BENCH (needs the CamRoll control runs)
