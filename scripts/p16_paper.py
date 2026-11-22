"""Phase 16: statistics, figures and tables of the ACL paper for the compact-index VGS (v2 image descriptions).
No API calls.  python scripts/p16_paper.py [stats|figs|tables|all]"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p16_compose as P  # noqa: E402
from analyze import boot_ci  # noqa: E402

OUT = ROOT / "results/p16"
POL = ROOT / "paper_acl/sections_polished"
FIGP = ROOT / "paper_acl/figures_polished"
BENCH = ["memeye", "memlens", "visualmem"]
SHORT = {"memeye": "ME", "memlens": "ML", "visualmem": "VM", "memlens64": "ML64"}
LEN_T = 165       # message-length gate threshold (characters), set on the development data with tau


def ranks(bench):
    return json.load(open(OUT / f"ranks_{bench}.json"))


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    pos, neg = np.array(pos), np.array(neg)
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def perm_p(d, n=20000, seed=0):
    """Two-sided paired sign-flip permutation test for a mean difference of zero."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(n, len(d)))
    return float((np.abs((flips * d).mean(axis=1)) >= obs - 1e-12).mean())


def calibration(bench, out):
    """(history visibility, n, widened - standard) per history (MemEye, VisualMem) or per quintile of histories (MemLens)."""
    H = P.hvis(bench)
    res = {}
    for reader, sk, wk in (("lite", "std_l", "wid_l"), ("strong", "std_s", "uni_s")):
        std, wid = out[sk], out[wk]
        qs = sorted(q for q in std if q in wid and q in H)
        pts = []
        if bench.startswith("memlens"):
            h = np.array([H[q]["H200"] for q in qs])
            order = np.argsort(h, kind="stable")
            for part in np.array_split(order, 5):
                sel = [qs[i] for i in part]
                pts.append([float(np.mean([H[q]["H200"] for q in sel])), len(sel), float(np.mean([wid[q]["score"] - std[q]["score"] for q in sel]))])
        else:
            by = {}
            for q in qs:
                by.setdefault(H[q]["hist"], []).append(q)
            for hist, sel in by.items():
                pts.append([H[sel[0]]["H200"], len(sel), float(np.mean([wid[q]["score"] - std[q]["score"] for q in sel]))])
        res[reader] = sorted(pts)
    return res


def stats():
    S = {}
    for bench in BENCH + ["memlens64"]:
        if not (OUT / P.VHAT[bench]).exists() or not all(Path(v).exists() for v in P.SRC[bench].values()):
            print(bench, "runs incomplete"); continue
        try:
            out, V = P.compose(bench)
        except SystemExit as ex:
            print(ex); continue
        if not out.get("vgs_l") or not out.get("vgs_s"):
            print(bench, "incomplete"); continue
        R = ranks(bench) if (OUT / f"ranks_{bench}.json").exists() else {}
        b = S[bench] = {}
        for reader, base_k, std_k, wid_k, vgs_k in [("lite", "rag_l", "std_l", "wid_l", "vgs_l"), ("strong", "rag_s", "std_s", "uni_s", "vgs_s")]:
            base, std, wid, vgs = out[base_k], out[std_k], out[wid_k], out[vgs_k]
            qs = sorted(set(base) & set(std) & set(wid) & set(vgs))
            acc = lambda D: float(np.mean([D[q]["score"] for q in qs]))  # noqa: E731
            usd = lambda D: 1000 * float(np.mean([D[q].get("usd", 0.0) for q in qs]))  # noqa: E731
            d = np.array([vgs[q]["score"] - base[q]["score"] for q in qs]); lo, hi = boot_ci(d)
            share = float(np.mean([out["route"].get(q, True) for q in qs]))        # history gate: share on the standard path
            share_q = float(np.mean([V.get(q, 1.0) >= P.TAU for q in qs]))        # per-question estimate v_hat >= tau
            r = {"n": len(qs), "rag": acc(base), "std": acc(std), "wid": acc(wid), "vgs": acc(vgs), "diff": [float(d.mean()), lo, hi],
                 "rag_usd": usd(base), "vgs_usd": usd(vgs) + P.CACHE_USD.get((bench, reader), 0.0), "share_std": share,
                 "oracle": float(np.mean([max(std[q]["score"], wid[q]["score"]) for q in qs]))}
            r["random"] = float(np.mean([share_q * std[q]["score"] + (1 - share_q) * wid[q]["score"] for q in qs]))
            r["qgate"] = float(np.mean([(std if V.get(q, 1.0) >= P.TAU else wid)[q]["score"] for q in qs]))
            r["share_q"] = share_q
            r["p"] = perm_p(d)
            r["std_usd"], r["wid_usd"] = usd(std), usd(wid)
            r["diff_std"] = [float(np.mean([std[q]["score"] - base[q]["score"] for q in qs]))]
            r["diff_wid"] = [float(np.mean([wid[q]["score"] - base[q]["score"] for q in qs]))]
            if R:
                r["length"] = float(np.mean([(wid if R[q]["top10_user_len"] >= LEN_T else std)[q]["score"] for q in qs]))
            pos = [-V.get(q, 1.0) for q in qs if wid[q]["score"] > std[q]["score"]]
            neg = [-V.get(q, 1.0) for q in qs if wid[q]["score"] < std[q]["score"]]
            r["auc"] = auc(pos, neg)
            r["tau"] = {f"{t:.2f}": float(np.mean([(std if V.get(q, 1.0) >= t else wid)[q]["score"] for q in qs])) for t in np.arange(0.4, 1.001, 0.05)}
            fs = [vgs[q]["full"] for q in qs if vgs[q].get("n_gold")]
            r["fsr"] = float(np.mean(fs)) if fs else None
            r["rounds"] = float(np.mean([vgs[q].get("n_loaded", 0) for q in qs]))
            b[reader] = r
        if R:   # story analyses with the lightweight reader
            base, vgs = out["rag_l"], out["vgs_l"]
            qs = sorted(set(base) & set(vgs) & set(R))
            tot = [0, 0, 0]; hit = [0, 0, 0]
            for q in qs:
                got = set(vgs[q].get("retrieved", []))
                for rid, rk in R[q]["gold"].items():
                    k = 0 if rk <= 10 else (1 if rk <= 40 else 2); tot[k] += 1; hit[k] += int(rid in got)
            inc = [q for q in qs if base[q].get("n_gold") and base[q]["full"] == 0]
            com = [q for q in qs if base[q].get("n_gold") and base[q]["full"] == 1]
            m = lambda D, Q: float(np.mean([D[q]["score"] for q in Q])) if Q else float("nan")  # noqa: E731
            b["story"] = {"share_beyond40": tot[2] / sum(tot), "share_11_40": tot[1] / sum(tot), "vgs_opens": [h / max(t, 1) for h, t in zip(hit, tot)],
                          "rag_complete": m(base, com), "rag_incomplete": m(base, inc), "vgs_on_rag_incomplete": m(vgs, inc),
                          "vgs_on_rag_complete": m(vgs, com), "n_incomplete": len(inc), "n_complete": len(com)}
        b["calibration"] = calibration(bench, out)
        hv = sorted({h["hist"]: h["H200"] for h in P.hvis(bench).values()}.values())
        b["hvis"] = {"mean": float(np.mean(hv)), "min": float(hv[0]), "max": float(hv[-1]), "n": len(hv)}
        print(bench, json.dumps({k: (v if k in ("story", "hvis") else None if k == "calibration" else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "tau"}) for k, v in b.items()})[:900])
    json.dump(S, open(OUT / "paper_stats.json", "w"), indent=1)
    return S


def figs(S=None):
    import make_figures_p13 as M
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    S = S or json.load(open(OUT / "paper_stats.json"))
    B = [b for b in BENCH if b in S and "story" in S[b]]
    fig, ax = M.row(4, h=1.36, wr=[0.9, 0.9, 1.55, 1.15])
    x = np.arange(len(B)); lab = [SHORT[b] for b in B]
    a = ax[0]
    a.bar(x - 0.19, [S[b]["story"]["share_beyond40"] for b in B], 0.36, color="#d9d9d9", **M.BAR)
    a.bar(x + 0.19, [S[b]["story"]["vgs_opens"][2] for b in B], 0.36, color=M.RED, **M.BAR)
    a.set_xticks(x, lab, fontsize=5.8); a.set_ylim(0, 1.0); a.set_ylabel("Share")
    a.set_title("(a) Beyond rank 40")
    a = ax[1]
    a.bar(x - 0.27, [S[b]["story"]["rag_complete"] for b in B], 0.26, color="#bcd2e8", **M.BAR)
    a.bar(x, [S[b]["story"]["rag_incomplete"] for b in B], 0.26, color=M.BLUE, **M.BAR)
    a.bar(x + 0.27, [S[b]["story"]["vgs_on_rag_incomplete"] for b in B], 0.26, color=M.RED, **M.BAR)
    a.set_xticks(x, lab, fontsize=5.8); a.set_ylim(0, 0.8); a.set_ylabel("Accuracy")
    a.set_title("(b) RAG@40 misses evidence")
    a = ax[2]
    groups = [(b, r) for r in ("lite", "strong") for b in BENCH if b in S]
    xg = np.arange(len(groups))
    for j, (k, c) in enumerate([("std", "#f3d37a"), ("wid", "#9ec3e6"), ("vgs", "#7b1f1f")]):
        a.bar(xg + (j - 1) * 0.26, [S[b][r][k] - S[b][r]["rag"] for b, r in groups], 0.26, color=c, **M.BAR)
    a.axhline(0, color="black", linewidth=0.6)
    a.set_xticks(xg, [f"{SHORT[b]}\n{'light' if r == 'lite' else 'strong'}" for b, r in groups], fontsize=5.4)
    a.set_ylabel("Minus RAG@40"); a.set_title("(c) Reading paths and the gate")
    a = ax[3]
    col = {"memeye": M.RED, "memlens": M.BLUE, "visualmem": M.TEAL}
    for b in BENCH:
        if b not in S:
            continue
        pts = S[b]["calibration"]["lite"]
        a.scatter([p_[0] for p_ in pts], [p_[2] for p_ in pts], s=[max(5.0, p_[1] / 5.0) for p_ in pts], color=col[b],
                  edgecolor="black", linewidth=0.35, zorder=3, label={"memeye": "MemEye", "memlens": "MemLens", "visualmem": "VisualMem"}[b])
    a.axvline(P.TAU_H, color=M.GRAY, linewidth=0.7, linestyle=":"); a.axhline(0, color="black", linewidth=0.5)
    a.set_xlim(0.2, 0.9); a.set_xlabel("History visibility"); a.set_ylabel("Widened − standard")
    a.set_title("(d) Visibility and widening")
    a.set_ylim(-0.17, 0.2)
    lg = a.legend(frameon=False, fontsize=5.0, loc="upper right", ncol=3, handletextpad=0.0, borderpad=0.05, columnspacing=0.3,
                  bbox_to_anchor=(1.03, 1.02))
    for h_ in lg.legend_handles:
        h_.set_sizes([9.0])
    hs = [Patch(color="#d9d9d9", label="share of gold rounds"), Patch(color=M.RED, label="opened by VGS (a), VGS (b)"),
          Patch(color="#bcd2e8", label="RAG@40, complete set"), Patch(color=M.BLUE, label="RAG@40, incomplete set"),
          Patch(color="#f3d37a", label="standard path for all"), Patch(color="#9ec3e6", label="widened path for all"),
          Patch(color="#7b1f1f", label="VGS gate")]
    M.finish(fig, "story_gate", handles=hs, ncol=7)
    import shutil
    shutil.copy(M.FIG / "story_gate.pdf", FIGP / "story_gate.pdf")
    plt.close(fig)
    print("wrote figures_polished/story_gate.pdf")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    S = stats() if what in ("stats", "all") else None
    if what in ("figs", "all"):
        figs(S)


def f3(x):
    return f"{x:.3f}"[1:] if 0 <= x < 1 else f"{x:.3f}"


def tables(S=None):
    S = S or json.load(open(OUT / "paper_stats.json"))
    sub = set(json.load(open(ROOT / "results/p11_memlens_sub150.json")))
    # ---- Table 1: replace the RAG@40 and VGS rows of the main table (150-question MemLens subset)
    cells = {}
    for bench, qids in [("memeye", None), ("memlens", sub), ("visualmem", None)]:
        out, V = P.compose(bench, qids)
        for reader, bk, vk in [("lite", "rag_l", "vgs_l"), ("strong", "rag_s", "vgs_s")]:
            base, vgs = out[bk], out[vk]; qs = sorted(set(base) & set(vgs))
            cells[(bench, reader)] = (float(np.mean([base[q]["score"] for q in qs])), float(np.mean([vgs[q]["score"] for q in qs])),
                                      1000 * float(np.mean([base[q].get("usd", 0) for q in qs])),
                                      1000 * float(np.mean([vgs[q].get("usd", 0) for q in qs])) + P.CACHE_USD.get((bench, reader), 0.0))
    t = (POL / "tab_main.tex").read_text()
    lines = t.split("\n")
    def money(a, b):
        return f"{a:.1f}\\,/\\,{b:.1f}" if b < 100 else f"{a:.1f}\\,/\\,{b:.0f}"
    fc = {"memeye": (.495, .819), "memlens": (.520, .740), "visualmem": (.454, .967)}
    for i, l in enumerate(lines):
        if l.startswith("RAG@40 &"):
            parts = ["RAG@40"]
            for b in ("memeye", "memlens", "visualmem"):
                (rl, _, cl, _), (rs, _, cs, _) = cells[(b, "lite")], cells[(b, "strong")]
                parts += [f3(rl), f3(rs), money(cl, cs)]
            lines[i] = " & ".join(parts) + " \\\\"
        if "\\textbf{\\vgs{} (ours)}" in l:
            parts = ["\\oursrow \\textbf{\\vgs{} (ours)}"]
            for b in ("memeye", "memlens", "visualmem"):
                (_, vl, _, cl), (_, vs, _, cs) = cells[(b, "lite")], cells[(b, "strong")]
                bl = vl >= max(cells[(b, "lite")][0], fc[b][0]); bs = vs >= max(cells[(b, "strong")][0], fc[b][1]) - 1e-9
                parts += [("\\textbf{%s}" % f3(vl)) if bl else f3(vl), ("\\textbf{%s}" % f3(vs)) if bs else f3(vs), money(cl, cs)]
            lines[i] = " & ".join(parts) + " \\\\"
    t = "\n".join(lines)
    if "\\textbf{.740}" not in t and abs(cells[("memlens", "strong")][1] - 0.740) < 5e-4:
        t = t.replace("& .520 & .740 &", "& .520 & \\textbf{.740} &")
    (POL / "tab_main.tex").write_text(t)
    # ---- Table 2: full MemLens test (466 questions), 32k and 64k histories
    rows = []
    for reader, lab in (("lite", "Light"), ("strong", "Strong")):
        first = True
        for bench, h in (("memlens", "32k"), ("memlens64", "64k")):
            if bench not in S: continue
            r = S[bench][reader]; d = r["diff"]
            bold = (lambda x, y: (f"\\textbf{{{f3(x)}}}" if x > y else f3(x)))
            rows.append(f"{'\\multirow{2}{*}{' + lab + '}' if first else ''} & {h} & {bold(r['rag'], r['vgs'])} & {bold(r['vgs'], r['rag'])} & "
                        f"${100*d[0]:+.1f}$" + ("$^{*}$" if d[1] > 0 or d[2] < 0 else "") + " \\\\")
            first = False
        rows.append("\\midrule")
    rows = rows[:-1]
    tab2 = (POL / "tab_memlens_full.tex").read_text()
    head, rest = tab2.split("\\midrule", 1)
    tail = rest[rest.index("\\bottomrule"):]
    (POL / "tab_memlens_full.tex").write_text(head + "\\midrule\n" + "\n".join(rows) + "\n" + tail)
    # ---- routing policies (appendix)
    g = ["\\begin{table*}[t]", "\\centering", "\\small", "\\begin{tabularx}{\\linewidth}{X|ccc|ccc}", "\\toprule",
         "\\rowcolor{hHead} & \\multicolumn{3}{c|}{\\textbf{Lightweight reader}} & \\multicolumn{3}{c}{\\textbf{Strong reader}} \\\\",
         "\\rowcolor{hHead} \\textbf{Policy} & MemEye & MemLens & VisualMem & MemEye & MemLens & VisualMem \\\\", "\\midrule"]
    for lab, k in [("Standard path for every question", "std"), ("Widened path for every question", "wid"),
                   ("Gate on message length", "length"), ("Random routing, same share", "random")]:
        g.append(lab + " & " + " & ".join(f"{S[b][r][k]:.3f}" for r in ("lite", "strong") for b in BENCH) + " \\\\")
    g.append("\\oursrow \\vgs{} gate ($\\hat v(q)\\ge0.7$) & " + " & ".join(f"{S[b][r]['vgs']:.3f}" for r in ("lite", "strong") for b in BENCH) + " \\\\")
    g += ["\\midrule", "Per-question oracle & " + " & ".join(f"{S[b][r]['oracle']:.3f}" for r in ("lite", "strong") for b in BENCH) + " \\\\",
          "AUC of $\\hat v(q)$ & " + " & ".join(f"{S[b][r]['auc']:.2f}" for r in ("lite", "strong") for b in BENCH) + " \\\\",
          "Share on the standard path & " + " & ".join(f"{S[b][r]['share_std']:.2f}" for r in ("lite", "strong") for b in BENCH) + " \\\\",
          "\\bottomrule", "\\end{tabularx}",
          "\\caption{\\textbf{Routing policies.} Accuracy for fixed paths and routing rules. The oracle chooses the better path per question; AUC measures whether estimated visibility predicts which path helps.}",
          "\\label{tab:gate}", "\\end{table*}"]
    (POL / "tab_gate.tex").write_text("\n".join(g) + "\n")
    print("tables written")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "tables":
    tables()


def ablation():
    F = ROOT / "results/p16/final"
    C = "__cnoassist+shortid+telegraph"
    LITE = "gemini-2.5-flash-lite"
    A = {}
    for bench in BENCH:
        out, V = P.compose(bench)
        FB = ROOT / "results/p16/final2" if bench == "memeye" else F
        pool = lambda k: P.L(FB / bench / f"ptoc38-100-40__{k}-medium__{LITE}{'__snip600' if k == 'readn' else ''}{C}.jsonl")  # noqa: E731
        if bench == "memlens":   # histories have at most 83 rounds: the 100-round pool is the whole history
            pstd, pwid = out["std_l"], out["wid_l"]
        elif bench == "camroll":  # the gate never widens CamRoll, so the pooled widened path is not needed
            pstd = pwid = pool("read")
        else:
            pstd, pwid = pool("read"), pool("readn")
        ctrl = {k: P.L(FB / bench / f"abl_{k}.jsonl") for k in ("cosqss", "rank8", "rankqss")}
        if not pstd or not pwid or not all(ctrl.values()):
            print(bench, "ablation runs incomplete"); continue
        rows = {"vgs": out["vgs_l"], "wid_only": out["wid_l"], "pool": P.gate(out["route"], pstd, pwid), "std_only": out["std_l"],
                "qss_pool": pstd, "cosqss": ctrl["cosqss"], "rank8": ctrl["rank8"], "rankqss": ctrl["rankqss"], "rag": out["rag_l"]}
        qs = sorted(set.intersection(*[set(D) for D in rows.values()]))
        A[bench] = {k: {"acc": float(np.mean([D[q]["score"] for q in qs])),
                        "fsr": float(np.mean([D[q]["full"] for q in qs if D[q].get("n_gold")])),
                        "rounds": float(np.mean([D[q].get("n_loaded", 0) for q in qs]))} for k, D in rows.items()}
        for k in ("cosqss", "rankqss", "rank8"):
            d = np.array([rows["qss_pool"][q]["score"] - rows[k][q]["score"] for q in qs]); lo, hi = boot_ci(d)
            A[bench][k]["qss_minus"] = [float(d.mean()), lo, hi]
        A[bench]["n"] = len(qs)
        print(bench, {k: (round(v["acc"], 3), round(v["fsr"], 3)) for k, v in A[bench].items() if isinstance(v, dict)})
    json.dump(A, open(OUT / "ablation.json", "w"), indent=1)
    if len(A) < len(BENCH):
        return A
    HC = {"memeye": "hME", "memlens": "hML", "visualmem": "hVM", "camroll": "hCR"}
    NM = {"memeye": "MemEye", "memlens": "MemLens", "visualmem": "VisualMem", "camroll": "CamRoll"}
    shade = lambda d: "" if d > -1 else ("\\cellcolor{dA}" if d > -3 else ("\\cellcolor{dB}" if d > -5 else ("\\cellcolor{dC}" if d > -8 else "\\cellcolor{dD}")))  # noqa: E731
    def cell(k, ref):
        a = A_[k]["acc"]; d = 100 * (a - A_[ref]["acc"])
        return f"{shade(d)}{f3(a)}\\,{{\\fontsize{{6.4}}{{7}}\\selectfont${d:+.1f}$}}"
    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3.2pt}", "\\renewcommand{\\arraystretch}{1.06}",
         "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{l" + "c" * len(BENCH) + "}", "\\toprule",
         "\\textbf{Configuration} & " + " & ".join(f"\\cellcolor{{{HC[b]}}}\\textbf{{{NM[b] if len(BENCH) < 4 else SHORT[b]}}}" for b in BENCH) + " \\\\", "\\midrule"]
    L.append("\\oursrow \\vgs{} (full) & " + " & ".join(f3(A[b]["vgs"]["acc"]) for b in BENCH) + " \\\\")
    for lab, k in [("\\quad w/o widened path", "std_only"), ("\\quad w/o standard path", "wid_only"), ("\\quad w/o whole index", "pool")]:
        L.append(lab + " & " + " & ".join((globals().__setitem__("A_", A[b]) or cell(k, "vgs")) for b in BENCH) + " \\\\")
    L += ["\\midrule", "\\oursrow \\qss{}, 100-round pool & " + " & ".join(f3(A[b]["qss_pool"]["acc"]) for b in BENCH) + " \\\\"]
    for lab, k in [("\\quad w/o index reading", "cosqss"), ("\\quad w/o set output", "rankqss"), ("\\quad w/o adaptive size", "rank8")]:
        L.append(lab + " & " + " & ".join((globals().__setitem__("A_", A[b]) or cell(k, "qss_pool")) for b in BENCH) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}",
          "\\caption{\\textbf{Ablations and selection controls.} Accuracy with the lightweight reader and point changes from the highlighted reference. Removing a reading path sends every history to the other one; \\emph{w/o whole index} limits navigation to a 100-round cosine pool. On that pool, the selector is replaced by cosine retrieval of the same size (\\emph{w/o index reading}), by its own ranking cut at the same size (\\emph{w/o set output}), or by that ranking cut at eight rounds (\\emph{w/o adaptive size}); full-set recall and the remaining rows are in Table~\\ref{tab:ablation_full}." + (" ME, ML, VM, and CR denote MemEye, MemLens, VisualMem, and CamRoll." if len(BENCH) > 3 else "") + "}",
          "\\label{tab:ablation}", "\\end{table}"]
    (POL / "tab_ablation.tex").write_text("\n".join(L) + "\n")
    names = [("Cosine top-40 (RAG@40)", "rag"), ("Cosine top-$k$, $k$ = size of the \\qss{} set", "cosqss"), ("Model ranking, top 8", "rank8"),
             ("Model ranking, top $k$, $k$ = size of the \\qss{} set", "rankqss"), ("\\qss{}, 100-round pool", "qss_pool"),
             ("\\qss{}, whole index (standard path)", "std_only"), ("Widened path for every question", "wid_only"), ("\\vgs{} (visibility gate)", "vgs")]
    T = ["\\begin{table*}[t]", "\\centering", "\\small", "\\begin{tabularx}{\\linewidth}{X" + "|cc" * len(BENCH) + "}", "\\toprule",
         "\\multirow{2}{*}{\\textbf{Reading}} & " + " & ".join(f"\\multicolumn{{2}}{{c{'|' if b != BENCH[-1] else ''}}}{{\\cellcolor{{{HC[b]}}}\\textbf{{{NM[b]}}}}}" for b in BENCH) + " \\\\",
         " & " + " & ".join(f"\\cellcolor{{{HC[b]}}}FSR & \\cellcolor{{{HC[b]}}}Acc." for b in BENCH) + " \\\\", "\\midrule"]
    for i, (lab, k) in enumerate(names):
        if i == 4: T.append("\\midrule")
        pre = "\\oursrow " if k == "vgs" else ""
        T.append(pre + lab + " & " + " & ".join(f"{f3(A[b][k]['fsr'])} & {f3(A[b][k]['acc'])}" for b in BENCH) + " \\\\")
    T += ["\\bottomrule", "\\end{tabularx}",
          "\\caption{\\textbf{Full ablation with the lightweight reader.} Full-set recall (FSR) of the rounds handed to the reader and answer accuracy. Ranking and matched-size controls use the same 100-round candidate pool as \\qss{}; MemLens histories have at most 83 rounds, so the pool is the whole history.}",
          "\\label{tab:ablation_full}", "\\end{table*}"]
    (POL / "tab_ablation_full.tex").write_text("\n".join(T) + "\n")
    print("ablation tables written")
    return A


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ablation":
    ablation()


def extras():
    """Cost/latency figure, standard metrics and latency rows for RAG@40 and the compact-index VGS."""
    import shutil
    import p11_table
    import make_figures_p13 as M
    from p13_metrics import f1, bleu1, rank_metrics, ids, gold_rounds
    sub = set(json.load(open(ROOT / "results/p11_memlens_sub150.json")))
    new = {}
    for bench in BENCH:
        out, _ = P.compose(bench)
        for reader, rk, vk in (("lite", "rag_l", "vgs_l"), ("strong", "rag_s", "vgs_s")):
            add = P.CACHE_USD.get((bench, reader), 0.0) / 1000
            vgs = {q: dict(r, usd=r.get("usd", 0.0) + add) for q, r in out[vk].items()}
            new.setdefault(bench, {})[reader] = (out[rk], vgs)
    p11_table.ours = lambda bench: new[bench]
    lat = {tuple(k.split("|")): v for k, v in json.load(open(ROOT / "results/p13/latency_fig.json")).items()}
    LV = json.load(open(OUT / "latency_vgs.json")); S = json.load(open(OUT / "paper_stats.json"))
    for reader in ("lite", "strong"):
        lat[("vgs", reader)] = {"lat": LV["memeye"][reader][0], "acc": S["memeye"][reader]["vgs"]}
        lat[("rag", reader)]["acc"] = S["memeye"][reader]["rag"]
    M.fig_cost(lat)
    shutil.copy(M.FIG / "cost.pdf", FIGP / "cost.pdf")
    # standard metrics (lightweight / strong answers; evidence metrics from the lightweight path)
    rows = {}
    for bench in ("memeye", "memlens", "visualmem"):
        G = gold_rounds(bench)
        for key in ("rag", "vgs"):
            r = rows.setdefault(key, {})
            for reader in ("lite", "strong"):
                D = new[bench][reader][0 if key == "rag" else 1]
                qs = [q for q in D if bench != "memlens" or q in sub]
                if bench != "visualmem":
                    r[(bench, reader, "f1")] = float(np.mean([f1(D[q]["pred"], str(D[q]["gold"])) for q in qs]))
                    r[(bench, reader, "bleu")] = float(np.mean([bleu1(D[q]["pred"], str(D[q]["gold"])) for q in qs]))
            D = new[bench]["lite"][0 if key == "rag" else 1]
            qs = [q for q in D if (bench != "memlens" or q in sub) and G.get(q)]
            mets = [rank_metrics(ids(D[q]), G[q]) for q in qs]
            r[(bench, "r10")] = float(np.mean([m["r10"] for m in mets if m])); r[(bench, "ndcg")] = float(np.mean([m["ndcg10"] for m in mets if m]))
    json.dump({k: {"|".join(kk): v for kk, v in d.items()} for k, d in rows.items()}, open(OUT / "std_metrics.json", "w"), indent=1)
    t = (POL / "tab_std.tex").read_text().split("\n")
    for i, l in enumerate(t):
        for key, lab in (("rag", "RAG@40 &"), ("vgs", "\\oursrow \\vgs{} (ours) &")):
            if l.startswith(lab):
                r = rows[key]
                if "Recall" in "\n".join(t[:i]) and l.count("&") == 6:
                    t[i] = lab + " " + " & ".join(f3(r[(b, m)]) for m in ("r10", "ndcg") for b in BENCH) + " \\\\"
                elif l.count("&") == 4:
                    t[i] = lab + " " + " & ".join(f"{f3(r[(b, 'lite', m)])} / {f3(r[(b, 'strong', m)])}" for m in ("f1", "bleu") for b in ("memeye", "memlens")) + " \\\\"
    (POL / "tab_std.tex").write_text("\n".join(t))
    t = (POL / "tab_latency.tex").read_text().split("\n")
    for i, l in enumerate(t):
        if l.startswith("\\oursrow \\vgs{} (ours)"):
            c = l.split("&")
            v = [f"{LV[b]['lite'][0]:.1f} / {LV[b]['strong'][0]:.1f} & {LV[b]['lite'][1]:.1f}" for b in BENCH]
            t[i] = "\\oursrow \\vgs{} (ours) & " + " & ".join(v) + " &" + "&".join(c[7:])
    (POL / "tab_latency.tex").write_text("\n".join(t))
    print("extras written:", {k: {"|".join(kk): round(v, 3) for kk, v in d.items()} for k, d in rows.items()})


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "extras":
    extras()
