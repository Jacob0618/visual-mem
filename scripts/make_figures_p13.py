"""Phase 13 figures: compact full-width rows of small panels (paper_acl/figures). No API calls.
Numbers come from result files where a loader exists, otherwise from the audited tables named in the comments.
python scripts/make_figures_p13.py [main|appendix|all]"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper_acl/figures"
sys.path.insert(0, str(ROOT / "scripts"))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7, "axes.titlesize": 7.4, "axes.labelsize": 6.8, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
    "legend.fontsize": 6.4, "axes.linewidth": 0.7, "lines.linewidth": 1.3, "lines.markersize": 3.8,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "pdf.fonttype": 42})
W = 6.3                      # ACL text width (inches)
RED, BLUE, GRAY, DARK = "#c0504d", "#4f81bd", "#9a9a9a", "#333333"
TEAL, ORANGE, PURPLE, GREEN, OLIVE = "#3a9a9a", "#e0973a", "#8e6bb5", "#7fae6a", "#b5a642"
EDGE = {"markeredgecolor": "black", "markeredgewidth": 0.45}
BAR = {"edgecolor": "black", "linewidth": 0.45}
ERR = {"elinewidth": 0.6, "capsize": 1.6, "capthick": 0.6, "ecolor": "#222222"}
METHODS = [("rag", "RAG@40", BLUE, "o"), ("fc", "Full context", DARK, "s"), ("simplemem", "SimpleMem", ORANGE, "^"), ("omni", "Omni-SimpleMem", OLIVE, "D"),
           ("mirix", "MIRIX", PURPLE, "X"), ("vmem", "V-Mem", TEAL, "v"), ("mma", "MMA", GREEN, "P"),
           ("telemem", "TeleMem", GRAY, "h"), ("vgs", "VGS (ours)", RED, "*")]


def row(n, h=1.24, w=W, wr=None):
    fig, axes = plt.subplots(1, n, figsize=(w, h), gridspec_kw={"width_ratios": wr} if wr else None)
    return fig, np.atleast_1d(axes)


def finish(fig, name, handles=None, ncol=None, legend_h=0.14, pad=0.25):
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=ncol or len(handles), frameon=False,
                   bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=0.9, handlelength=1.4,
                   fontsize=6.0)
        fig.tight_layout(pad=pad, w_pad=0.9, rect=(0, legend_h, 1, 1))
    else:
        fig.tight_layout(pad=pad, w_pad=0.9)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def errbars(ax, x, y, lo, hi, **kw):
    y, lo, hi = map(np.asarray, (y, lo, hi))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", **ERR, **kw)


# =============================================================================================== main text
def fig_problem():
    """Section 2. (a) retrieval vs full context on text and multimodal memory (tab:prelim); (b) visual dependence;
    (c) full-set recall by gold set size; (d) rounds opened by gold set size (MemEye)."""
    fig, ax = row(4)
    # (a) tab:prelim, strong reader
    names = ["LoCoMo", "LongMemEval-S", "MemEye"]
    small = [(-0.084, -0.105, -0.064), (-0.050, -0.092, -0.017), (-0.140, -0.184, -0.098)]   # MemEye: leak-free prompt (results/p16/sec2)
    large = [(-0.009, -0.023, 0.006), (-0.008, -0.034, 0.017), (-0.046, -0.075, -0.018)]
    x = np.arange(3)
    for off, d, c, lab in [(-0.19, small, "#bcd2e8", "top-10"), (0.19, large, BLUE, "top-40/50/100")]:
        v = [t[0] for t in d]
        ax[0].bar(x + off, v, 0.36, color=c, **BAR, label=lab)
        errbars(ax[0], x + off, v, [t[1] for t in d], [t[2] for t in d])
    ax[0].axhline(0, color="black", linewidth=0.6)
    ax[0].set_xticks(x, ["LoCoMo", "LongMem.", "MemEye"])
    ax[0].set_ylabel("Accuracy minus full context")
    ax[0].set_title("(a) Cosine retrieval")
    ax[0].legend(frameon=False, loc="lower left", fontsize=5.8, handlelength=1.0)
    ax[0].set_ylim(-0.19, 0.03)
    # (b) visual dependence (tab:prelim; tab:vmscope; Mem-Gallery half subset, RESEARCH_LOG P3)
    vd = [("MemEye", 0.810, 0.216), ("VisualMem", 0.609, 0.297), ("Mem-Gal.", 0.909, 0.892)]
    for i, (lab, w, wo) in enumerate(vd):
        ax[1].bar(i - 0.19, w, 0.36, color=BLUE, **BAR)
        ax[1].bar(i + 0.19, wo, 0.36, color="#d9d9d9", **BAR)
    ax[1].set_xticks(range(3), [v[0] for v in vd], fontsize=5.7)
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(0, 1.22)
    ax[1].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax[1].set_title("(b) Removing visual input")
    ax[1].legend(handles=[Patch(color=BLUE, label="with images"), Patch(color="#d9d9d9", label="without")],
                 frameon=False, loc="upper center", fontsize=5.8, handlelength=1.0, ncol=2, columnspacing=0.6)
    # (c, d) set analysis (results/mm2/set_analysis_rows.json, select_eval_toc38-40.json)
    rows = json.load(open(ROOT / "results/p16/set_analysis_rows_v2.json"))
    qss = {r["qid"]: r for r in json.load(open(ROOT / "results/p16/select_eval_compact_std.json"))["per_q"]}
    B = [(1, 1, "1"), (2, 3, "2–3"), (4, 5, "4–5"), (6, 99, "≥6")]
    xb = np.arange(len(B))
    mean = lambda f: [float(np.mean([f(r) for r in rows if lo <= r["g"] <= hi])) for lo, hi, _ in B]
    for lab, f, c, m, ls in [("cosine top-10", lambda r: r["cos10"], "#bcd2e8", "o", "--"),
                              ("cosine top-40", lambda r: r["cos40"], BLUE, "s", "--"),
                              ("set selection", lambda r: qss[r["qid"]]["full"], RED, "D", "-")]:
        ax[2].plot(xb, mean(f), marker=m, color=c, linestyle=ls, label=lab, **EDGE)
    ax[2].set_xticks(xb, [b[2] for b in B])
    ax[2].set_xlabel("Gold rounds per question")
    ax[2].set_ylabel("Full-set recall")
    ax[2].set_ylim(0, 1.04)
    ax[2].set_title("(c) Complete evidence sets")
    ax[2].legend(frameon=False, loc="lower left", fontsize=5.2, handlelength=1.4, labelspacing=0.15,
                 bbox_to_anchor=(-0.03, -0.03))
    nq = mean(lambda r: qss[r["qid"]]["n"])
    ax[3].plot(xb, nq, marker="D", color=RED, label="set selection", **EDGE)
    for k, c in [(10, "#bcd2e8"), (40, BLUE)]:
        ax[3].axhline(k, color=c, linestyle="--", linewidth=1.0, label=f"cosine top-{k}")
    ax[3].set_yscale("log")
    ax[3].set_yticks([2, 5, 10, 20, 40], ["2", "5", "10", "20", "40"])
    ax[3].minorticks_off()
    ax[3].set_xticks(xb, [b[2] for b in B])
    ax[3].set_xlabel("Gold rounds per question")
    ax[3].set_ylabel("Rounds opened")
    ax[3].set_title("(d) Budget follows the set")
    ax[3].legend(frameon=False, loc="lower right", fontsize=5.4, handlelength=1.6, labelspacing=0.15)
    finish(fig, "problem")
    print("  FSR by bucket:", [round(v, 3) for v in mean(lambda r: qss[r["qid"]]["full"])], "rounds:", [round(v, 1) for v in nq])


def fig_gains():
    """Section 5: (a, b) longer histories (tab:robust); (c, d) parts of the widened path (former Table 3)."""
    fig, ax = row(4)
    sc = {"light": {"Full context": (0.495, 0.368), "RAG@40": (0.488, 0.471), "Set selection": (0.541, 0.538)},
          "strong": {"RAG@40": (0.741, 0.730), "Set selection": (0.761, 0.764)}}
    sty = {"Full context": (DARK, "s", ":"), "RAG@40": (BLUE, "o", "--"), "Set selection": (RED, "D", "-")}
    for a, (rd, title, ylim) in zip(ax[:2], [("light", "(a) Longer histories, light", (0.33, 0.58)),
                                              ("strong", "(b) Longer histories, strong", (0.70, 0.79))]):
        for lab, (y0, y1) in sc[rd].items():
            c, m, ls = sty[lab]
            a.plot([0, 1], [y0, y1], marker=m, color=c, linestyle=ls, label=lab, **EDGE)
        a.set_xticks([0, 1], ["126", "4,088"])
        a.set_xlim(-0.3, 1.3)
        a.set_ylim(*ylim)
        a.set_xlabel("Rounds per history")
        a.set_title(title)
    ax[0].set_ylabel("Accuracy")
    abl = [("RAG@40", 0.488, 0.466, BLUE), ("Selection,\n200 chars", 0.541, 0.483, "#f2b8b5"),
           ("Selection,\n600 chars", 0.528, 0.479, "#e48a86"), ("+ notes", 0.517, 0.536, RED),
           ("Cosine-20\n+ notes", 0.471, 0.485, "#bcd2e8"), ("VGS\n(gate)", 0.541, 0.536, "#7a1f1d")]
    for a, k, title, ylim in [(ax[2], 1, "(c) Readings on MemEye", (0.44, 0.555)),
                              (ax[3], 2, "(d) Readings on MemLens", (0.44, 0.555))]:
        v = [t[k] for t in abl]
        a.bar(range(len(abl)), v, 0.72, color=[t[3] for t in abl], **BAR)
        for i, y in enumerate(v):
            a.text(i, y + 0.0015, f"{y:.3f}".lstrip("0"), ha="center", va="bottom", fontsize=5.0)
        a.set_xticks([])
        a.set_ylim(*ylim)
        a.set_title(title)
    ax[2].set_ylabel("Accuracy (light reader)")
    hl = [Line2D([], [], color=sty[k][0], marker=sty[k][1], linestyle=sty[k][2], label=k, **EDGE) for k in sty]
    hb = [Patch(facecolor=t[3], edgecolor="black", linewidth=0.45, label=t[0].replace("\n", " ")) for t in abl]
    finish(fig, "gains", hl + hb, ncol=9, legend_h=0.1)


def fig_visibility():
    """Section 5: (a, b) masking question words in the MemEye index (tab:visibility); (c) widened-path gain by
    quartile of v_hat (MemLens test); (d) VGS minus RAG@40 below and above the gate (tab:gate, tab:vmpath)."""
    fig, ax = row(4)
    p = [0, 0.5, 1.0]
    ax[0].plot(p, [0.903, 0.879, 0.814], marker="D", color=RED, label="set selection", **EDGE)
    ax[0].set_ylabel("Full-set recall")
    ax[0].set_ylim(0.78, 0.93)
    ax[0].set_title("(a) Masking: recall")
    ax[1].plot(p, [0.597, 0.604, 0.511], marker="D", color=RED, label="set selection", **EDGE)
    ax[1].axhline(0.552, color=BLUE, linestyle="--", linewidth=1.0, label="RAG@40")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(0.49, 0.63)
    ax[1].set_title("(b) Masking: accuracy")
    ax[1].legend(frameon=False, loc="lower left", fontsize=5.8, handlelength=1.6)
    for a in ax[:2]:
        a.set_xticks(p, ["0", "0.5", "1"])
        a.set_xlabel("Masking probability $p$")
    g = np.array([0.026, 0.128, 0.052, 0.009])
    lo = np.array([-0.043, 0.051, -0.026, -0.060])
    hi = np.array([0.094, 0.205, 0.129, 0.078])
    ax[2].bar(range(4), g, 0.65, color=["#f2b8b5", RED, "#f2b8b5", "#e6e6e6"], **BAR)
    errbars(ax[2], range(4), g, lo, hi)
    ax[2].axhline(0, color="black", linewidth=0.6)
    ax[2].set_xticks(range(4), ["Q1", "Q2", "Q3", "Q4"])
    ax[2].set_xlabel(r"Quartile of $\hat v(q)$ (MemLens)")
    ax[2].set_ylabel("Gain over plain selection")
    ax[2].set_title("(c) Notes help at low visibility")
    gate = [("MemEye", None, (0.053, 0.009, 0.098)), ("MemLens", (0.084, 0.042, 0.125), (-0.017, -0.136, 0.085)),
            ("VisualMem", (0.176, 0.123, 0.227), (0.025, -0.043, 0.093))]
    for i, (lab, below, above) in enumerate(gate):
        for off, t, c in [(-0.19, below, RED), (0.19, above, "#d9d9d9")]:
            if t:
                ax[3].bar(i + off, t[0], 0.36, color=c, **BAR)
                errbars(ax[3], [i + off], [t[0]], [t[1]], [t[2]])
    ax[3].axhline(0, color="black", linewidth=0.6)
    ax[3].set_xticks(range(3), [g_[0] for g_ in gate])
    ax[3].set_ylabel("VGS minus RAG@40")
    ax[3].set_title("(d) Gains by gate side")
    ax[3].set_ylim(-0.16, 0.33)
    ax[3].legend(handles=[Patch(color=RED, label=r"$\hat v<0.7$"), Patch(color="#d9d9d9", label=r"$\hat v\geq0.7$")],
                 frameon=False, loc="upper left", fontsize=5.6, handlelength=1.0, ncol=2, columnspacing=0.6)
    finish(fig, "visibility")


def fig_gate():
    """Section 5.4: (a, b) masking the question words in the index on MemEye (lightweight reader with thinking,
    tab:visibility) and MemLens (phase-14 runs, p = 0 and 1); (c) fixed readings against the gate for both readers
    (results/p14/gate_v2.json, results/p14/gate.json); (d) sensitivity to the threshold tau."""
    import glob
    G2 = json.load(open(ROOT / "results/p14/gate_v2.json"))
    GS = json.load(open(ROOT / "results/p14/gate.json"))["_strong"]
    fig, ax = row(4, h=1.42, wr=[1.0, 1.0, 1.45, 0.95])
    # (a, b) index manipulations: question words masked (p = 1) and image descriptions removed from the index lines
    from p11_analyze import L as LD
    keep = set(json.load(open(ROOT / "results/p14/vm_mask_qids.json")))
    V = ROOT / "results/p14/runs/visualmem"
    sub = lambda D: {q: r for q, r in D.items() if q in keep}  # noqa: E731
    vm0, vrag = sub(LD(V / "toc38-40__read-medium__gemini-2.5-flash-lite__xcache.jsonl")), sub(LD(V / "rag-40__read-medium__gemini-2.5-flash-lite.jsonl"))
    vm1 = LD(ROOT / "results/p14/runs/visualmem_mask/toc38-40__read-medium__gemini-2.5-flash-lite__mask1.0.jsonl")
    vmt = LD(next((ROOT / "results/p14/runs/visualmem_mask").glob("*toctext*.jsonl")))
    d = ROOT / "results/p14/runs/memlens_v2"
    ml0, ml1 = LD(d / "toc38-40__read-medium__gemini-2.5-flash-lite.jsonl"), LD(d / "toc38-40__read-medium__gemini-2.5-flash-lite__mask1.0.jsonl")
    from p11_analyze import M as MM
    test = set(json.load(open(ROOT / "results/memlens_test.json")))
    mrag = {q: r for q, r in LD(MM / "memlens_32k/rag-40__read-medium__gemini-2.5-flash-lite.jsonl").items() if q in test}
    fsr = lambda D, Q: float(np.mean([D[q]["full"] for q in Q if D[q].get("n_gold")]))  # noqa: E731
    dacc = lambda D, R, Q: float(np.mean([D[q]["score"] - R[q]["score"] for q in Q]))  # noqa: E731
    qv = sorted(set(vm0) & set(vm1) & set(vmt) & set(vrag)); ql = sorted(set(ml0) & set(ml1) & set(mrag))
    # MemEye: tab:visibility (lightweight reader with thinking; RAG@40 with thinking 0.552) and the text-only index
    # (results/mm2/select_eval_toc38-40_toctext.json, selection only)
    F = {"MemEye": [0.903, 0.814, 0.906], "MemLens": [fsr(ml0, ql), fsr(ml1, ql), None],
         "VisualMem": [fsr(vm0, qv), fsr(vm1, qv), fsr(vmt, qv)]}
    A = {"MemEye": [0.597 - 0.552, 0.511 - 0.552, None], "MemLens": [dacc(ml0, mrag, ql), dacc(ml1, mrag, ql), None],
         "VisualMem": [dacc(vm0, vrag, qv), dacc(vm1, vrag, qv), dacc(vmt, vrag, qv)]}
    cols3 = [RED, "#f2b8b5", "#d9d9d9"]
    labs3 = ["whole index", "question words masked", "no image descriptions"]
    for a, Dd in ((ax[0], F), (ax[1], A)):
        for i, (bn, vals) in enumerate(Dd.items()):
            for j, (v, c) in enumerate(zip(vals, cols3)):
                if v is not None:
                    a.bar(i + (j - 1) * 0.27, v, 0.27, color=c, **BAR)
        a.set_xticks(range(3), ["MemEye", "MemLens", "VisualM."], fontsize=5.6)
    ax[0].set_ylabel("Full-set recall")
    ax[0].set_ylim(0, 1.0)
    ax[1].axhline(0, color="black", linewidth=0.6)
    ax[1].set_ylabel("Accuracy minus RAG@40")
    ax[0].set_title("(a) Hiding evidence: recall")
    ax[1].set_title("(b) Hiding evidence: accuracy")
    hand = [Patch(facecolor=c, edgecolor="black", linewidth=0.45, label=l) for c, l in zip(cols3, labs3)]
    print("  manipulations FSR:", {k: [None if x is None else round(x, 3) for x in v] for k, v in F.items()})
    print("  manipulations acc-RAG:", {k: [None if x is None else round(x, 3) for x in v] for k, v in A.items()})
    # (c) fixed readings against the gate
    groups = [("ME\nlight", G2["memeye"], "l"), ("ML\nlight", G2["memlens"], "l"), ("VM\nlight", G2["visualmem"], "l"),
              ("ME\nstrong", GS["memeye"], "s"), ("ML\nstrong", GS["memlens"], "s")]
    cols = ["#fbe3a8", "#bcd2e8", "#7a1f1d"]
    for i, (lab, g, kind) in enumerate(groups):
        if kind == "l":
            vals = [g["always_standard"] - g["rag"], g["always_widened"] - g["rag"], g["gate"] - g["rag"]]
        else:
            vals = [g["always_qss"] - g["always_rag"], 0.0, g["gate"] - g["always_rag"]]
        for j, (v, c) in enumerate(zip(vals, cols)):
            ax[2].bar(i + (j - 1) * 0.26, v, 0.26, color=c, **BAR)
    ax[2].axhline(0, color="black", linewidth=0.6)
    ax[2].set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=5.6)
    ax[2].set_ylabel("Accuracy minus RAG@40")
    ax[2].set_title("(c) Fixed readings and the gate")
    hand += [Patch(facecolor=c, edgecolor="black", linewidth=0.45, label=l) for c, l in
             zip(cols, ["standard path for all", "widened path for all", "VGS gate"])]
    # (d) threshold sensitivity
    curves = [("ME light", G2["memeye"]["tau_curve"], RED, "-"), ("ML light", G2["memlens"]["tau_curve"], BLUE, "-"),
              ("VM light", G2["visualmem"]["tau_curve"], TEAL, "-"),
              ("ME strong", GS["memeye"]["tau_curve"], RED, "--"), ("ML strong", GS["memlens"]["tau_curve"], BLUE, "--")]
    for lab, tc, c, ls in curves:
        t = np.array([float(k) for k in tc]); v = np.array(list(tc.values()))
        ax[3].plot(t, v - tc["0.70"], color=c, linestyle=ls, linewidth=1.0, label=lab)
    ax[3].axvline(0.7, color="black", linestyle=":", linewidth=0.7)
    ax[3].axhline(0, color="black", linewidth=0.5)
    ax[3].set_xlabel(r"Threshold $\tau$")
    ax[3].set_ylabel(r"Accuracy minus $\tau{=}0.7$")
    ax[3].set_title(r"(d) Sensitivity to $\tau$")
    ax[3].set_xlim(0.3, 1.0)
    ax[3].set_ylim(-0.04, 0.012)
    finish(fig, "gate", hand, ncol=6, legend_h=0.11)


def fig_cost(latency=None):
    """Section 5: accuracy against cost per 1,000 questions on the question sets of Table 2 (scripts/p11_table.cells),
    and, if latency results exist, accuracy against median online latency (results/p13/latency_summary.json)."""
    from p11_table import cells
    col = {"lite": BLUE, "strong": RED}
    n = 4 if latency else 3
    fig, ax = row(n, h=1.32)
    for a, (bench, title) in zip(ax, [("memeye", "(a) MemEye"), ("memlens", "(b) MemLens-32k (150)"), ("visualmem", "(c) VisualMem")]):
        for reader in ("lite", "strong"):
            C = cells(bench, reader)
            for key, lab, _, m in METHODS:
                v = C.get(key)
                if v:
                    a.scatter(v["usd"], v["acc"], color=col[reader], marker=m, s=46 if key == "vgs" else 17,
                              edgecolor="black", linewidth=0.45, zorder=3)
        a.set_xscale("log")
        if bench == "memlens":
            a.set_xlim(0.25, 6000)
            a.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1k"])
        else:
            a.set_xlim(0.2, 150)
            a.set_xticks([1, 10, 100], ["1", "10", "100"])
        a.minorticks_off()
        a.set_xlabel("USD per 1k questions")
        a.set_title(title)
    ax[0].set_ylabel("Accuracy")
    if latency:
        a = ax[3]
        for (key, reader), v in latency.items():
            m = dict((k, mm) for k, _, _, mm in METHODS)[key]
            a.scatter(v["lat"], v["acc"], color=col[reader], marker=m, s=46 if key == "vgs" else 17,
                      edgecolor="black", linewidth=0.45, zorder=3)
        a.set_xscale("log")
        a.minorticks_off()
        a.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
        a.set_xlabel("Seconds per question (median)")
        a.set_title("(d) Latency (MemEye)")
    handles = [Line2D([], [], color="#bbbbbb", marker=m, linestyle="", markersize=5.5 if k == "vgs" else 4,
                      label=lab, **EDGE) for k, lab, _, m in METHODS]
    handles += [Line2D([], [], color=col[r], marker="s", linestyle="", markersize=4, label=lab, **EDGE)
                for r, lab in (("lite", "light reader"), ("strong", "strong reader"))]
    finish(fig, "cost", handles, ncol=11, legend_h=0.12)
    return fig


def main_figs():
    fig_problem()
    if (ROOT / "results/p14/gate_v2.json").exists():
        fig_gate()
    lat = ROOT / "results/p13/latency_fig.json"
    L = None
    if lat.exists():
        L = {tuple(k.split("|")): v for k, v in json.load(open(lat)).items()}
    fig_cost(L)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "main"
    if what in ("main", "all"):
        main_figs()
    if what in ("appendix", "all"):
        import make_figures_p13_app
        make_figures_p13_app.main()
