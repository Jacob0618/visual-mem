"""Figures for the ACL version (paper_acl/figures). No API calls."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper_acl/figures"
BUCKETS = [(1, 1, "1"), (2, 3, "2–3"), (4, 5, "4–5"), (6, 99, "≥6")]
plt.rcParams.update({"font.size": 8, "font.family": "serif"})


def fig_setsize():
    rows = json.load(open(ROOT / "results/mm2/set_analysis_rows.json"))
    qss = {r["qid"]: r for r in json.load(open(ROOT / "results/mm2/select_eval_toc38-40.json"))["per_q"]}
    series = [("Cosine top-10", [r["cos10"] for r in rows], "#bbbbbb", "o", "--"),
              ("Cosine top-40", [r["cos40"] for r in rows], "#777777", "s", "--"),
              ("Set selection (QSS), 7.3 rounds", [qss[r["qid"]]["full"] for r in rows], "#d1495b", "D", "-")]
    fig, ax = plt.subplots(figsize=(3.05, 1.75))
    x = np.arange(len(BUCKETS))
    for lab, vals, c, m, ls in series:
        ys = [np.mean([vals[i] for i, r in enumerate(rows) if lo <= r["g"] <= hi]) for lo, hi, _ in BUCKETS]
        ax.plot(x, ys, marker=m, color=c, label=lab, linestyle=ls, linewidth=1.3, markersize=4)
    ns = [sum(1 for r in rows if lo <= r["g"] <= hi) for lo, hi, _ in BUCKETS]
    ax.set_xticks(x, [f"{b[2]}\n(n={n})" for b, n in zip(BUCKETS, ns)])
    ax.set_xlabel("Gold evidence rounds per question")
    ax.set_ylabel("Full-set recall")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "setsize.pdf")
    for lab, vals, *_ in series:
        print(lab, [round(float(np.mean([vals[i] for i, r in enumerate(rows) if lo <= r["g"] <= hi])), 3) for lo, hi, _ in BUCKETS])


if __name__ == "__main__":
    fig_setsize()


def fig_visibility():
    """(a) masking question words in the MemEye index; (b) gain of the widened path over plain QSS by v_hat quartile."""
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.6), gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axes[0]
    p = [0, 0.5, 1.0]
    ax.plot(p, [0.903, 0.879, 0.814], marker="D", color="#d1495b", label="Full-set recall")
    ax.plot(p, [0.597, 0.604, 0.511], marker="o", color="#2e86ab", label="Accuracy (set selection)")
    ax.axhline(0.552, color="#2e86ab", linestyle="--", linewidth=1, label="Accuracy (RAG@40)")
    ax.set_xticks(p, ["0", "0.5", "1.0"]); ax.set_xlabel("Masking probability of question words")
    ax.set_ylim(0.45, 0.95); ax.legend(frameon=False, fontsize=5.8, loc="center", bbox_to_anchor=(0.5, 0.6))
    ax.set_title("(a) Hiding the evidence from the index (MemEye)", fontsize=7.5)
    ax = axes[1]
    q = ["Q1\n0.00–0.25", "Q2\n0.26–0.40", "Q3\n0.40–0.57", "Q4\n0.57–1.00"]
    g = np.array([0.026, 0.128, 0.052, 0.009]); lo = np.array([-0.043, 0.051, -0.026, -0.060]); hi = np.array([0.094, 0.205, 0.129, 0.078])
    ax.bar(range(4), g, color=["#9ecae1", "#3182bd", "#9ecae1", "#d9d9d9"], yerr=[g - lo, hi - g], capsize=2, error_kw={"linewidth": 0.8})
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(range(4), q, fontsize=6.3); ax.set_xlabel(r"Quartile of estimated visibility $\hat v(q)$")
    ax.set_ylabel("Accuracy gain over\nplain set selection", fontsize=6.8)
    ax.set_title("(b) Gain of the widened path (MemLens test)", fontsize=7.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "visibility.pdf")


def fig_cost():
    """Accuracy against cost per 1,000 questions on identical question sets (scripts/p11_table.cells).
    Marker = method, colour = reader (blue: lightweight, red: strong)."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from p11_table import cells, NAMES
    from matplotlib.lines import Line2D
    col = {"lite": "#2e86ab", "strong": "#d1495b"}
    mk = {"rag": ("o", "RAG@40"), "simplemem": ("^", "SimpleMem"), "omni": ("D", "Omni-SimpleMem"), "mirix": ("X", "MIRIX"),
          "vmem": ("v", "V-Mem"), "mma": ("P", "MMA"), "telemem": ("h", "TeleMem"), "vgs": ("*", "VGS (ours)")}
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.85))
    for ax, (bench, title) in zip(axes, [("memeye", "MemEye"), ("memlens", "MemLens-32k (150 questions)"), ("visualmem", "VisualMem")]):
        for reader in ("lite", "strong"):
            C = cells(bench, reader)
            for _, key in NAMES:
                v = C.get(key)
                if not v:
                    continue
                m, _ = mk[key]
                ax.scatter(v["usd"], v["acc"], color=col[reader], marker=m, s=60 if key == "vgs" else 22,
                           edgecolor="black", linewidth=0.5, zorder=3)
        ax.set_xscale("log")
        if bench == "memlens":
            ax.set_xlim(0.8, 6000); ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1k"], fontsize=6)
        else:
            ax.set_xlim(0.5, 150); ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"], fontsize=6)
        ax.minorticks_off(); ax.tick_params(labelsize=6)
        ax.set_title(title, fontsize=7)
    axes[1].set_xlabel("Cost per 1,000 questions (USD, log scale, construction included)", fontsize=6.6)
    axes[0].set_ylabel("Accuracy", fontsize=7)
    handles = [Line2D([], [], color="#888888", marker=mk[k][0], linestyle="", markersize=6 if k == "vgs" else 4.5,
                      markeredgecolor="black", markeredgewidth=0.5, label=mk[k][1]) for _, k in NAMES]
    handles += [Line2D([], [], color=col[r], marker="s", linestyle="", markersize=4.5, label=lab)
                for r, lab in (("lite", "lightweight reader"), ("strong", "strong reader"))]
    fig.legend(handles=handles, loc="upper center", ncol=10, fontsize=5.2, frameon=False, bbox_to_anchor=(0.5, 1.02),
               handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "cost.pdf")


def fig_scale():
    """Accuracy on the original MemEye histories and on histories enlarged 32-fold (about 4,088 rounds)."""
    vals = {"Lightweight reader": {"Full context": (0.495, 0.368), "RAG@40": (0.488, 0.471), "Set selection": (0.541, 0.538)},
            "Strong reader": {"RAG@40": (0.741, 0.730), "Set selection": (0.761, 0.764)}}
    col = {"Full context": ("#222222", "s", ":"), "RAG@40": ("#888888", "o", "--"), "Set selection": ("#d1495b", "D", "-")}
    fig, axes = plt.subplots(1, 2, figsize=(3.05, 1.65))
    for ax, (title, d) in zip(axes, vals.items()):
        for lab, (a0, a1) in d.items():
            c, m, ls = col[lab]
            ax.plot([0, 1], [a0, a1], marker=m, color=c, linestyle=ls, label=lab, linewidth=1.2, markersize=3.5)
        ax.set_xticks([0, 1], ["126\nrounds", "4,088\nrounds"], fontsize=6.5)
        ax.set_xlim(-0.25, 1.25)
        ax.set_title(title, fontsize=7.2)
        ax.tick_params(labelsize=6.5)
    axes[0].legend(frameon=False, fontsize=5.8, loc="lower left")
    axes[0].set_ylabel("Accuracy", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "scale.pdf")


if __name__ == "__main__":
    fig_visibility()
    fig_cost()
    fig_scale()
