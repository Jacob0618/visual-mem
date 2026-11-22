"""Phase 13 appendix figures that replace tables (paper_acl/figures/app_*.pdf). No API calls.
Values are those of the audited appendix tables named in each docstring (they were computed from results/ by the
scripts listed in Appendix Reproducibility); the paired baseline differences are recomputed by scripts/p11_table.py."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures_p13 import (BAR, BLUE, DARK, EDGE, ERR, GRAY, GREEN, METHODS, OLIVE, ORANGE, PURPLE, RED, TEAL, W,  # noqa: E402
                              errbars, finish, plt, row)

LIGHTRED, LIGHTBLUE = "#f2b8b5", "#bcd2e8"


def hbars(ax, labels, vals, colors, fmt="{:.3f}", xlim=None, fs=5.6):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, 0.7, color=colors, **BAR)
    for yi, v in zip(y, vals):
        ax.text(v + (xlim[1] - xlim[0]) * 0.01 if xlim else v, yi, fmt.format(v), va="center", fontsize=fs - 0.4)
    ax.set_yticks(y, labels, fontsize=fs)
    if xlim:
        ax.set_xlim(*xlim)


def app_keys_reader():
    """tab:keys and tab:reader (MemEye, 371 questions)."""
    fig, ax = row(2, h=1.55, w=W, wr=[1, 1.25])
    keys = [("Image only", 0.300), ("Dialogue text only", 0.538), ("Text + images, one embedding", 0.671),
            ("Text + dense caption", 0.774), ("Text + short caption", 0.774)]
    hbars(ax[0], [k for k, _ in keys], [v for _, v in keys], [GRAY, LIGHTBLUE, LIGHTBLUE, BLUE, BLUE], xlim=(0, 0.9))
    ax[0].set_xlabel("Coverage@10 of gold rounds")
    ax[0].set_title("(a) Retrieval keys")
    rd = {"3.8-flash": [0.819, 0.689, 0.741, 0.746], "3.1-flash-lite": [0.639, 0.601, 0.612, 0.694],
          "2.5-flash-lite": [0.495, 0.446, 0.488, 0.539]}
    inputs = ["Full context", "RAG@10", "RAG@40", "Annotated evidence"]
    cols = [DARK, LIGHTBLUE, BLUE, RED]
    x = np.arange(3)
    for j, (inp, c) in enumerate(zip(inputs, cols)):
        ax[1].bar(x + (j - 1.5) * 0.2, [rd[r][j] for r in rd], 0.2, color=c, label=inp, **BAR)
    ax[1].set_xticks(x, list(rd))
    ax[1].set_ylim(0.4, 0.86)
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("(b) Readers and inputs")
    ax[1].legend(frameon=False, ncol=2, fontsize=5.6, loc="upper right", handlelength=1.0, columnspacing=0.6)
    finish(fig, "app_keys_reader")


def app_cluster():
    """tab:cluster: question-level and scenario-level 95% intervals for the main MemEye contrasts."""
    acc = [("Light: QSS − RAG@40", 0.053, (0.009, 0.098), (-0.014, 0.120), "6/8"),
           ("Light, thinking: QSS − RAG@40", 0.045, (0.003, 0.086), (-0.005, 0.095), "6/8"),
           ("Strong: QSS + index − RAG@40", 0.055, (0.021, 0.089), (0.004, 0.126), "6/8"),
           ("Strong: QSS − RAG@40", 0.020, (-0.014, 0.055), (-0.030, 0.073), "6/8"),
           ("Strong: QSS + index − full context", -0.024, (-0.050, 0.002), (-0.061, 0.022), "1/8")]
    fsr = [("QSS − cosine, same size", 0.364, (0.310, 0.418), (0.221, 0.551), "8/8"),
           ("QSS (P=100) − usefulness ranking", 0.059, (0.032, 0.089), (0.008, 0.111), "5/8")]
    fig, ax = row(2, h=1.7, wr=[1.5, 1])
    for a, rows_, title, xl in [(ax[0], acc, "(a) Accuracy contrasts", "Paired difference (accuracy)"),
                                (ax[1], fsr, "(b) Full-set recall contrasts", "Paired difference (FSR)")]:
        y = np.arange(len(rows_))[::-1]
        for yi, (lab, m, q, s, k) in zip(y, rows_):
            a.plot([q[0], q[1]], [yi + 0.13] * 2, color=BLUE, linewidth=1.6, solid_capstyle="butt")
            a.plot([s[0], s[1]], [yi - 0.13] * 2, color=RED, linewidth=1.6, solid_capstyle="butt")
            a.plot([m], [yi], marker="D", color="white", markersize=3.2, **EDGE)
            a.text(1.02, yi, k, transform=a.get_yaxis_transform(), va="center", fontsize=5.6)
        a.axvline(0, color="black", linewidth=0.6)
        a.set_yticks(y, [r[0] for r in rows_], fontsize=5.8)
        a.set_xlabel(xl)
        a.set_title(title)
    hs = [Line2D([], [], color=BLUE, linewidth=1.6, label="question-level 95% CI"),
          Line2D([], [], color=RED, linewidth=1.6, label="scenario-level 95% CI"),
          Line2D([], [], color="white", marker="D", linestyle="", label="mean difference", **EDGE)]
    finish(fig, "app_cluster", hs, legend_h=0.1)


def app_robust():
    """tab:robust: enlarged MemEye histories and Mem-Gallery."""
    fig, ax = row(3, h=1.45, wr=[1.3, 0.8, 1])
    meth = ["Full context", "RAG@20", "RAG@40", "QSS"]
    cols = [DARK, LIGHTBLUE, BLUE, RED]
    data = [("(a) MemEye+1k, light reader", [0.368, 0.451, 0.471, 0.538], (0.3, 0.6)),
            ("(b) MemEye+1k, strong", [None, None, 0.730, 0.764], (0.65, 0.8)),
            ("(c) Mem-Gallery", None, (0.78, 0.94))]
    for a, (title, v, yl) in zip(ax[:2], data[:2]):
        xs = [i for i, x in enumerate(v) if x is not None]
        a.bar(range(len(xs)), [v[i] for i in xs], 0.7, color=[cols[i] for i in xs], **BAR)
        for k, i in enumerate(xs):
            a.text(k, v[i] + (yl[1] - yl[0]) * 0.015, f"{v[i]:.3f}".lstrip("0"), ha="center", fontsize=5.0)
        a.set_xticks([])
        a.set_ylim(*yl)
        a.set_title(title)
    ax[0].set_ylabel("Accuracy")
    mg = {"light": [0.817, None, 0.839, 0.852], "strong": [0.917, None, 0.912, 0.908]}
    for j, (rd, vals) in enumerate(mg.items()):
        xs = [i for i, x in enumerate(vals) if x is not None]
        for k, i in enumerate(xs):
            ax[2].bar(j * 4 + k, vals[i], 0.75, color=cols[i], **BAR)
            ax[2].text(j * 4 + k, vals[i] + 0.002, f"{vals[i]:.3f}".lstrip("0"), ha="center", fontsize=4.8)
    ax[2].set_xticks([1, 5], ["light reader", "strong reader"])
    ax[2].set_ylim(*data[2][2])
    ax[2].set_title(data[2][0])
    hb = [Patch(facecolor=c, edgecolor="black", linewidth=0.45, label=m) for m, c in zip(meth, cols)]
    finish(fig, "app_robust", hb, legend_h=0.1)


def app_ablate():
    """Selector models under the shared prompt (MemEye, selection only)."""
    items = [("gemini-3.8-flash (QSS)", 0.903, 7.3), ("gemini-3.5-flash-lite", 0.685, 3.9), ("gemini-3.1-flash-lite", 0.660, 4.3),
             ("gemini-2.5-flash-lite", 0.712, 9.8), ("2.5-flash-lite, thinking", 0.636, 4.6), ("gpt-5-mini", 0.728, 5.9),
             ("gpt-5$^\\ddagger$", 0.573, 2.8), ("Qwen3-235B, thinking$^\\ddagger$", 0.407, 2.3), ("Qwen3-235B, instruct", 0.264, 1.5)]
    fig, ax = row(1, h=1.7, w=W * 0.5)
    a = ax[0]
    y = np.arange(len(items))[::-1]
    a.barh(y, [v for _, v, _ in items], 0.7, color=[RED] + [LIGHTRED] * (len(items) - 1), **BAR)
    for yi, (_, v, n) in zip(y, items):
        a.text(v + 0.01, yi, f"{v:.3f} ({n:g})", va="center", fontsize=5.0)
    a.set_yticks(y, [l for l, _, _ in items], fontsize=5.6)
    a.set_xlim(0, 1.15)
    a.set_xticks([0, 0.5, 1.0])
    a.set_xlabel("Full-set recall (rounds opened)", fontsize=6)
    finish(fig, "app_ablate")


def app_dev():
    """tab:p5dev (development) and tab:p5test (registered test, paired differences from RAG@40)."""
    fig, ax = row(2, h=1.85, wr=[1.15, 1])
    mech = [("RAG@40", 0.488, 0.489), ("RAG@40, thinking", 0.552, 0.515), ("QSS, 200 chars", 0.541, 0.485),
            ("  + session closure", 0.526, 0.476), ("QSS, model gist", 0.504, 0.515), ("QSS, 600 chars", 0.528, 0.549),
            ("  + session closure", None, 0.472), ("  + notes", 0.517, 0.562), ("Cosine-20 + notes", 0.471, 0.536),
            ("Sentence top-20", None, 0.489), ("VGS (gate)", 0.541, 0.558)]
    y = np.arange(len(mech))[::-1]
    for off, k, c, lab in [(0.18, 1, BLUE, "MemEye"), (-0.18, 2, RED, "MemLens dev")]:
        vals = [m[k] for m in mech]
        yy = [yi + off for yi, v in zip(y, vals) if v is not None]
        ax[0].barh(yy, [v for v in vals if v is not None], 0.34, color=c, label=lab, **BAR)
    ax[0].set_yticks(y, [m[0] for m in mech], fontsize=5.6)
    ax[0].set_xlim(0.44, 0.615)
    ax[0].set_xlabel("Accuracy (light reader)")
    ax[0].set_title("(a) Development")
    ax[0].legend(frameon=False, fontsize=5.6, loc="upper right", handlelength=1.0)
    test = [("RAG@40, thinking", 0.030, -0.013, 0.073, BLUE), ("QSS, 200 chars", 0.017, -0.019, 0.054, LIGHTRED),
            ("QSS, 600 chars", 0.013, -0.024, 0.049, LIGHTRED), ("QSS, 600 + notes", 0.071, 0.032, 0.109, RED),
            ("Cosine-20 + notes", 0.019, -0.019, 0.058, LIGHTBLUE), ("VGS", 0.071, 0.032, 0.109, "#7a1f1d"),
            ("VGS-cheap", 0.028, -0.011, 0.069, GRAY), ("Strong: QSS, 200", -0.071, -0.107, -0.034, LIGHTRED),
            ("Strong: QSS, 600", -0.030, -0.062, 0.002, LIGHTRED), ("Strong: VGS-strong", -0.034, -0.069, -0.002, "#7a1f1d")]
    y = np.arange(len(test))[::-1]
    for yi, (lab, m, lo, hi, c) in zip(y, test):
        ax[1].plot([lo, hi], [yi, yi], color=c if c not in (LIGHTRED, LIGHTBLUE) else DARK, linewidth=1.3)
        ax[1].plot([m], [yi], marker="o", color=c, markersize=3.6, **EDGE)
    ax[1].axvline(0, color="black", linewidth=0.6)
    ax[1].axhline(2.5, color=GRAY, linewidth=0.5, linestyle=":")
    ax[1].set_yticks(y, [t[0] for t in test], fontsize=5.6)
    ax[1].set_xlabel("Difference from RAG@40 (95% CI)")
    ax[1].set_title("(b) Registered test (MemLens)")
    finish(fig, "app_dev")


def app_types_gold():
    """Full-set recall on MemLens-32k under alternative gold definitions."""
    fig, ax = row(1, h=1.6, w=W * 0.62)
    x = np.arange(4)
    gold = ["Derived\ngold", "Evidence-turn\nrounds", "Evidence-image\nrounds", "Answer sessions\ncovered"]
    ret = [("QSS, 200 chars (6.1)", [0.375, 0.614, 0.251, 0.687], RED), ("cosine, same size", [0.176, 0.385, 0.106, 0.554], LIGHTRED),
           ("QSS, 600 chars (3.9)", [0.286, 0.828, 0.133, 0.828], "#7a1f1d"), ("cosine, same size ", [0.127, 0.320, 0.052, 0.441], "#e6c3c1"),
           ("Cosine top-40", [0.645, 0.881, 0.576, 0.997], BLUE)]
    for j, (lab, v, c) in enumerate(ret):
        ax[0].bar(x + (j - 2) * 0.155, v, 0.155, color=c, label=lab, **BAR)
    ax[0].set_xticks(x, gold, fontsize=5.4)
    ax[0].set_ylabel("Full-set recall")
    ax[0].set_ylim(0, 1.25)
    ax[0].legend(frameon=False, fontsize=5.2, ncol=3, loc="upper left", handlelength=1.0, columnspacing=0.5)
    finish(fig, "app_gold")


def app_compact_zoom():
    """tab:compact (selection with compact indexes) and tab:zoom (reader-driven zooming)."""
    fig, ax = row(2, h=1.65, w=W * 0.78, wr=[1, 0.85])
    comp = [("MemEye", BLUE, "o", [(11286, 0.903, "original"), (8976, 0.900, "no assistant"), (8115, 0.887, "+ telegraph"),
                                   (6658, 0.881, "+ short ids"), (5797, 0.876, "all three")]),
            ("MemEye+1k", TEAL, "D", [(16791, 0.914, "original"), (10669, 0.922, "compact"), (16173, 0.919, "compact, P=320")]),
            ("MemLens dev", RED, "s", [(7166, 0.382, "original, 200"), (3202, 0.361, "all three, 200"),
                                       (9328, 0.292, "original, 600"), (4026, 0.266, "all three, 600")])]
    for a, sel, yl, title in [(ax[0], comp[:2], (0.865, 0.93), "(a) Compact index, MemEye"),
                              (ax[1], comp[2:], (0.25, 0.4), "(b) Compact index, MemLens")]:
        for lab, c, m, pts in sel:
            a.scatter([p[0] / 1000 for p in pts], [p[1] for p in pts], color=c, marker=m, s=16, label=lab,
                      edgecolor="black", linewidth=0.45, zorder=3)
            for t_, f_, n_ in pts:
                a.annotate(n_, (t_ / 1000, f_), xytext=(2.5, 1.5), textcoords="offset points", fontsize=4.6)
        a.set_ylim(*yl)
        a.set_xlabel("Selector input tokens (k)")
        a.set_title(title)
    ax[0].set_xlim(4.5, 21)
    ax[1].set_xlim(2, 13)
    ax[0].set_ylabel("Full-set recall")
    ax[0].legend(frameon=False, fontsize=5.4, loc="lower right", handlelength=1.0)
    finish(fig, "app_compact")


def app_visualmem():
    """tab:vmscope (scope check) and tab:vmpath (results by gate path)."""
    fig, ax = row(2, h=1.45, w=W * 0.78, wr=[0.8, 1.2])
    for i, (lab, w_, wo) in enumerate([("RAG@40", 0.454, 0.411), ("Annotated\nevidence", 0.609, 0.297)]):
        ax[0].bar(i - 0.19, w_, 0.36, color=BLUE, **BAR)
        ax[0].bar(i + 0.19, wo, 0.36, color="#d9d9d9", **BAR)
        ax[0].text(i, max(w_, wo) + 0.02, f"kept {wo / w_:.0%}".replace("%", "\\%") if False else f"kept {round(100 * wo / w_)}%",
                   ha="center", fontsize=5.4)
    ax[0].axhline(0.25, color="black", linestyle=":", linewidth=0.7)
    ax[0].text(-0.55, 0.258, "chance", fontsize=5, ha="left", va="bottom")
    ax[0].set_xticks([0, 1], ["RAG@40", "Annotated\nevidence"], fontsize=5.6)
    ax[0].set_ylim(0, 0.75)
    ax[0].set_ylabel("Accuracy (light reader)")
    ax[0].set_title("(a) Visual dependence")
    ax[0].legend(handles=[Patch(color=BLUE, label="with images"), Patch(color="#d9d9d9", label="without")],
                 frameon=False, fontsize=5.4, loc="upper left", handlelength=1.0)
    paths = [("All, standard", 0.114, 0.072, 0.155, LIGHTRED), (r"$\hat v<0.7$, widened", 0.176, 0.123, 0.227, RED),
             (r"$\hat v<0.7$, standard", 0.173, 0.123, 0.227, LIGHTRED), (r"$\hat v\geq0.7$, standard", 0.025, -0.043, 0.093, GRAY),
             (r"Strong, $\hat v\geq0.7$, cached", 0.078, 0.036, 0.125, "#7a1f1d")]
    y = np.arange(len(paths))[::-1]
    for yi, (lab, m, lo, hi, c) in zip(y, paths):
        ax[1].barh(yi, m, 0.6, color=c, **BAR)
        ax[1].errorbar([m], [yi], xerr=[[m - lo], [hi - m]], fmt="none", **ERR)
    ax[1].axvline(0, color="black", linewidth=0.6)
    ax[1].set_yticks(y, [p[0] for p in paths], fontsize=5.6)
    ax[1].set_xlabel("Difference from RAG@40 (95% CI)")
    ax[1].set_title("(b) Gains by gate path")
    finish(fig, "app_visualmem")


def app_baselines():
    """Paired differences of every method from RAG@40 (scripts/p11_table.cells): six small panels."""
    from p11_table import cells
    fig, ax = row(6, h=1.75)
    order = [m for m in METHODS if m[0] != "rag"]
    for a, (bench, reader, title) in zip(ax, [(b, r, f"{t}, {rl}") for b, t in [("memeye", "MemEye"), ("memlens", "MemLens"),
                                                                                 ("visualmem", "VisualMem")]
                                               for r, rl in [("lite", "light"), ("strong", "strong")]]):
        C = cells(bench, reader)
        y = np.arange(len(order))[::-1]
        for yi, (key, lab, c, mk) in zip(y, order):
            v = C.get(key)
            if not v:
                continue
            a.plot([v["lo"], v["hi"]], [yi, yi], color=c, linewidth=1.2)
            a.plot([v["d"]], [yi], marker=mk, color=c, markersize=4.4 if key == "vgs" else 3.4, **EDGE)
        a.axvline(0, color="black", linewidth=0.6)
        a.set_yticks(y, [o[1] for o in order] if a is ax[0] else [], fontsize=5.6)
        a.set_title(title, fontsize=6.6)
        a.tick_params(axis="x", labelsize=5.4)
        a.set_xlim(-0.72, 0.2)
        a.set_xticks([-0.6, -0.3, 0])
    fig.supxlabel("Accuracy minus RAG@40 with paired 95% bootstrap interval", fontsize=6.4, y=0.02)
    finish(fig, "app_baselines")


def main():
    for f in (app_keys_reader, app_cluster, app_robust, app_ablate, app_types_gold, app_compact_zoom,
              app_visualmem, app_baselines):
        f()


if __name__ == "__main__":
    main()
