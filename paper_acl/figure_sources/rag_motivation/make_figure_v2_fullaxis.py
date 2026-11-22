"""Figure 1: RAG accuracy and its gap to full-context reading with gemini-3.8-flash. No model calls.
Data: rag_accuracy.csv (audited; see README.txt). Writes rag_motivation.pdf / .png / .svg next to this script;
copy the PDF to paper_acl/figures_polished/. Run with the project .venv (matplotlib, numpy)."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

OUT = Path(__file__).resolve().parent
rows = list(csv.DictReader((OUT / "rag_accuracy.csv").open()))
rag = np.array([100 * float(r["rag_accuracy"]) for r in rows])
full = np.array([100 * float(r["full_context_accuracy"]) for r in rows])
is_text = np.array([r["modality"] == "Text" for r in rows])
names = ["LoCoMo", "LongMemEval-S", "MemEye", "MemLens", "VisualMem"]

BLUE, RED, DARK = "#4f81bd", "#c0504d", "#333333"
BAND = {True: "#e9eff7", False: "#f8eceb"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7, "axes.linewidth": 0.6, "xtick.major.size": 0, "ytick.major.size": 2.0,
    "ytick.major.width": 0.6, "pdf.fonttype": 42})

fig, ax = plt.subplots(figsize=(3.03, 1.72))
fig.subplots_adjust(left=0.125, right=0.99, bottom=0.235, top=0.86)
x = np.array([0.0, 1.2, 2.55, 3.55, 4.55])         # room for the long label and a gap between the groups
w = 0.6
ax.axvspan(-0.5, 1.7, color=BAND[True], lw=0, zorder=0)
ax.axvspan(2.05, 5.05, color=BAND[False], lw=0, zorder=0)
for xi, r, f, t in zip(x, rag, full, is_text):
    c = BLUE if t else RED
    ax.bar(xi, r, w, color=c, edgecolor="black", linewidth=0.45, zorder=2)
    ax.bar(xi, f - r, w, bottom=r, facecolor="white", edgecolor=c, hatch="//////", linewidth=0.5, zorder=2)
    ax.plot([xi - w / 2, xi + w / 2], [f, f], color=DARK, linewidth=0.9, solid_capstyle="butt", zorder=3)
    ax.text(xi, r - 3.0, f"{r:.1f}", ha="center", va="top", color="white", fontsize=6.4, fontweight="bold", zorder=4)
    ax.text(xi, f + 1.6, f"+{f - r:.1f}", ha="center", va="bottom", fontsize=6.6, zorder=4,
            color=DARK if t else "#a3302b", fontweight="normal" if t else "bold")
ax.set_xlim(-0.5, 5.05)
ax.set_ylim(0, 107)
ax.set_yticks([0, 25, 50, 75, 100])
ax.tick_params(axis="y", labelsize=6.3, pad=1.5)
ax.set_ylabel("Accuracy (%)", fontsize=6.8, labelpad=2)
ax.set_xticks(x, names, fontsize=6.4)
ax.tick_params(axis="x", pad=2.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
tr = blended_transform_factory(ax.transData, ax.transAxes)
for xc, label, c in [(0.6, "Text memory", BLUE), (3.55, "Multimodal memory", RED)]:
    ax.text(xc, 1.035, label, transform=tr, ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=c)
solid = tuple(Patch(facecolor=c, edgecolor="black", linewidth=0.45) for c in (BLUE, RED))
hatched = tuple(Patch(facecolor="white", edgecolor=c, hatch="//////", linewidth=0.5) for c in (BLUE, RED))
fig.legend([solid, hatched], ["RAG", "gap to full context (points)"], handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
           loc="lower center", bbox_to_anchor=(0.56, 0.0), ncol=2, frameon=False, fontsize=6.3, handlelength=2.0,
           handleheight=0.8, columnspacing=1.6, handletextpad=0.5, borderaxespad=0.2)
for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT / f"rag_motivation.{ext}", dpi=300, facecolor="white")
plt.close(fig)
