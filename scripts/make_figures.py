"""Paper figures (no API calls) -> paper/figures/*.pdf"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_p2 import load_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper/figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
BUCKETS = [(1, 1, "1"), (2, 3, "2–3"), (4, 5, "4–5"), (6, 99, "≥6")]


def per_q_full(name):
    R = json.load(open(ROOT / f"results/mm2/select_eval_{name}.json"))["per_q"]
    return {r["qid"]: (r["n_gold"], r["full"], r["n"]) for r in R if r["full"] is not None}


def fig_setsize():
    rows = json.load(open(ROOT / "results/mm2/set_analysis_rows.json"))
    scu = per_q_full("scu-5-2-40")
    rk = {r["qid"]: r for r in json.load(open(ROOT / "results/mm2/rank_control_P100.json"))["per_q"]}
    series = [("Cosine top-10", [r["cos10"] for r in rows], "#bbbbbb", "o", "--"),
              ("Cosine top-40", [r["cos40"] for r in rows], "#777777", "s", "--"),
              ("Lightweight selector, consensus", [scu[r["qid"]][1] for r in rows], "#4c9be8", "^", "-"),
              ("Strong model, ranking (same size)", [rk[r["qid"]]["rank_full_kq"] for r in rows], "#e8a0a8", "v", ":"),
              ("QSS, strong selector", [rk[r["qid"]]["qss_full"] for r in rows], "#d1495b", "D", "-")]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    x = np.arange(len(BUCKETS))
    for lab, vals, c, m, ls in series:
        ys = []
        for lo, hi, _ in BUCKETS:
            v = [vals[i] for i, r in enumerate(rows) if lo <= r["g"] <= hi]
            ys.append(np.mean(v))
        ax.plot(x, ys, marker=m, color=c, label=lab, linestyle=ls, linewidth=1.3, markersize=4)
    ns = [sum(1 for r in rows if lo <= r["g"] <= hi) for lo, hi, _ in BUCKETS]
    ax.set_xticks(x, [f"{b[2]}\n(n={n})" for b, n in zip(BUCKETS, ns)])
    ax.set_xlabel("Number of gold evidence rounds")
    ax.set_ylabel("Full-set recall")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "setsize.pdf"); fig.savefig(FIG / "setsize.png", dpi=200)


PR = {"gemini-3.8-flash": (0.75, 0.075, 3.75), "gemini-2.5-flash-lite": (0.10, 0.01, 0.40)}
SEL_IN = {r["qid"]: r["in_tok"] for r in json.load(open(ROOT / "results/mm2/select_eval_toc38-40.json"))["per_q"]}
SEL_OUT, Q_TOK = 230, 150


def warm_cost(name, R, reader):
    """Per-question cost with the static per-user index served from the prefix cache (results/mm2/cost_warmcache_v2.md):
    the selection call bills the index at the cached rate; the index-as-context reading call does the same for its
    index part; everything else is billed at full price."""
    if not name.startswith("toc38"):
        return np.mean([r.get("usd", 0) for r in R])
    si, sc, so = PR["gemini-3.8-flash"]
    ri, rc, ro = PR[reader]
    out = []
    for r in R:
        sel_in = SEL_IN[r["qid"]]
        toc = sel_in - Q_TOK
        read_in, read_out = r["in_tok"] - sel_in, max(0, r["out_tok"] - SEL_OUT)
        sel = (toc * sc + Q_TOK * si + SEL_OUT * so) / 1e6
        rd = ((toc * rc + (read_in - toc) * ri) if "+toc" in name else read_in * ri) / 1e6 + read_out * ro / 1e6
        out.append(sel + rd)
    return np.mean(out)


def fig_cost():
    pts = {"gemini-2.5-flash-lite": [("fc|read:medium", "FC"), ("rag:10|read:medium", "RAG@10"),
                                     ("rag:20|read:medium", "RAG@20"), ("rag:40|read:medium", "RAG@40"),
                                     ("scu:5:2:40|read:medium", "lightweight consensus"),
                                     ("toc38:40|read:medium", "QSS")],
           "gemini-3.8-flash": [("fc|read:low", "FC"), ("rag:10|read:low", "RAG@10"), ("rag:40|read:low", "RAG@40"),
                                ("toc38:40|read:low", "QSS"), ("toc38:40|read+toc:low", "QSS, index as context")]}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    for ax, (reader, names) in zip(axes, pts.items()):
        D = load_all(reader)
        for name, lab in names:
            if name not in D:
                continue
            R = list(D[name].values())
            acc = np.mean([r["score"] for r in R])
            cost = warm_cost(name, R, reader)
            c = "#d1495b" if lab.startswith("QSS") else ("#4c9be8" if lab.startswith("lightweight") else "#555555")
            ax.scatter(1000 * cost, acc, color=c, s=18, zorder=3)
            ax.annotate(lab, (1000 * cost, acc), textcoords="offset points", xytext=(3, 3), fontsize=6.5)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if v in (0.3, 0.5, 2, 3, 5, 20) else ""))
        ax.set_xlabel("USD per 1k questions (log scale)")
        ax.set_ylabel("Accuracy (MemEye-371)")
        ax.set_title(reader.replace("gemini-", "") + " reader", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "cost.pdf"); fig.savefig(FIG / "cost.png", dpi=200)


def fig_scale():
    vals = {"2.5-flash-lite reader": {"FC": (0.495, 0.368), "RAG@40": (0.488, 0.471), "QSS": (0.541, 0.538)},
            "3.8-flash reader": {"RAG@40": (0.741, 0.730), "QSS": (0.761, 0.764)}}
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0), sharey=False)
    for ax, (title, d) in zip(axes, vals.items()):
        for lab, (a0, a1) in d.items():
            c = "#d1495b" if lab == "QSS" else ("#777777" if lab == "RAG@40" else "#222222")
            ax.plot([0, 1], [a0, a1], marker="o", color=c, label=lab, linewidth=1.3, markersize=4)
        ax.set_xticks([0, 1], ["original", "+1,000 imgs"])
        ax.set_title(title, fontsize=7.5)
        ax.legend(frameon=False, fontsize=6)
    axes[0].set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(FIG / "scale.pdf"); fig.savefig(FIG / "scale.png", dpi=200)


if __name__ == "__main__":
    fig_setsize()
    fig_cost()
    fig_scale()
    print("figures written to", FIG)
