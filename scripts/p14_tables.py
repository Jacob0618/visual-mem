"""Phase 14: the ablation table of the ACL paper (lightweight reader; full-set recall and accuracy per benchmark) from
per-question outputs. MemLens: test split (466 questions; stored runs + phase-14 readings of stored selections, whose
histories have at most 83 rounds, so a 100-round pool is the whole history). MemEye and VisualMem: phase-14 re-runs with
fresh image descriptions (results/p14/runs). No API calls.  python scripts/p14_tables.py"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from analyze import boot_ci  # noqa: E402
from p11_analyze import L, M  # noqa: E402

R = ROOT / "results/p14/runs"
TAU = 0.7
LITE = "gemini-2.5-flash-lite"


def gate(V, std, wid):
    return {q: (std[q] if V[q] >= TAU else wid[q]) for q in std if q in wid and q in V}


def sources():
    test = set(json.load(open(ROOT / "results/memlens_test.json")))
    V1 = json.load(open(ROOT / "results/p5/visibility_hat.json"))
    ml_std = {q: r for q, r in L(M / f"memlens_32k/toc38-40__read-medium__{LITE}.jsonl").items() if q in test}
    ml_wid = L(M / f"memlens_32k_memlens_test/toc38-40__readn-medium__{LITE}__snip600.jsonl")
    S = {"memlens": {
        "rag": {q: r for q, r in L(M / f"memlens_32k/rag-40__read-medium__{LITE}.jsonl").items() if q in test},
        "cosqss": L(R / "memlens/cosqss.jsonl"), "rank8": L(R / "memlens/rank8.jsonl"), "rankqss": L(R / "memlens/rankqss.jsonl"),
        "qss_pool": ml_std, "qss": ml_std, "wid": ml_wid, "vgs": gate(V1, ml_std, ml_wid), "vgs_pool": gate(V1, ml_std, ml_wid)}}
    for bench, suf in [("memeye", ""), ("visualmem", "__xcache")]:
        d = R / bench
        V = json.load(open(ROOT / f"results/p14/vhat_{bench}.json"))
        f = lambda n: L(d / n) if (d / n).exists() else None  # noqa: E731
        std, wid = f(f"toc38-40__read-medium__{LITE}{suf}.jsonl"), f(f"toc38-40__readn-medium__{LITE}__snip600{suf}.jsonl")
        pstd, pwid = f(f"ptoc38-100-40__read-medium__{LITE}.jsonl"), f(f"ptoc38-100-40__readn-medium__{LITE}__snip600.jsonl")
        S[bench] = {"rag": f(f"rag-40__read-medium__{LITE}.jsonl"), "cosqss": f("cosqss.jsonl"), "rank8": f("rank8.jsonl"),
                    "rankqss": f("rankqss.jsonl"), "qss_pool": pstd, "qss": std, "wid": wid,
                    "vgs": gate(V, std, wid) if std and wid else None, "vgs_pool": gate(V, pstd, pwid) if pstd and pwid else None}
        cstd = f(f"toc38-40__read-medium__{LITE}__cnoassist+shortid+telegraph__xcache.jsonl")
        cwid = f(f"toc38-40__readn-medium__{LITE}__snip600__cnoassist+shortid+telegraph__xcache.jsonl")
        if cstd and cwid:
            S[bench]["vgs_compact"] = gate(V, cstd, cwid)
    return S


ROWS = [("rag", "Cosine top-40 (RAG@40)"), ("cosqss", "Cosine top-$k$, $k$ = size of the \\qss{} set"),
        ("rank8", "Model ranking, top 8"), ("rankqss", "Model ranking, top $k$, $k$ = size of the \\qss{} set"),
        ("qss_pool", "\\qss{}, 100-round pool"), ("qss", "\\qss{}, whole index (standard path)"),
        ("wid", "Widened path for every question"), ("vgs", "\\vgs{} (visibility gate)")]


def main():
    S = sources()
    B = ["memeye", "memlens", "visualmem"]
    stats = {}
    for b in B:
        base = S[b]["rag"]
        for k, _ in ROWS + [("vgs_pool", ""), ("vgs_compact", "")]:
            X = S[b].get(k)
            if not X:
                continue
            bad = [q for q, r in X.items() if r.get("error")]
            qs = sorted(q for q in X if q in base and not X[q].get("error"))
            acc = np.array([X[q]["score"] for q in qs])
            fsr = [X[q]["full"] for q in qs if X[q].get("n_gold")]
            d = acc - np.array([base[q]["score"] for q in qs])
            lo, hi = boot_ci(d)
            usd = 1000 * float(np.mean([X[q].get("usd", 0.0) for q in qs]))
            stats[(b, k)] = {"n": len(qs), "errors": len(bad), "acc": float(acc.mean()), "fsr": float(np.mean(fsr)) if fsr else None,
                             "rounds": float(np.mean([X[q]["n_loaded"] for q in qs])), "diff": [float(d.mean()), float(lo), float(hi)],
                             "usd": usd}
            s = stats[(b, k)]
            print(f"{b:9s} {k:11s} n={s['n']:3d} err={s['errors']} FSR {s['fsr'] if s['fsr'] is None else round(s['fsr'], 3)} "
                  f"rounds {s['rounds']:5.1f} acc {s['acc']:.3f} vs RAG {s['diff'][0]:+.3f} [{s['diff'][1]:+.3f},{s['diff'][2]:+.3f}] ${usd:.2f}/1k")
    json.dump({f"{b}|{k}": v for (b, k), v in stats.items()}, open(ROOT / "results/p14/ablation.json", "w"), indent=1)
    best = {b: max(stats[(b, k)]["acc"] for k, _ in ROWS if (b, k) in stats) for b in B}
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{l|cc|cc|cc}", r"\toprule",
             r"\multirow{2}{*}{\textbf{Reading}} & \multicolumn{2}{c|}{\cellcolor{hME}\textbf{MemEye}} & "
             r"\multicolumn{2}{c|}{\cellcolor{hML}\textbf{MemLens}} & \multicolumn{2}{c}{\cellcolor{hVM}\textbf{VisualMem}} \\",
             r" & \cellcolor{hME}FSR & \cellcolor{hME}Acc. & \cellcolor{hML}FSR & \cellcolor{hML}Acc. & \cellcolor{hVM}FSR & \cellcolor{hVM}Acc. \\",
             r"\midrule"]
    for k, name in ROWS:
        if k == "qss_pool":
            lines.append(r"\midrule")
        cells = [r"\oursrow " + name] if k == "vgs" else [name]
        for b in B:
            s = stats.get((b, k))
            if not s:
                cells += ["--", "--"]
                continue
            fsr = "--" if k == "vgs" or s["fsr"] is None else f"{s['fsr']:.3f}".lstrip("0")
            acc = f"{s['acc']:.3f}".lstrip("0")
            cells += [fsr, r"\textbf{" + acc + "}" if abs(s["acc"] - best[b]) < 1e-9 else acc]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Full ablation with the lightweight reader:} full-set recall (FSR) of the rounds handed to "
              r"the reader, and accuracy. The model ranking uses the selector and the \qss{} instruction over the same "
              r"100-round pool as \qss{} and is cut at 8 rounds or at the size that \qss{} chose. MemLens histories have "
              r"at most 83 rounds, so the pool is the whole history there. MemEye and VisualMem come from an independent "
              r"run with regenerated image descriptions.}",
              r"\label{tab:ablation_full}", r"\end{table*}"]
    out = ROOT / "paper_acl/sections/tab_ablation_full.tex"
    open(out, "w").write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()


def gate_tex():
    """Appendix table: routing policies for both readers (results/p14/gate_v2.json, results/p14/gate.json)."""
    G2 = json.load(open(ROOT / "results/p14/gate_v2.json"))
    GS = json.load(open(ROOT / "results/p14/gate.json"))["_strong"]
    cols = [("memeye", G2), ("memlens", G2), ("visualmem", G2), ("memeye", GS), ("memlens", GS)]
    rows = [("Standard path for every question", "always_standard", "always_qss"),
            ("Widened path for every question", "always_widened", "always_rag"),
            ("Gate on message length", "length_gate", "length_gate"),
            ("Random routing, same share", "random_same_share", "random_same_share"),
            ("\\vgs{} gate ($\\hat v(q)\\ge0.7$)", "gate", "gate"),
            ("Per-question oracle", "oracle", "oracle")]
    CAP = [r"\caption{\textbf{Routing policies.} Each cell is the accuracy when every question takes the given path or when a rule "
          r"routes it; the widened path of the strong reader is top-40 retrieval. The message-length gate widens a "
          r"question when the user messages of its ten most similar rounds average at least " +
          f"{G2['_thresholds']['length_chars']:.0f}" + r" characters, a threshold chosen on MemEye and the remaining MemLens "
          r"questions as for $\tau$. AUC: how well $-\hat v(q)$ ranks the questions on which the widened path scores higher "
          r"than the standard path above those on which it scores lower. MemEye and VisualMem with the "
          r"lightweight reader use the run of Table~\ref{tab:ablation}.}", r"\label{tab:gate}"]
    L_ = [r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{l|ccc|cc}", r"\toprule",
          r"\rowcolor{hHead} & \multicolumn{3}{c|}{\textbf{Lightweight reader}} & \multicolumn{2}{c}{\textbf{Strong reader}} \\",
          r"\rowcolor{hHead} \textbf{Policy} & MemEye & MemLens & VisualMem & MemEye & MemLens \\", r"\midrule"]
    for name, kl, ks in rows:
        vals = [G[b][kl] if G is G2 else G[b][ks] for b, G in cols]
        if name.startswith("Per-question"):
            L_.append(r"\midrule")
        L_.append((r"\oursrow " if "gate ($" in name else "") + " & ".join([name] + [f"{v:.3f}" for v in vals]) + r" \\")
    L_.append(" & ".join(["AUC of $\\hat v(q)$"] + [f"{G2[b]['auc_vhat']:.2f}" for b in ("memeye", "memlens", "visualmem")] + ["--", "--"]) + r" \\")
    L_ += [r"\bottomrule", r"\end{tabular}"] + CAP + [r"\end{table*}"]
    open(ROOT / "paper_acl/sections/tab_gate.tex", "w").write("\n".join(L_) + "\n")
    print("wrote tab_gate.tex")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "gate":
    gate_tex()


def ablation_wo_tex():
    """Main-text ablation in w/o form (reference style): accuracy of the lightweight reader, each w/o row shaded by its
    drop from the highlighted row above it."""
    S = sources()
    B = ["memeye", "memlens", "visualmem"]

    def acc(b, k):
        X, base = S[b].get(k), S[b]["rag"]
        qs = [q for q in X if q in base and not X[q].get("error")]
        return float(np.mean([X[q]["score"] for q in qs]))

    blocks = [("\\vgs{} (full)", "vgs", [("w/o widened path", "qss"), ("w/o whole index", "vgs_pool"), ("w/o standard path", "wid")]),
              ("Set selection (\\qss{})", "qss_pool", [("w/o index reading", "cosqss"), ("w/o adaptive size", "rank8"),
                                                      ("w/o set output", "rankqss")])]
    shade = lambda d: "" if d < 1 else "dA" if d < 3 else "dB" if d < 5 else "dC" if d < 8 else "dD"  # noqa: E731
    L_ = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.2pt}", r"\renewcommand{\arraystretch}{1.06}",
          r"\begin{tabular}{lccc}", r"\toprule",
          r"\textbf{Configuration} & \cellcolor{hME}\textbf{MemEye} & \cellcolor{hML}\textbf{MemLens} & \cellcolor{hVM}\textbf{VisualMem} \\",
          r"\midrule"]
    for bi, (ref_name, ref_key, rows) in enumerate(blocks):
        if bi:
            L_.append(r"\midrule")
        ref = {b: acc(b, ref_key) for b in B}
        L_.append(r"\oursrow " + ref_name + " & " + " & ".join(f"{ref[b]:.3f}"[1:] for b in B) + r" \\")
        vals = []
        for name, k in rows:
            v = {b: acc(b, k) for b in B}
            vals.append((float(np.mean([ref[b] - v[b] for b in B])), name, v))
        for _, name, v in sorted(vals, key=lambda t: -t[0]):
            cells = []
            for b in B:
                d = 100 * (round(ref[b], 3) - round(v[b], 3))       # change between the displayed values
                sgn = "0.0" if abs(d) < 0.05 else (f"-{d:.1f}" if d > 0 else f"+{-d:.1f}")
                t = f"{v[b]:.3f}"[1:] + r"\,{\fontsize{6.4}{7}\selectfont$" + sgn + "$}"
                cells.append((rf"\cellcolor{{{shade(d)}}}" if shade(d) else "") + t)
            L_.append(r"\quad " + name + " & " + " & ".join(cells) + r" \\")
    L_ += [r"\bottomrule", r"\end{tabular}",
           r"\caption{\textbf{Ablations with the lightweight reader.} Accuracy and change in points from the highlighted "
           r"row above; shading marks drops of 1--3, 3--5, 5--8, and over 8 points. \emph{w/o widened} or \emph{standard "
           r"path}: one path for every question; \emph{w/o whole index}: both paths read a 100-round cosine pool; "
           r"\emph{w/o index reading}: cosine top-$k$ at the size that \qss{} chose; \emph{w/o adaptive size}: the "
           r"selector's ranking of the index cut at 8 rounds; \emph{w/o set output}: the same ranking cut at the size "
           r"that \qss{} chose. The second block uses the 100-round pool (on MemLens, whole histories of at most 83 "
           r"rounds). MemEye and VisualMem come from a run with regenerated image descriptions; full-set recall is in "
           r"Table~\ref{tab:ablation_full}.}",
           r"\label{tab:ablation}", r"\end{table}"]
    open(ROOT / "paper_acl/sections/tab_ablation.tex", "w").write("\n".join(L_) + "\n")
    print("wrote tab_ablation.tex (w/o form)")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "wo":
    ablation_wo_tex()
