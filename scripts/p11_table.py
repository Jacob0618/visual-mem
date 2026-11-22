"""Phase 11: main comparison table (LaTeX rows) with external memory systems, on identical question sets.
MemEye: all 371; MemLens-32k: the 150-question stratified subset of the test part; VisualMem: all 696.
Costs per 1,000 questions include memory construction (amortised over the benchmark's questions) and, for VGS, the
creation and storage of caches. No API calls.
python scripts/p11_table.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import boot_ci  # noqa: E402
from p11_analyze import bl, ours  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUB = set(json.load(open(ROOT / "results/p11_memlens_sub150.json")))
CACHE_OVERHEAD = {("memeye", "strong"): 0.38, ("visualmem", "lite"): 3.69, ("visualmem", "strong"): 3.60}
NAMES = [("RAG@40", "rag"), ("Full context", "fc"), ("SimpleMem", "simplemem"), ("Omni-SimpleMem", "omni"), ("MIRIX", "mirix"),
         ("V-Mem", "vmem"), ("MMA", "mma"), ("TeleMem", "telemem"), ("\\vgs{} (ours)", "vgs")]


def fc(bench, reader):
    """Full-context reading of the whole history (phase-2 runs on MemEye, phase-5/14 runs on MemLens and VisualMem)."""
    from p11_analyze import L, M
    P14 = ROOT / "results/p14/runs"
    if bench == "memeye":
        from analyze_p2 import load_all
        return load_all("gemini-2.5-flash-lite" if reader == "lite" else "gemini-3.8-flash")["fc|read:medium" if reader == "lite" else "fc|read:low"]
    f = {("memlens", "lite"): M / "memlens_32k/fc__read-medium__gemini-2.5-flash-lite.jsonl",
         ("memlens", "strong"): P14 / "memlens_fc/fc__read-low__gemini-3.8-flash.jsonl",
         ("visualmem", "lite"): P14 / "visualmem/fc__read-medium__gemini-2.5-flash-lite.jsonl",
         ("visualmem", "strong"): P14 / "visualmem/fc__read-low__gemini-3.8-flash.jsonl"}[(bench, reader)]
    return L(f) if f.exists() else {}


def cells(bench, reader):
    R40, VGS = ours(bench)[reader]
    X = {"rag": R40, "vgs": VGS, "fc": fc(bench, reader)}
    for _, s in NAMES[2:-1]:
        X[s] = bl(bench, s, reader)
    qs = set(R40) & set(VGS)
    if bench == "memlens":
        qs &= SUB
    out = {}
    for key, D in X.items():
        q2 = sorted(q for q in qs if q in D and D[q].get("score") is not None)
        if not q2:
            out[key] = None
            continue
        s = np.array([D[q]["score"] for q in q2])
        d = s - np.array([R40[q]["score"] for q in q2])
        lo, hi = boot_ci(d)
        usd = 1000 * np.mean([D[q].get("usd", 0.0) for q in q2]) + (CACHE_OVERHEAD.get((bench, reader), 0.0) if key == "vgs" else 0.0)
        out[key] = {"n": len(q2), "acc": s.mean(), "d": d.mean(), "lo": lo, "hi": hi, "usd": usd, "full": len(q2) == len(qs)}
    return out


def main():
    rows = {}
    for reader in ["lite", "strong"]:
        for bench in ["memeye", "memlens", "visualmem"]:
            rows[(reader, bench)] = cells(bench, reader)
    for reader, title in [("lite", "Lightweight reader (\\texttt{gemini-2.5-flash-lite})"), ("strong", "Strong reader (\\texttt{gemini-3.8-flash})")]:
        print(f"\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for name, key in NAMES:
            line = [name]
            for bench in ["memeye", "memlens", "visualmem"]:
                c = rows[(reader, bench)][key]
                if c is None:
                    line += ["--", "--"]
                    continue
                best = max(v["acc"] for v in rows[(reader, bench)].values() if v)
                acc = f"{c['acc']:.3f}"
                acc = f"\\textbf{{{acc}}}" if abs(c["acc"] - best) < 1e-9 else acc
                if key != "rag":
                    sig = "$^{*}$" if (c["lo"] > 0 or c["hi"] < 0) else ""
                    acc = f"{acc}{sig}"
                if not c["full"]:
                    acc += f" ({c['n']})"
                line += [acc, f"{c['usd']:.2f}"]
            print(" & ".join(line) + " \\\\")
        print("\\midrule")
    print("\n% paired differences vs RAG@40 [95% CI]:")
    for (reader, bench), C in rows.items():
        for name, key in NAMES[1:]:
            c = C[key]
            if c:
                print(f"% {reader:6s} {bench:9s} {key:9s} n={c['n']:3d} acc {c['acc']:.3f} diff {c['d']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] ${c['usd']:.2f}")


if __name__ == "__main__":
    main()


def write_tex(path=ROOT / "paper_acl/sections/tab_main.tex"):
    """Full-width main table (reference style): accuracy of both readers and cost per benchmark, coloured benchmark
    headers, method groups separated by rules, our row highlighted, caption below."""
    rows = {(r, b): cells(b, r) for r in ["lite", "strong"] for b in ["memeye", "memlens", "visualmem"]}

    def acc(reader, bench, key):
        c = rows[(reader, bench)][key]
        if c is None:
            return "--"
        best = max(v["acc"] for v in rows[(reader, bench)].values() if v and v["full"])
        t = f"{c['acc']:.3f}"[1:]
        t = rf"\textbf{{{t}}}" if abs(c["acc"] - best) < 1e-9 else t
        if not c["full"]:
            t += r"$^{\dagger}$"
        return t

    def usd(bench, key):
        f = lambda c: "--" if c is None else (f"{c['usd']:.0f}" if c["usd"] >= 100 else f"{c['usd']:.1f}")  # noqa: E731
        return f"{f(rows[('lite', bench)][key])}\\,/\\,{f(rows[('strong', bench)][key])}"

    H = {"memeye": ("hME", "MemEye"), "memlens": ("hML", "MemLens-32k (150)"), "visualmem": ("hVM", "VisualMem")}
    L = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.6pt}",
         r"\renewcommand{\arraystretch}{1.06}", r"\begin{tabular}{l|ccc|ccc|ccc}", r"\toprule",
         r"\multirow{2}{*}{\textbf{Method}} & " + " & ".join(
             rf"\multicolumn{{3}}{{c{'|' if b != 'visualmem' else ''}}}{{\cellcolor{{{H[b][0]}}}\textbf{{{H[b][1]}}}}}" for b in H) + r" \\",
         " & " + " & ".join(rf"\cellcolor{{{H[b][0]}}}Light & \cellcolor{{{H[b][0]}}}Strong & \cellcolor{{{H[b][0]}}}\$/1k" for b in H) + r" \\",
         r"\midrule"]
    groups = [["rag", "fc"], ["simplemem", "omni", "mirix", "telemem"], ["vmem", "mma"], ["vgs"]]
    name = dict((k, n) for n, k in NAMES)
    for gi, g in enumerate(groups):
        if gi:
            L.append(r"\midrule")
        for key in g:
            cells_ = []
            for bench in ["memeye", "memlens", "visualmem"]:
                cells_ += [acc("lite", bench, key), acc("strong", bench, key), usd(bench, key)]
            if key == "vgs":
                L.append(r"\oursrow \textbf{\vgs{} (ours)} & " + " & ".join(cells_) + r" \\")
            else:
                L.append(name[key] + " & " + " & ".join(cells_) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Main results on identical question sets.} Accuracy with the lightweight (Light) and the "
          r"strong reader, and cost per 1,000 questions in USD (lightweight\,/\,strong), including memory construction "
          r"and, for \vgs{}, the creation and storage of caches. Methods are grouped into raw retrieval and "
          r"full-context reading, memory systems that write text at ingestion, memory systems that keep raw turns, "
          r"and ours; all share the same readers, prompts, and judges. MemLens uses a stratified subset of 150 test "
          r"questions (full test split in Table~\ref{tab:fulltest}); paired intervals are in "
          r"Figure~\ref{fig:app_baselines}. Best accuracy per column in bold. $^{\dagger}$On the 269 MemEye and 140 "
          r"MemLens questions whose histories MIRIX finished building.}",
          r"\label{tab:main}", r"\end{table*}"]
    open(path, "w").write("\n".join(L) + "\n")
    return path


def write_ci_tex(path=ROOT / "paper_acl/sections/tab_baselines_ci.tex"):
    """Appendix table: paired differences from RAG@40 with 95% bootstrap intervals."""
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Paired differences from RAG@40 with 95\% bootstrap intervals on the question sets of Table~\ref{tab:main}.}",
         r"\label{tab:baselines_ci}", r"\small", r"\begin{tabular}{llccc}", r"\toprule",
         r"Reader & Method & MemEye & MemLens-32k (150) & VisualMem \\", r"\midrule"]
    for reader, rname in [("lite", "Lightweight"), ("strong", "Strong")]:
        C = {b: cells(b, reader) for b in ["memeye", "memlens", "visualmem"]}
        for k, (name, key) in enumerate(NAMES[1:]):
            line = [rname if k == 0 else "", name]
            for b in ["memeye", "memlens", "visualmem"]:
                c = C[b][key]
                line.append("--" if c is None else f"${c['d']:+.3f}$ $[{c['lo']:+.3f}, {c['hi']:+.3f}]$" + ("" if c["full"] else r"$^{\dagger}$"))
            L.append(" & ".join(line) + r" \\")
        L.append(r"\midrule" if reader == "lite" else r"\bottomrule")
    L += [r"\end{tabular}", r"\end{table*}"]
    open(path, "w").write("\n".join(L) + "\n")
    return path
