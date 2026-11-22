"""Phase 13: Table 3 of the ACL paper (standard metrics and latency) from results/p13/standard_metrics.json and
results/p13/latency_summary.json. No API calls.  python scripts/p13_table.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = [("RAG@40", "rag"), ("SimpleMem", "simplemem"), ("Omni-SimpleMem", "omni"), ("MIRIX", "mirix"),
         ("V-Mem", "vmem"), ("MMA", "mma"), ("TeleMem", "telemem"), (r"\vgs{} (ours)", "vgs")]
B3 = ["memeye", "memlens", "visualmem"]


def f3(x):
    return "--" if x is None else f"{x:.3f}".replace("0.", ".", 1) if x < 1 else f"{x:.2f}"


def main():
    M = json.load(open(ROOT / "results/p13/standard_metrics.json"))
    Lp = ROOT / "results/p13/latency_summary.json"
    LAT = json.load(open(Lp)) if Lp.exists() else {}
    g = lambda b, r, s, k: M.get(f"{b}|{r}|{s}", {}).get(k)
    best = {}
    for b in B3:
        for r in ("lite", "strong"):
            for k in ("f1", "bleu1", "r10", "ndcg10"):
                vals = [g(b, r, s, k) for _, s in NAMES if g(b, r, s, k) is not None]
                best[(b, r, k)] = max(vals) if vals else None

    def cell(b, r, s, k):
        v = g(b, r, s, k)
        t = f3(v)
        return rf"\textbf{{{t}}}" if v is not None and best[(b, r, k)] is not None and abs(v - best[(b, r, k)]) < 1e-9 else t

    def lat(b, s):
        v = LAT.get(f"{b}|lite|{s}")
        return "--" if v is None else (f"{v['median']:.1f}" if v["median"] < 100 else f"{v['median']:.0f}")

    CAP = [r"\caption{\textbf{Standard metrics.} On the question sets of Table~\ref{tab:main}. Left: token F1 and BLEU-1 of the answers "
         r"(lightweight / strong reader) on the open-ended benchmarks; VisualMem is multiple choice, so its answer metric is "
         r"the exact match of Table~\ref{tab:main}. Right: Recall@10 and nDCG@10 of the rounds that a method hands to the "
         r"reader, in its own order, against the gold rounds (\vgs{}: its lightweight-reader path). They apply to the "
         r"methods that return raw rounds; the other memory systems return rewritten records. Latency is in "
         r"Table~\ref{tab:latency}.}", r"\label{tab:std}"]
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.4pt}",
         r"\begin{tabular}[t]{l|cc|cc}", r"\toprule",
         r"\rowcolor{hHead} & \multicolumn{2}{c|}{\textbf{Token F1}} & \multicolumn{2}{c}{\textbf{BLEU-1}} \\",
         r"\rowcolor{hHead} \textbf{Method} & MemEye & MemLens & MemEye & MemLens \\", r"\midrule"]
    for name, s in NAMES:
        if s == "vgs":
            L.append(r"\midrule")
        row = [name]
        for k in ("f1", "bleu1"):
            for b in ("memeye", "memlens"):
                row.append(f"{cell(b, 'lite', s, k)} / {cell(b, 'strong', s, k)}")
        L.append((r"\oursrow " if s == "vgs" else "") + " & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}\hfill", r"\begin{tabular}[t]{l|ccc|ccc}", r"\toprule",
          r"\rowcolor{hHead} & \multicolumn{3}{c|}{\textbf{Recall@10}} & \multicolumn{3}{c}{\textbf{nDCG@10}} \\",
          r"\rowcolor{hHead} \textbf{Method} & ME & ML & VM & ME & ML & VM \\", r"\midrule"]
    for name, s in NAMES:
        if g("memeye", "lite", s, "r10") is None:
            continue
        if s == "vgs":
            L.append(r"\midrule")
        L.append((r"\oursrow " if s == "vgs" else "") + " & ".join([name] + [cell(b, "lite", s, k) for k in ("r10", "ndcg10") for b in B3]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"] + CAP + [r"\end{table*}"]
    out = ROOT / "paper_acl/sections/tab_std.tex"
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()


def write_latency_tex():
    """Appendix table: median latency per question (lightweight / strong reader), median retrieval time, and memory
    construction time per history (results/p13/latency_summary.json, build times in the latency records)."""
    import numpy as np
    S = json.load(open(ROOT / "results/p13/latency_summary.json"))
    bt = json.load(open(ROOT / "results/p13/build_time_ours.json")) if (ROOT / "results/p13/build_time_ours.json").exists() else {}
    L = ROOT / "results/p13/latency"

    def build(b, s):
        if s in ("rag", "vgs"):
            v = [h["caption_s"] + h["embed_s"] for h in bt.get(b, [])]
            return float(np.median(v)) if v else None
        if s == "mma":                     # its latency runs reused cached image embeddings: fresh re-measurement
            m = ROOT / "results/p13/build_time_mma.json"
            v = [h["build_s"] for h in json.load(open(m)).get(b, [])] if m.exists() else []
            return float(np.median(v)) if v else None
        # SimpleMem was re-run with its keyword index on replayed construction calls: the first (fresh) build counts
        p = L / b / (f"{s}_no_fts.jsonl" if s == "simplemem" else f"{s}.jsonl")
        if not p.exists():
            return None
        per_ep = {}
        for r in map(json.loads, open(p)):
            t = r.get("build_seconds", r.get("t_build_s"))
            if t is not None:
                per_ep[r.get("task") or r["qid"] if b == "memlens" else "h"] = t
        return float(np.median(list(per_ep.values()))) if per_ep else None

    def fmt(x):
        return "--" if x is None else (f"{x:.1f}" if x < 100 else f"{x:,.0f}")

    cap = [r"\caption{\textbf{Latency and construction time.} On the latency samples (Appendix~\ref{app:latency}): median seconds per question with the "
            r"lightweight / strong reader, the median retrieval time of each method (including its model calls; for "
            r"\vgs{}, the selection call), and the memory construction time per history (seconds; MemEye: one history "
            r"of 120 rounds, MemLens: median over the 10 histories, VisualMem: one persona of 1,013 rounds)."
            + ("" if all(S.get(f"{b}|lite|{s}") for b in B3 for _, s in NAMES) else
               r" ``--'': not measured (MIRIX did not finish building this memory).") + "}",
            r"\label{tab:latency}"]
    rows = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.2pt}", r"\begin{tabular}{l|cc|cc|cc|ccc}", r"\toprule",
            r"\rowcolor{hHead} & \multicolumn{2}{c|}{\cellcolor{hME}\textbf{MemEye}} & \multicolumn{2}{c|}{\cellcolor{hML}\textbf{MemLens}} & "
            r"\multicolumn{2}{c|}{\cellcolor{hVM}\textbf{VisualMem}} & \multicolumn{3}{c}{\textbf{Construction (s per history)}} \\",
            r"\rowcolor{hHead} \textbf{Method} & Answer & Retrieval & Answer & Retrieval & Answer & Retrieval & MemEye & MemLens & VisualMem \\",
            r"\midrule"]
    for name, s in NAMES:
        if s == "vgs":
            rows.append(r"\midrule")
        line = [name]
        for b in B3:
            lt, st = S.get(f"{b}|lite|{s}"), S.get(f"{b}|strong|{s}")
            line.append("--" if not lt else f"{fmt(lt['median'])} / {fmt(st['median']) if st else '--'}")
            line.append("--" if not lt else fmt(lt["retrieve_median"]))
        for b in B3:
            line.append(fmt(build(b, s)))
        rows.append((r"\oursrow " if s == "vgs" else "") + " & ".join(line) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"] + cap + [r"\end{table*}"]
    out = ROOT / "paper_acl/sections/tab_latency.tex"
    open(out, "w").write("\n".join(rows) + "\n")
    print("wrote", out)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "latency":
    write_latency_tex()
