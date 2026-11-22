"""Held-out MemLens analysis (no API calls) -> results/TABLES_MEMLENS.md
Accuracy uses the official MemLens judge prompt with Qwen3-VL-235B ('score'); 'score2' is gemini-3.8-flash on the
same prompt where available. Paired bootstrap over questions (every question has its own haystack).
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import boot_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = []


def p(s=""):
    OUT.append(s)
    print(s)


def load(d, reader):
    D = {}
    for f in glob.glob(str(ROOT / d / f"*__{reader}*.jsonl")):
        stem = os.path.basename(f)[:-6]
        parts = stem.split("__")
        name = parts[0].replace("-", ":") + "|" + parts[1].replace("-", ":", 1) + "".join(
            f" +{x}" for x in parts[3:])
        D[name] = {json.loads(l)["qid"]: json.loads(l) for l in open(f)}
    return D


def ci_diff(A, B, field="score"):
    q = sorted(set(A) & set(B))
    d = np.array([A[x][field] - B[x][field] for x in q])
    lo, hi = boot_ci(d)
    return d.mean(), lo, hi, len(q)


def table(D, names, refs, title, field="score"):
    p(f"\n### {title}\n")
    p("| condition | n | accuracy [95% CI] | judge-2 acc | rounds | full-set recall | $/1k q | " +
      " | ".join(f"Δ vs {r}" for r in refs) + " |")
    p("|---|---:|---|---:|---:|---:|---:|" + "---|" * len(refs))
    for n in names:
        if n not in D:
            continue
        R = list(D[n].values())
        s = np.array([r[field] for r in R]); lo, hi = boot_ci(s)
        s2 = [r["score2"] for r in R if r.get("score2") is not None]
        fs = [r["full"] for r in R if r.get("full") is not None]
        cells = []
        for rf in refs:
            if rf in D and rf != n:
                m, a, b, _ = ci_diff(D[n], D[rf], field)
                cells.append(f"{m:+.3f} [{a:+.3f}, {b:+.3f}]")
            else:
                cells.append("—")
        p(f"| {n} | {len(R)} | {s.mean():.3f} [{lo:.3f}, {hi:.3f}] | {np.mean(s2):.3f} | "
          f"{np.mean([r.get('n_loaded') or 0 for r in R]):.1f} | {np.mean(fs):.3f} | "
          f"{1000 * np.mean([r.get('usd', 0) for r in R]):.2f} | " + " | ".join(cells) + " |"
          if s2 else
          f"| {n} | {len(R)} | {s.mean():.3f} [{lo:.3f}, {hi:.3f}] | – | "
          f"{np.mean([r.get('n_loaded') or 0 for r in R]):.1f} | {np.mean(fs) if fs else float('nan'):.3f} | "
          f"{1000 * np.mean([r.get('usd', 0) for r in R]):.2f} | " + " | ".join(cells) + " |")


def by_type(D, names, title):
    p(f"\n### {title}\n")
    types = ["information_extraction", "temporal_reasoning", "multi_session_reasoning", "knowledge_update"]
    p("| condition | " + " | ".join(types) + " |")
    p("|---|" + "---:|" * len(types))
    for n in names:
        if n not in D:
            continue
        R = list(D[n].values())
        p(f"| {n} | " + " | ".join(f"{np.mean([r['score'] for r in R if r['x'] == t]):.3f}" for t in types) + " |")


def selection():
    p("\n### Selection only, MemLens-32k (699 questions)\n")
    p("| selector | rounds | FSR [cosine same size] | FSR, >=4 gold rounds [cosine] | coverage | session-FSR | $/1k q |")
    p("|---|---:|---|---|---:|---:|---:|")
    for f in sorted(glob.glob(str(ROOT / "results/mm2/select_eval_memlens32k_*.json"))):
        J = json.load(open(f)); s = J["summary"]
        if s["sel"].startswith("rank38"):
            continue
        R = J["per_q"]
        p(f"| {s['sel']}{' (snippet 600)' if 'snip600' in f else ''} | {s['mean_loaded']:.1f} | {s['full']:.3f} "
          f"[{s['rag_full_same_k']:.3f}] | {s['full_ge4']:.3f} [{s['rag_full_ge4']:.3f}] | {s['cov']:.3f} | "
          f"{session_fsr(R):.3f} | {s['usd_1k']:.2f} |")
    # ranking control: same model ranks all rounds; cut at the size QSS chose
    fr = ROOT / "results/mm2/select_eval_memlens32k_rank38-999.json"
    fq = ROOT / "results/mm2/select_eval_memlens32k_toc38-40.json"
    if fr.exists() and fq.exists():
        Rk = {r["qid"]: r for r in json.load(open(fr))["per_q"]}
        Q = {r["qid"]: r for r in json.load(open(fq))["per_q"]}
        d, d4 = [], []
        for q, r in Q.items():
            k = max(1, len(r["ids"]))
            hit = float(set(r["gold"]) <= set(Rk[q]["ids"][:k]))
            d.append((r["full"], hit, len(r["gold"])))
        a = np.array([x[0] - x[1] for x in d]); lo, hi = boot_ci(a)
        a4 = np.array([x[0] - x[1] for x in d if x[2] >= 4]); lo4, hi4 = boot_ci(a4)
        p(f"\nRanking control (same model ranks all rounds, cut at the QSS size): FSR {np.mean([x[1] for x in d]):.3f} "
          f"vs QSS {np.mean([x[0] for x in d]):.3f}; QSS - ranking {a.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]; "
          f">=4 gold rounds {a4.mean():+.3f} [{lo4:+.3f}, {hi4:+.3f}] (n={len(a4)})")


def session_fsr(R):
    out = []
    for r in R:
        if not r.get("gold"):
            continue
        gs = {g.split(":")[0] for g in r["gold"]}
        ss = {i.split(":")[0] for i in r["ids"]}
        out.append(float(gs <= ss))
    return float(np.mean(out)) if out else float("nan")


def main():
    lite = "gemini-2.5-flash-lite"
    D = load("results/mm2/memlens_32k", lite)
    names = ["fc|read:medium", "rag:10|read:medium", "rag:20|read:medium", "rag:40|read:medium", "oracle|read:medium",
             "toc38:40|read:medium", "toc38:40|read:medium +snip600", "grade38:999:2|read:medium",
             "scu:5:2:40|read:medium", "toc@qwen:40|read:medium", "toc@gpt5m:40|read:medium", "utoc38:40:20|read:medium"]
    table(D, names, ["rag:40|read:medium", "fc|read:medium"], "MemLens-32k, reader gemini-2.5-flash-lite (699 questions)")
    by_type(D, names, "MemLens-32k accuracy by question type, reader gemini-2.5-flash-lite")
    S = load("results/mm2/memlens_32k_memlens_sub200", "gemini-3.8-flash")
    table(S, ["fc|read:low", "rag:40|read:low", "oracle|read:low", "toc38:40|read:low", "toc38:40|read+toc:low",
              "utoc38:40:40|read:low"],
          ["rag:40|read:low", "fc|read:low"], "MemLens-32k, reader gemini-3.8-flash (fixed 200-question subset)")
    L = load("results/mm2/memlens_128k_memlens_sub200", lite)
    table(L, ["fc|read:medium", "rag:40|read:medium", "oracle|read:medium", "toc38:40|read:medium", "grade38:999:2|read:medium"],
          ["rag:40|read:medium", "fc|read:medium"], "MemLens-128k, reader gemini-2.5-flash-lite (same 200-question subset)")
    sub = set(json.load(open(ROOT / "results/memlens_sub200.json")))
    D32s = {n: {q: r for q, r in v.items() if q in sub} for n, v in D.items()}
    table(D32s, ["fc|read:medium", "rag:40|read:medium", "oracle|read:medium", "toc38:40|read:medium", "grade38:999:2|read:medium"],
          ["rag:40|read:medium", "fc|read:medium"], "MemLens-32k restricted to the same 200 questions (for the 128k comparison)")
    # judge agreement
    p("\n### Judge agreement (Qwen3-VL-235B vs gemini-3.8-flash, same official prompt)\n")
    for n in names:
        if n in D:
            R = [r for r in D[n].values() if r.get("score2") is not None]
            if R:
                p(f"- {n}: n={len(R)}, exact agreement {np.mean([r['score'] == r['score2'] for r in R]):.3f}, "
                  f"Qwen {np.mean([r['score'] for r in R]):.3f} vs Gemini {np.mean([r['score2'] for r in R]):.3f}")
    selection()
    (ROOT / "results/TABLES_MEMLENS.md").write_text("# Held-out evaluation on MemLens (generated by scripts/analyze_memlens.py)\n"
                                                     + "\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
