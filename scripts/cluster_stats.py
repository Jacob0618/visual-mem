"""Scenario-level uncertainty for MemEye (8 histories): per-scenario differences and a two-stage (cluster) bootstrap
that resamples scenarios and then questions within scenarios. No API calls. -> results/cluster_stats.md
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import boot_ci  # noqa: E402
from analyze_p2 import load_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def cluster_boot(diff_by_task, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    tasks = list(diff_by_task)
    arrs = [np.asarray(diff_by_task[t]) for t in tasks]
    out = np.empty(reps)
    for b in range(reps):
        pick = rng.integers(0, len(tasks), len(tasks))
        vals = np.concatenate([arrs[i][rng.integers(0, len(arrs[i]), len(arrs[i]))] for i in pick])
        out[b] = vals.mean()
    return np.quantile(out, [0.025, 0.975])


def selection_pairs():
    """full-set recall contrasts from selection-only records (per question)."""
    def load(name):
        return {r["qid"]: r for r in json.load(open(ROOT / f"results/mm2/select_eval_{name}.json"))["per_q"]}
    rk = {r["qid"]: r for r in json.load(open(ROOT / "results/mm2/rank_control_P100.json"))["per_q"]}
    qss = load("toc38-40")
    out = {}
    out["FSR: QSS - cosine (same size)"] = {q: (r["full"] - r["rag_full_same_k"], r["task"]) for q, r in qss.items()
                                           if r["full"] is not None}
    task_of = {q: r["task"] for q, r in qss.items()}
    out["FSR: QSS(P=100) - ranking control"] = {q: (r["qss_full"] - r["rank_full_kq"], task_of[q]) for q, r in rk.items()}
    return out


def main():
    lines = ["# Scenario-level uncertainty (MemEye, 8 scenario histories)\n",
             "Question bootstrap = 10k resamples of questions; cluster bootstrap = 10k resamples of scenarios, then of "
             "questions within each sampled scenario. Wilcoxon = signed-rank test over the 8 per-scenario differences.\n",
             "| contrast | mean | question-bootstrap 95% CI | cluster-bootstrap 95% CI | scenarios > 0 | Wilcoxon p |",
             "|---|---:|---|---|---:|---:|"]
    contrasts = [("gemini-2.5-flash-lite", "toc38:40|read:medium", "rag:40|read:medium"),
                 ("gemini-2.5-flash-lite", "toc38:40|read:medium", "fc|read:medium"),
                 ("gemini-2.5-flash-lite", "toc38:40|read:medium +think1024", "rag:40|read:medium +think1024"),
                 ("gemini-2.5-flash-lite", "scu:5:2:40|read:medium", "rag:40|read:medium"),
                 ("gemini-2.5-flash-lite", "toc:40|read:medium +think1024 +usageonline", "rag:40|read:medium +think1024"),
                 ("gemini-3.8-flash", "toc38:40|read+toc:low", "rag:40|read:low"),
                 ("gemini-3.8-flash", "toc38:40|read:low", "rag:40|read:low"),
                 ("gemini-3.8-flash", "toc38:40|read+toc:low", "fc|read:low")]
    cache = {}
    rows = {}
    for reader, a, b in contrasts:
        if reader not in cache:
            cache[reader] = load_all(reader)
        D = cache[reader]
        q = sorted(set(D[a]) & set(D[b]))
        rows[f"{reader.replace('gemini-', '')}: {a} - {b}"] = {x: (D[a][x]["score"] - D[b][x]["score"], D[a][x]["task"]) for x in q}
    rows.update(selection_pairs())
    for name, d in rows.items():
        by = {}
        for v, t in d.values():
            by.setdefault(t, []).append(v)
        allv = np.array([v for v, _ in d.values()])
        lo, hi = boot_ci(allv)
        clo, chi = cluster_boot(by)
        per = [np.mean(v) for v in by.values()]
        try:
            p = wilcoxon(per).pvalue
        except ValueError:
            p = float("nan")
        lines.append(f"| {name} | {allv.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | [{clo:+.3f}, {chi:+.3f}] | "
                     f"{sum(x > 0 for x in per)}/{len(per)} | {p:.3f} |")
    (ROOT / "results/cluster_stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
