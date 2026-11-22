"""Phase 11: construction/query cost of thread-mode runs = the run's total (fresh + replayed calls of its process)
divided by the questions it processed; reading cost stays per question. No API calls.
python scripts/p11_fix_costs.py omni memeye 2700  [memlens 2659 ...]"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = {"gemini-3.8-flash": (0.75, 3.75), "gemini-2.5-flash-lite": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50)}


def run_total(pid, prefix):
    tot = 0.0
    for l in open(ROOT / "cache/usage_bl.jsonl"):
        r = json.loads(l)
        if r.get("pid") == pid and r["tag"].startswith(prefix):
            pi, po = P.get(r["model"], (0.75, 3.75))
            tot += r["prompt"] * pi / 1e6 + (r["out"] + r["thoughts"]) * po / 1e6
    return tot


def main():
    system = sys.argv[1]
    for bench, pid in zip(sys.argv[2::2], sys.argv[3::2]):
        p = ROOT / f"results/baselines/{bench}/{system}.jsonl"
        R = [json.loads(l) for l in open(p)]
        mine = [r for r in R if "thread mode" in r.get("cost_note", "")]
        nq = len({r["qid"] for r in mine})
        tot = run_total(int(pid), f"bl:{system}:{bench}:")
        for r in mine:
            r["usd_build_share"], r["usd_query"] = tot / max(1, nq), 0.0
            r["usd"] = r["usd_build_share"] + r["usd_read"]
            r["cost_note"] = f"thread mode: run total ${tot:.2f} over {nq} questions"
        open(p, "w").write("".join(json.dumps(r) + "\n" for r in R))
        print(f"{system} {bench}: run {pid} total ${tot:.2f} over {nq} questions -> ${1000 * tot / max(1, nq):.2f}/1k construction+query")


if __name__ == "__main__":
    main()
