"""Phase 12: per-question cost of TeleMem from its ledger. Its runs were sharded across processes that shared one
ledger, so the per-episode construction cost written by the runner is unreliable; here the construction spend of each
benchmark (all processes that ran it, including restarted ones, whose cached calls the final runs reused) is divided by
the benchmark's questions, and the per-question query and reading costs are kept. No API calls.
python scripts/p12_fix_costs.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMLENS_PIDS = {17524, 17843, 18307, 18315, 18316, 18676, 18677, 18678, 18679}
VISUALMEM_PIDS = {18319, 18320, 18321, 18322}


def main():
    L = [json.loads(l) for l in open(ROOT / "results/p12_budget/telemem.jsonl")]
    build = {"memeye": 0.0, "memlens": 0.0, "visualmem": 0.0}
    for r in L:
        if not r["what"].startswith("build"):
            continue
        b = "memlens" if r["pid"] in MEMLENS_PIDS else "visualmem" if r["pid"] in VISUALMEM_PIDS else "memeye"
        build[b] += r["usd"]
    for b, tot in build.items():
        p = ROOT / f"results/baselines/{b}/telemem.jsonl"
        if not p.exists():
            continue
        R = [json.loads(l) for l in open(p)]
        nq = len({r["qid"] for r in R if not r.get("error")})
        for r in R:
            if r.get("error"):
                continue
            r["usd_build_share"] = tot / max(1, nq)
            r["usd"] = r["usd_build_share"] + r.get("usd_query", 0.0) + r["usd_read"]
            r["cost_note"] = f"construction: ledger total ${tot:.2f} over {nq} questions"
        open(p, "w").write("".join(json.dumps(r) + "\n" for r in R))
        print(f"TeleMem {b}: construction ${tot:.2f} over {nq} questions -> ${1000 * tot / max(1, nq):.2f}/1k")


if __name__ == "__main__":
    main()
