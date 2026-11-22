"""Phase 15: build MIRIX memories for the latency samples despite MIRIX hanging inside its agents. Each history runs in
its own process (scripts/p11_mirix.py --direct, .venv_mirix); when an add hangs, the worker records the session and
exits, and this driver restarts it, which resumes after the last completed session (a session that hangs twice is
skipped). Contexts and retrieval times go to --out; construction time = summed durations of the completed adds.
.venv/bin/python scripts/p15_mirix_driver.py --bench memlens --qids_file results/p13/latency_qids_memlens.json"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--qids_file", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--home_root", default=str(ROOT / "baselines/runs/mirix_lat2"))
    ap.add_argument("--parallel", type=int, default=5)
    ap.add_argument("--max_restarts", type=int, default=12)
    a = ap.parse_args()
    from run_mm import episodes
    qids = json.load(open(a.qids_file))
    eps = episodes(a.bench, "32k" if a.bench == "memlens" else "all", qids if a.bench != "memeye" else None)
    groups = [[q.qid for q in e.qas if q.qid in set(qids)] for e in eps]
    groups = [g for g in groups if g]
    out = Path(a.out or ROOT / f"results/p13/latency/{a.bench}/mirix_ctx_build.jsonl")
    part = out.parent / "mirix_parts"
    part.mkdir(parents=True, exist_ok=True)
    log = ROOT / f"results/p15/mirix_{a.bench}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    def run(k, g):
        qf, of = part / f"q{k}.json", part / f"ctx{k}.jsonl"
        json.dump(g, open(qf, "w"))
        for attempt in range(a.max_restarts):
            have = {json.loads(l)["qid"] for l in open(of)} if of.exists() else set()
            if set(g) <= have:
                return k, attempt, "done"
            with open(log, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} history {k} attempt {attempt}\n")
                f.flush()
                rc = subprocess.run([str(ROOT / ".venv_mirix/bin/python"), "scripts/p11_mirix.py", "--bench", a.bench,
                                     "--qids_file", str(qf), "--direct", "--out", str(of), "--home_root", a.home_root],
                                    cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
                f.write(f"{time.strftime('%H:%M:%S')} history {k} exit {rc}\n")
        return k, a.max_restarts, "gave up"

    with ThreadPoolExecutor(a.parallel) as ex:
        for k, att, status in ex.map(lambda kg: run(*kg), enumerate(groups)):
            print(f"history {k}: {status} after {att} restarts", flush=True)
    recs = []
    for k in range(len(groups)):
        of = part / f"ctx{k}.jsonl"
        if of.exists():
            recs += [json.loads(l) for l in open(of)]
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"{len(recs)} contexts -> {out}", flush=True)


if __name__ == "__main__":
    main()
