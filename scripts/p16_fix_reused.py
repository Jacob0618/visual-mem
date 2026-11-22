"""Phase 16: remove baseline records (and run dirs) of episodes that were built on top of an earlier store, so that the
resumable runner rebuilds them from scratch.  python scripts/p16_fix_reused.py <system> <bench> <qids.json>
The output file is backed up to results/p16/bl_discarded/ first. Run dirs are matched by question id (MemLens) or by the
task name of the removed records' questions (MemEye, VisualMem: pass the question ids of the whole history)."""
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    system, bench, qf = sys.argv[1:4]
    qids = set(json.load(open(qf)))
    out = ROOT / f"results/p16/baselines/{bench}/{system}.jsonl"
    R = [json.loads(l) for l in open(out)]
    keep = [r for r in R if r["qid"] not in qids]
    bak = ROOT / f"results/p16/bl_discarded/{system}_{bench}_before_fix_{time.strftime('%H%M%S')}.jsonl"
    bak.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, bak)
    with open(out, "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    n = 0
    for q in qids:
        d = ROOT / "baselines/runs" / system / bench / re.sub(r"[^A-Za-z0-9_.-]", "_", q)
        if d.exists():
            shutil.rmtree(d)
            n += 1
    print(f"{system} {bench}: removed {len(R) - len(keep)} records of {len(qids)} questions, {n} run dirs; backup {bak.name}")


if __name__ == "__main__":
    main()
