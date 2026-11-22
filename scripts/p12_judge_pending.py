"""Judge MemLens predictions whose judgment was deferred (score None) once the OpenRouter key works again.
.venv/bin/python scripts/p12_judge_pending.py vmem mma telemem"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import mm2  # noqa: E402
import p12_common as C  # noqa: E402

eps = C.load_bench("memlens")
QA = {q.qid: q for e in eps for q in e.qas}
for system in sys.argv[1:]:
    p = ROOT / f"results/baselines/memlens/{system}.jsonl"
    if not p.exists():
        continue
    R = [json.loads(l) for l in open(p)]
    n = 0
    for r in R:
        if not r.get("error") and r.get("score") is None:
            r["score"] = mm2.memlens_judge(QA[r["qid"]], r["pred"])["score"]
            n += 1
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in R))
    print(system, "judged", n)
