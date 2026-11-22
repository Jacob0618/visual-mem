"""Add a second (lenient, Mem0-style) judge score to existing LoCoMo prediction files: field 'score_mem0'.
python scripts/rejudge_text.py results/text/locomo
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import textmem  # noqa: E402
from gem import pmap  # noqa: E402

for f in sorted(glob.glob(sys.argv[1] + "/*.jsonl")):
    R = [json.loads(l) for l in open(f)]
    if all("score_mem0" in r for r in R):
        continue
    S = pmap(lambda r: textmem.judge_mem0(r["question"], r["gold"], r["pred"]), R, workers=32)
    for r, s in zip(R, S):
        r["score_mem0"] = s
    with open(f, "w") as fh:
        for r in R:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f, "strict", round(sum(r["score"] for r in R) / len(R), 4), "mem0", round(sum(S) / len(S), 4))
