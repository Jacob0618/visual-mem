"""Phase 6: classify each question as single-item vs multi-item (set) from the question text only, with the cheap
model. Output: results/p6/setness.json {qid: true (multi) | false}.  python scripts/setness.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm2  # noqa: E402
from gem import generate, pmap  # noqa: E402
from run_mm import episodes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SYS = ("You classify questions that a user asks an assistant about their own past conversations (which include photos "
       "and screenshots). Decide whether answering requires combining information about TWO OR MORE separate items, "
       "events, images, or points in time from the history (for example counting, totals, comparisons, ordering, "
       "durations between events, how something changed over time, or all members of a group), or whether ONE fact, "
       "item, or image is enough. Return JSON {\"multi\": true|false}.")
_SCHEMA = {"type": "object", "properties": {"multi": {"type": "boolean"}}, "required": ["multi"]}


def classify(q: str) -> bool:
    q = q.split("\n", 1)[1] if q.startswith("Question date:") else q
    r = generate([f"Question: {q}"], model="gemini-2.5-flash-lite", system=SYS, thinking=0, json_schema=_SCHEMA,
                 tag="p6:setness", max_output_tokens=50)
    return bool(mm2.parse_json(r["text"]).get("multi"))


def main():
    out = {}
    sub = set(json.load(open(ROOT / "results/memgal_subset.json"))["qids"])
    for bench, split in [("memeye", "all"), ("memlens", "32k"), ("memgal", "all")]:
        eps = episodes(bench, split)
        jobs = [qa for e in eps for qa in e.qas if bench != "memgal" or qa.qid in sub]
        for qid, m in pmap(lambda qa: (qa.qid, classify(qa.question)), jobs, workers=16, desc=bench):
            out[qid] = m
        print(bench, len(jobs), "multi share", round(sum(out[qa.qid] for qa in jobs) / len(jobs), 3), flush=True)
    json.dump(out, open(ROOT / "results/p6/setness.json", "w"))


if __name__ == "__main__":
    main()
