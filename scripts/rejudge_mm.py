"""Second-judge robustness check: re-score existing multimodal predictions with a different judge model
(same MemEye rubric) and store it as field 'score_<model>'.
python scripts/rejudge_mm.py --model gemma-4-31b-it results/mm/memeye_dev/fc__raw-low.jsonl ...
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mm  # noqa: E402
from gem import generate, generate_or, pmap  # noqa: E402


def judge2(q, gt, pred, model):
    p = mm._JUDGE_T.replace("{{question}}", q).replace("{{ground_truth}}", gt).replace("{{model_output}}", pred)
    for attempt in range(3):
        if "/" in model:   # cross-family judge through OpenRouter
            r = generate_or(p, model=model, tag=f"judge2:{model}", seed=attempt or None, max_output_tokens=1024)
        else:
            r = generate(p, model=model, thinking=None if "gemma" in model else "low", tag=f"judge2:{model}",
                         seed=attempt or None)
        m = re.search(r'"score"\s*:\s*"?([0-9.]+)', r["text"] or "")
        if m and float(m.group(1)) in (0, 0.25, 0.5, 0.75, 1):
            return float(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-31b-it")
    ap.add_argument("--sample", type=int, default=0, help="only re-judge this many questions (same qids in every file)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    keep = None
    if a.sample:
        import random
        q0 = sorted(json.loads(l)["qid"] for l in open(a.files[0]))
        keep = set(random.Random(a.seed).sample(q0, a.sample))
    field = "score_" + a.model.replace("-", "_").replace(".", "_").replace("/", "_")
    for f in a.files:
        R = [json.loads(l) for l in open(f)]
        todo = [r for r in R if field not in r and (keep is None or r["qid"] in keep)]
        def safe(r):
            try:
                return judge2(r["question"], r["gold"], r["pred"], a.model)
            except Exception:  # quota errors etc. -> leave unjudged, can be resumed
                return None
        S = pmap(safe, todo, workers=a.workers)
        for r, s in zip(todo, S):
            if s is not None:
                r[field] = s
        with open(f, "w") as fh:
            for r in R:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        ok = [r for r in R if r.get(field) is not None]
        print(f, "n", len(ok), "judge1", round(sum(r["score"] for r in ok) / len(ok), 4),
              "judge2", round(sum(r[field] for r in ok) / len(ok), 4),
              "exact-agree", round(sum(r["score"] == r[field] for r in ok) / len(ok), 3))


if __name__ == "__main__":
    main()
