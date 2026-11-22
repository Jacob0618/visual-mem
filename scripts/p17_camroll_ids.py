"""Phase 17: round lists of the LLM-rerank baseline on CamRoll (top 40 of the ranking control over the 100 most similar
photos). No API calls.  -> results/p17/ids/camroll_rerank40.json"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
f = ROOT / "results/mm2/select_eval_camroll_rankr38-100_cnoassist+shortid+telegraph.json"
R = json.load(open(f))["per_q"]
out = {r["qid"]: r["ids"][:40] for r in R}
(ROOT / "results/p17/ids").mkdir(parents=True, exist_ok=True)
json.dump(out, open(ROOT / "results/p17/ids/camroll_rerank40.json", "w"))
print(len(out), "questions; mean list length", sum(len(v) for v in out.values()) / len(out))
