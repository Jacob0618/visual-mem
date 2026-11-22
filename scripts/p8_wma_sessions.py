"""Phase 8: session-level retrieval check on WorldMemArena lifelong-personal (evidence is annotated per session):
share of questions whose evidence sessions are all touched by the opened rounds, for QSS and for cosine retrieval of the
same size per question. No API calls (cached embeddings)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm  # noqa: E402
from analyze import boot_ci  # noqa: E402
from data_wma import load_wma  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ES = json.load(open(ROOT / "results/p8/wma_evidence_sessions.json"))


def main():
    eps = {e.name: e for e in load_wma() if e.name.startswith("personal_")}
    caps = mm.build_captions(list(eps.values()), "short")
    Q = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "results/mm2/wma_all_wma_personal_qids/toc38-40__read-medium__gemini-2.5-flash-lite__xcache.jsonl")}
    a, b = [], []
    for qid, r in Q.items():
        e = eps[r["task"]]
        qa = next(x for x in e.qas if x.qid == qid)
        ev = set(ES.get(qid, []))
        if not ev:
            continue
        S = set(x.split(":")[0] for x in r["retrieved"])
        ix = mm.RoundIndex(e, {"short": caps, "dense": caps}, ["cap:short"])
        C = set(e.rounds[i].sid for i in ix.topk(qa, "cap:short", max(1, len(r["retrieved"]))))
        a.append(float(ev <= S)); b.append(float(ev <= C))
    d = np.array(a) - np.array(b); lo, hi = boot_ci(d)
    print(f"WorldMemArena personal ({len(a)} q): all evidence sessions touched: QSS {np.mean(a):.3f}, cosine same size {np.mean(b):.3f}, diff {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
