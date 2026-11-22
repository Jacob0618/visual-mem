"""Phase 8: WorldMemArena lifelong-personal -- question ids, the frozen visibility estimate v_hat (top-10 cosine rounds,
200-character index lines, cap:short key), and the high/low split at tau = 0.7. Uses cached captions and embeddings."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mm  # noqa: E402
import mm2  # noqa: E402
from data_wma import EVIDENCE_SESSIONS, load_wma  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    eps = [e for e in load_wma() if e.name.startswith("personal_")]
    caps = mm.build_captions(eps, "short")
    V = {}
    for e in eps:
        ix = mm.RoundIndex(e, {"short": caps, "dense": caps}, ["cap:short"])
        for qa in e.qas:
            q = set(mm2._words(qa.question)); vals = []
            for i in ix.topk(qa, "cap:short", 10):
                r = e.rounds[i]
                full = set(mm2._words(r.user + " " + r.assistant + " " + " ".join(caps.get(p, "") for p in r.images)))
                line = set(mm2._words(mm2.toc_line(r, caps, e)))
                m = q & full
                if m:
                    vals.append(len(m & line) / len(m))
            V[qa.qid] = float(np.mean(vals)) if vals else 1.0
    qids = sorted(V)
    out = ROOT / "results/p8"
    out.mkdir(exist_ok=True)
    json.dump(V, open(out / "wma_visibility_hat.json", "w"))
    json.dump(qids, open(out / "wma_personal_qids.json", "w"))
    json.dump([q for q in qids if V[q] >= 0.7], open(out / "wma_personal_hivis.json", "w"))
    json.dump([q for q in qids if V[q] < 0.7], open(out / "wma_personal_lovis.json", "w"))
    json.dump({q: EVIDENCE_SESSIONS[q] for q in qids}, open(out / "wma_evidence_sessions.json", "w"))
    v = np.array(list(V.values()))
    print(f"WorldMemArena personal: {len(eps)} histories, {len(qids)} questions, rounds/history {np.mean([len(e.rounds) for e in eps]):.1f}, "
          f"v_hat mean {v.mean():.3f}, share >= 0.7: {np.mean(v >= 0.7):.3f}")


if __name__ == "__main__":
    main()
