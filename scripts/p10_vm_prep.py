"""Phase 10: VisualMem -- the frozen visibility estimate v_hat (top-10 cosine rounds, 200-character index lines,
cap:short key) and the gate split at tau = 0.7. Uses cached captions and embeddings."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mm  # noqa: E402
import mm2  # noqa: E402
from data_visualmem import load_visualmem  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    eps = load_visualmem()
    caps = mm.build_captions(eps, "short")
    V = {}
    for e in eps:
        ix = mm.RoundIndex(e, {"short": caps, "dense": caps}, ["cap:short"])
        for qa in e.qas:
            q = set(mm2._words(qa.question.split("\nA.")[0]))
            vals = []
            for i in ix.topk(qa, "cap:short", 10):
                r = e.rounds[i]
                full = set(mm2._words(r.user + " " + r.assistant + " " + " ".join(caps.get(p, "") for p in r.images)))
                line = set(mm2._words(mm2.toc_line(r, caps, e)))
                m = q & full
                if m:
                    vals.append(len(m & line) / len(m))
            V[qa.qid] = float(np.mean(vals)) if vals else 1.0
    out = ROOT / "results/p10"
    out.mkdir(exist_ok=True)
    json.dump(V, open(out / "vm_visibility_hat.json", "w"))
    qids = sorted(V)
    json.dump(qids, open(out / "vm_qids.json", "w"))
    json.dump([q for q in qids if V[q] >= 0.7], open(out / "vm_hivis.json", "w"))
    json.dump([q for q in qids if V[q] < 0.7], open(out / "vm_lovis.json", "w"))
    v = np.array(list(V.values()))
    print(f"VisualMem: {len(eps)} personas, {len(qids)} questions, {np.mean([len(e.rounds) for e in eps]):.1f} rounds per history, "
          f"v_hat mean {v.mean():.3f}, share >= 0.7: {np.mean(v >= 0.7):.3f}")


if __name__ == "__main__":
    main()
