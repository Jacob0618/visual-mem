"""Phase 14: the visibility estimate v_hat on the re-runs with fresh image descriptions (MemEye, VisualMem), with the
definition of the paper: ten rounds most similar to the question (the retrieval key of the benchmark), share of the
question words present in a round that its 200-character index line shows. Uses cached captions and embeddings.
python scripts/p14_vhat.py -> results/p14/vhat_<bench>.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import mm  # noqa: E402
import mm2  # noqa: E402
from run_mm import episodes  # noqa: E402


def vhat(bench):
    eps = episodes(bench, "all")
    caps = {"short": mm.build_captions(eps, "short")}
    caps["dense"] = mm.build_captions(eps, "dense") if bench == "memeye" else caps["short"]
    key = "cap:dense" if bench == "memeye" else "cap:short"
    V = {}
    for e in eps:
        ix = mm.RoundIndex(e, caps, [key])
        for qa in e.qas:
            q = set(mm2._words(qa.question.split("\nA.")[0]))
            vals = []
            for i in ix.topk(qa, key, 10):
                r = e.rounds[i]
                full = set(mm2._words(r.user + " " + r.assistant + " " + " ".join(caps["short"].get(p, "") for p in r.images)))
                line = set(mm2._words(mm2.toc_line(r, caps["short"], e)))
                m = q & full
                if m:
                    vals.append(len(m & line) / len(m))
            V[qa.qid] = float(np.mean(vals)) if vals else 1.0
    return V


def main():
    for bench in sys.argv[1:] or ["memeye", "visualmem"]:
        V = vhat(bench)
        json.dump(V, open(ROOT / f"results/p14/vhat_{bench}.json", "w"))
        v = np.array(list(V.values()))
        print(f"{bench}: {len(v)} questions, mean {v.mean():.3f}, share >= 0.7: {np.mean(v >= 0.7):.3f}")


if __name__ == "__main__":
    main()
