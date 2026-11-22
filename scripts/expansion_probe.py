"""Phase 5 probe: once one member of an evidence set is found, how well do (a) image-image similarity, (b) round
text+caption similarity to the found member, and (c) question similarity rank the OTHER members (image rounds only)?
python scripts/expansion_probe.py --bench memeye | --bench memlens --split dev
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm  # noqa: E402
from run_mm import episodes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="memeye")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    if a.bench == "memlens":
        eps = episodes("memlens", "32k", json.load(open(ROOT / f"results/memlens_{a.split}.json")))
        caps = {"short": mm.build_captions(eps, "short")}; caps["dense"] = caps["short"]; key = "cap:short"
    else:
        eps = episodes("memeye", "all"); caps = {s: mm.build_captions(eps, s) for s in ["short", "dense"]}; key = "cap:dense"
    R = {"image-image to seed": [], "text+caption to seed": [], "question": []}
    nq = 0
    for e in eps:
        ix = mm.RoundIndex(e, caps, [key, "img"]); r2i = e.rid2idx
        img_rounds = [i for i, r in enumerate(e.rounds) if r.images]
        if len(img_rounds) < 3:
            continue
        for qa in e.qas:
            gold = [r2i[c] for c in qa.clue if c in r2i and e.rounds[r2i[c]].images]
            if len(gold) < 2:
                continue
            nq += 1
            qs = ix.scores(qa, key)
            seed = max(gold, key=lambda i: qs[i])
            others = [i for i in gold if i != seed]
            pool = [i for i in img_rounds if i != seed]
            crit = {"image-image to seed": ix.M["img"][pool] @ ix.M["img"][seed],
                    "text+caption to seed": ix.M[key][pool] @ ix.M[key][seed],
                    "question": qs[pool]}
            for name, s in crit.items():
                order = [pool[j] for j in np.argsort(-s)]
                rank = {i: r for r, i in enumerate(order)}
                R[name] += [rank[o] for o in others]
    print(f"== {a.bench}: questions with >=2 gold image rounds: {nq}; other members ranked among {''}image rounds")
    for name, v in R.items():
        v = np.array(v)
        print(f"   {name:22s} median rank {np.median(v):.0f} | within top-3 {np.mean(v < 3):.3f} | top-5 {np.mean(v < 5):.3f} | top-10 {np.mean(v < 10):.3f}")


if __name__ == "__main__":
    main()
