"""Phase 16: per-question cosine ranks of the gold rounds and the user-message lengths of the ten most similar rounds
(retrieval key of each benchmark, v2 image descriptions). Cache-only: makes no API calls (a cache miss raises).
python scripts/p16_ranks.py memeye memlens memlens64 visualmem -> results/p16/ranks_<bench>.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "results/p16"))
from load import captions, episodes, load  # noqa: E402  (cache-only loader: the API client is disabled)
import mm  # noqa: E402

NAMES = {"memeye": "memeye", "memlens": "memlens_test", "memlens64": "memlens_test64", "visualmem": "visualmem", "camroll": "camroll"}


def main():
    for bench in sys.argv[1:]:
        eps = load(NAMES[bench])
        caps = {"short": captions(eps)[0]}
        key = "cap:dense" if bench == "memeye" else "cap:short"
        caps["dense"] = {p: mm.caption(p, "dense") for p in caps["short"]} if bench == "memeye" else caps["short"]
        out = {}
        for e in eps:
            ix = mm.RoundIndex(e, caps, [key])
            for qa in e.qas:
                order = np.argsort(-ix.scores(qa, key))
                rank = {int(i): k + 1 for k, i in enumerate(order)}
                r2i = e.rid2idx
                out[qa.qid] = {"gold": {e.rounds[r2i[c]].rid: rank[r2i[c]] for c in qa.clue if c in r2i},
                               "top10_user_len": float(np.mean([len(e.rounds[int(i)].user) for i in order[:10]]))}
        json.dump(out, open(ROOT / f"results/p16/ranks_{bench}.json", "w"))
        print(bench, len(out))


if __name__ == "__main__":
    main()
