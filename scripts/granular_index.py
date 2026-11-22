"""Phase 5: sentence-granular index. Units = user-message sentences and image captions of each round; a round is scored
by its best unit (max-sim). Compared with whole-round embeddings at equal budgets (FSR, coverage). Embeddings only.
python scripts/granular_index.py --bench memlens
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm  # noqa: E402
from gem import embed  # noqa: E402
from run_mm import episodes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QPREFIX = "task: search result | query: "
DPREFIX = "title: none | text: "


def units_of(r, caps):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", r.user) if len(s.strip()) > 15]
    out = [("u", s[:400]) for s in sents] or [("u", r.user[:400] or "(empty)")]
    out += [("img", caps[p]) for p in r.images if caps.get(p)]
    return out


class GranularIndex:
    def __init__(self, ep, caps):
        self.ep = ep
        self.units, self.owner = [], []
        for i, r in enumerate(ep.rounds):
            for kind, t in units_of(r, caps):
                self.units.append(t); self.owner.append(i)
        self.owner = np.array(self.owner)
        self.M = embed([DPREFIX + t for t in self.units])

    def round_scores(self, question):
        q = embed([QPREFIX + question])[0]
        s = self.M @ q
        out = np.full(len(self.ep.rounds), -1.0)
        np.maximum.at(out, self.owner, s)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="memlens")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    if a.bench == "memlens":
        eps = episodes("memlens", "32k", json.load(open(ROOT / f"results/memlens_{a.split}.json")))
        caps = {"short": mm.build_captions(eps, "short")}; caps["dense"] = caps["short"]; key = "cap:short"
    else:
        eps = episodes("memeye", "all"); caps = {s: mm.build_captions(eps, s) for s in ["short", "dense"]}; key = "cap:dense"
    res = {}
    for e in eps:
        gi = GranularIndex(e, caps["short"]); ix = mm.RoundIndex(e, caps, [key]); r2i = e.rid2idx
        for qa in e.qas:
            gold = {r2i[c] for c in qa.clue if c in r2i}
            if not gold:
                continue
            q = qa.question.split("\n", 1)[1] if qa.question.startswith("Question date:") else qa.question
            sr = ix.scores(qa, key); sg = gi.round_scores(q)
            ranks = {"round": np.argsort(-sr), "sentence max-sim": np.argsort(-sg)}
            # reciprocal-rank fusion of the two rankings
            rr = np.zeros(len(e.rounds))
            for o in ranks.values():
                rr[o] += 1.0 / (60 + np.arange(len(o)))
            ranks["fusion (RRF)"] = np.argsort(-rr)
            for name, o in ranks.items():
                for k in [10, 20, 40]:
                    top = set(o[:k].tolist())
                    res.setdefault((name, k), []).append((float(gold <= top), len(gold & top) / len(gold)))
    print(f"== {a.bench} ({a.split if a.bench == 'memlens' else 'all'}): full-set recall / coverage at equal budgets")
    for (name, k), v in sorted(res.items(), key=lambda x: (x[0][1], x[0][0])):
        v = np.array(v)
        print(f"   top-{k:<3d} {name:18s} FSR {v[:, 0].mean():.3f} | coverage {v[:, 1].mean():.3f}")


if __name__ == "__main__":
    main()
