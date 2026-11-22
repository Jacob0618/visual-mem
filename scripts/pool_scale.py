"""Phase 7: candidate-pool recall for long histories (MemEye + 1,000 distractor images). Which rounds should the strong
selector see when the index cannot be read in full? Cosine top-P vs cosine + BM25 fusion. No API calls (cached
embeddings)."""
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm  # noqa: E402
import scale  # noqa: E402
from run_mm import episodes  # noqa: E402

TOK = re.compile(r"[a-z0-9]+")


def bm25_scores(docs, query, k1=1.2, b=0.75):
    toks = [TOK.findall(d.lower()) for d in docs]
    N = len(toks); avg = sum(map(len, toks)) / max(1, N)
    df = Counter(t for ts in toks for t in set(ts))
    q = [t for t in TOK.findall(query.lower()) if len(t) > 1]
    out = np.zeros(N)
    for i, ts in enumerate(toks):
        tf = Counter(ts); L = len(ts)
        for t in q:
            if t in tf:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                out[i] += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * L / avg))
    return out


def main():
    eps = [scale.grow(e, 1000) for e in episodes("memeye", "all")]
    caps = {s: mm.build_captions(eps, s) for s in ["short", "dense"]}
    res = {}
    for e in eps:
        ix = mm.RoundIndex(e, caps, ["cap:dense"]); r2i = e.rid2idx
        docs = [r.user + " " + " ".join(caps["short"].get(p, "") for p in r.images) for r in e.rounds]
        for qa in e.qas:
            gold = {r2i[c] for c in qa.clue if c in r2i}
            if not gold:
                continue
            cs = ix.scores(qa, "cap:dense"); bs = bm25_scores(docs, qa.question)
            oc, ob = np.argsort(-cs), np.argsort(-bs)
            rr = np.zeros(len(e.rounds))
            rr[oc] += 1.0 / (60 + np.arange(len(oc))); rr[ob] += 1.0 / (60 + np.arange(len(ob)))
            of = np.argsort(-rr)
            g = len(gold)
            for P in [100, 200, 320, 500]:
                for name, o in [("cosine", oc), ("cosine + BM25 (RRF)", of)]:
                    res.setdefault((P, name), []).append((float(gold <= set(o[:P].tolist())), g))
    print(f"== MemEye + 1,000 distractor images: {np.mean([len(e.rounds) for e in eps]):.0f} rounds per history")
    for (P, name), v in sorted(res.items()):
        a = np.array(v)
        big = a[a[:, 1] >= 6, 0]
        print(f"   pool P={P:3d} {name:20s} pool FSR {a[:, 0].mean():.3f} | sets >=6 rounds {big.mean():.3f} (n={len(big)})")


if __name__ == "__main__":
    main()
