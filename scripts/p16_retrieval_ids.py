"""Phase 16: round lists of two stronger retrieval baselines at the budget of RAG@40 (no API calls; cache-only).
  hybrid40  reciprocal-rank fusion (k = 60) of the dense ranking (the retrieval key of RAG@k) and BM25 over the same
            round text (dialogue + image descriptions); top 40 rounds.
  rerank40  dense top-100 candidates re-ranked by gemini-3.8-flash (the ranking control of the ablation, which uses the
            selector instruction and the same index lines); top 40 rounds.
Also prints full-set recall of dense / BM25 / hybrid / rerank at 40 rounds.
python scripts/p16_retrieval_ids.py memeye memlens visualmem -> results/p16/ids/<bench>_{hybrid40,rerank40}.json"""
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "results/p16"))
from load import captions, load  # noqa: E402  (cache-only loader: the API client is disabled)
import mm  # noqa: E402
import mm2  # noqa: E402

NAMES = {"memeye": "memeye", "memlens": "memlens_test", "visualmem": "visualmem", "camroll": "camroll"}
C = "cnoassist+shortid+telegraph"
RANK = {"memeye": f"select_eval_rankr38-100_{C}.json", "visualmem": f"select_eval_visualmem_rankr38-100_{C}.json",
        "memlens": f"select_eval_memlens32k_memlens_test_rankr38-100_{C}.json",
        "camroll": f"select_eval_camroll_rankr38-100_{C}.json"}


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.toks = [mm2._words(d) for d in docs]
        self.N, self.k1, self.b = len(docs), k1, b
        self.avg = float(np.mean([len(t) for t in self.toks])) or 1.0
        self.df = collections.Counter(w for t in self.toks for w in set(t))
        self.tf = [collections.Counter(t) for t in self.toks]

    def scores(self, query):
        q = mm2._words(query)
        out = np.zeros(self.N)
        for i, (tf, t) in enumerate(zip(self.tf, self.toks)):
            s = 0.0
            for w in q:
                if w in tf:
                    idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    s += idf * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * len(t) / self.avg))
            out[i] = s
        return out


def main():
    out_dir = ROOT / "results/p16/ids"
    out_dir.mkdir(parents=True, exist_ok=True)
    for bench in sys.argv[1:] or ["memeye", "memlens", "visualmem"]:
        eps = load(NAMES[bench])
        caps = {"short": captions(eps)[0]}
        key = "cap:dense" if bench == "memeye" else "cap:short"
        caps["dense"] = {p: mm.caption(p, "dense") for p in caps["short"]} if bench == "memeye" else caps["short"]
        rank = {r["qid"]: r["ids"] for r in json.load(open(ROOT / "results/mm2" / RANK[bench]))["per_q"]}
        hyb, rer, fsr = {}, {}, collections.defaultdict(list)
        for e in eps:
            ix = mm.RoundIndex(e, caps, [key])
            bm = BM25([mm.round_plain(r, caps[key[4:]]) for r in e.rounds])
            r2i = e.rid2idx
            for qa in e.qas:
                d = np.argsort(-ix.scores(qa, key), kind="stable")
                s = np.argsort(-bm.scores(qa.question.split("\nA.")[0]), kind="stable")
                rr = collections.defaultdict(float)
                for order in (d, s):
                    for k, i in enumerate(order):
                        rr[int(i)] += 1.0 / (60 + k + 1)
                h = sorted(rr, key=lambda i: (-rr[i], i))[:40]
                hyb[qa.qid] = [e.rounds[i].rid for i in h]
                if qa.qid in rank:
                    rer[qa.qid] = rank[qa.qid][:40]
                gold = {r2i[c] for c in qa.clue if c in r2i}
                if gold:
                    fsr["dense@40"].append(gold <= set(int(i) for i in d[:40]))
                    fsr["bm25@40"].append(gold <= set(int(i) for i in s[:40]))
                    fsr["hybrid@40"].append(gold <= set(h))
                    if qa.qid in rank:
                        fsr["rerank@40"].append(gold <= {r2i[x] for x in rank[qa.qid][:40] if x in r2i})
                        fsr["dense@100"].append(gold <= set(int(i) for i in d[:100]))
        json.dump(hyb, open(out_dir / f"{bench}_hybrid40.json", "w"))
        json.dump(rer, open(out_dir / f"{bench}_rerank40.json", "w"))
        print(bench, len(hyb), "questions |", " ".join(f"{k} FSR {np.mean(v):.3f}" for k, v in fsr.items()), "| reranked:", len(rer))


if __name__ == "__main__":
    main()
