"""Ranking control for set semantics: the strong model ranks the same cosine top-P candidates (RankGPT-style);
we take the top-k at the size that QSS (ptoc38:P:40) chose for each question, and at fixed k.
python scripts/rank_control.py --P 100
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm  # noqa: E402
import mm2  # noqa: E402
from gem import pmap  # noqa: E402
from run_mm import episodes  # noqa: E402
from run_mm2 import usd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KEY = "cap:dense"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=100)
    ap.add_argument("--recall", action="store_true", help="ranking prompt with the recall-oriented instructions of QSS")
    a = ap.parse_args()
    eps = episodes("memeye", "all")
    caps = {s: mm.build_captions(eps, s) for s in ["short", "dense"]}
    idx = {e.name: mm.RoundIndex(e, caps, [KEY]) for e in eps}
    Q = {r["qid"]: r for r in json.load(open(ROOT / f"results/mm2/select_eval_ptoc38-{a.P}-40.json"))["per_q"]}

    def one(job):
        e, qa = job
        r2i = e.rid2idx
        gold = {r2i[c] for c in qa.clue if c in r2i}
        pre = idx[e.name].topk(qa, KEY, a.P)
        ranking, u = mm2.rank_select(e, qa, caps["short"], pre, "gemini-3.8-flash", "low", recall=a.recall)
        cos = pre  # cosine order
        k_q = max(1, len(Q[qa.qid]["ids"]))
        out = {"qid": qa.qid, "n_gold": len(gold), "k_qss": k_q, "usd": usd([u], "gemini-3.8-flash"),
               "qss_full": Q[qa.qid]["full"]}
        for name, order in [("rank", ranking), ("cos", cos)]:
            for k in [k_q, 10, 20]:
                tag = "kq" if k == k_q else str(k)
                if gold:
                    out[f"{name}_full_{tag}"] = float(gold <= set(order[:k]))
            if gold:
                out[f"{name}_full_10"] = float(gold <= set(order[:10]))
                out[f"{name}_full_20"] = float(gold <= set(order[:20]))
        return out

    jobs = [(e, qa) for e in eps for qa in e.qas]
    R = [r for r in pmap(one, jobs, workers=12, desc="rank") if r["n_gold"]]
    res = {"P": a.P, "n": len(R), "mean_k_qss": round(float(np.mean([r["k_qss"] for r in R])), 2),
           "qss_full": round(float(np.mean([r["qss_full"] for r in R])), 3),
           "rank_full_at_qss_size": round(float(np.mean([r["rank_full_kq"] for r in R])), 3),
           "cos_full_at_qss_size": round(float(np.mean([r["cos_full_kq"] for r in R])), 3),
           "rank_full_10": round(float(np.mean([r["rank_full_10"] for r in R])), 3),
           "cos_full_10": round(float(np.mean([r["cos_full_10"] for r in R])), 3),
           "rank_full_20": round(float(np.mean([r["rank_full_20"] for r in R])), 3),
           "cos_full_20": round(float(np.mean([r["cos_full_20"] for r in R])), 3)}
    for lo, hi, lab in [(1, 1, "1"), (2, 3, "2-3"), (4, 99, ">=4")]:
        G = [r for r in R if lo <= r["n_gold"] <= hi]
        res[f"by_{lab}"] = {"n": len(G), "qss": round(float(np.mean([r["qss_full"] for r in G])), 3),
                            "rank@qss_size": round(float(np.mean([r["rank_full_kq"] for r in G])), 3),
                            "cos@qss_size": round(float(np.mean([r["cos_full_kq"] for r in G])), 3)}
    res["usd_1k"] = round(1000 * float(np.mean([r["usd"] for r in R])), 2)
    print(json.dumps(res, indent=1))
    json.dump({"summary": res, "per_q": R}, open(ROOT / f"results/mm2/rank_control_P{a.P}{'_recall' if a.recall else ''}.json", "w"))


if __name__ == "__main__":
    main()
