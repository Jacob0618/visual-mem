"""Phase 6: measure the API-side prefix cache for a persistent per-user memory (MemEye). For each history, one warm-up
call, then the remaining questions of that history in parallel. Measures (a) the QSS selection call and (b) strong-reader
reading with the index as context, with the local response cache bypassed. Cost only; selections are compared with the
original run.  python scripts/cache_measure.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gem  # noqa: E402
import mm  # noqa: E402
import mm2  # noqa: E402
from gem import PRICES, pmap  # noqa: E402
from run_mm import episodes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def usd(u, m):
    p = PRICES[m]
    return ((u["prompt"] - u["cached"]) * p["in"] + u["cached"] * p["cached"] + (u["out"] + u["thoughts"]) * p["out"]) / 1e6


def main():
    gem.FRESH_TAGS = ["p2:toc_select", "p6:measure"]
    eps = episodes("memeye", "all")
    caps = {s: mm.build_captions(eps, s) for s in ["short", "dense"]}
    SEL = {r["qid"]: r for r in json.load(open(ROOT / "results/mm2/select_eval_toc38-40.json"))["per_q"]}
    rows = []
    for e in eps:
        qas = list(e.qas)

        def sel(qa):
            ids, u = mm2.toc_select(e, qa, caps["short"], 40, "gemini-3.8-flash", "low")
            return qa.qid, [e.rounds[i].rid for i in ids], u

        def rd(qa):
            idxs = [e.rid2idx[x] for x in SEL[qa.qid]["ids"] if x in e.rid2idx]
            _, us = mm2.read(e, qa, idxs, "low", caps, "gemini-3.8-flash", "low", with_toc=True, tag="p6:measure:read+toc")
            return qa.qid, us[0]

        out_s = [sel(qas[0])]; time.sleep(4)
        out_s += pmap(sel, qas[1:], workers=8, desc=f"sel {e.name[:20]}")
        out_r = [rd(qas[0])]; time.sleep(4)
        out_r += pmap(rd, qas[1:], workers=8, desc=f"read {e.name[:20]}")
        R = dict(out_r)
        for qid, rids, u in out_s:
            rows.append({"qid": qid, "task": e.name, "sel_usage": u, "sel_same": set(rids) == set(SEL[qid]["ids"]),
                         "read_usage": R[qid]})
    json.dump(rows, open(ROOT / "results/p6/cache_measure_memeye.json", "w"))
    su = [r["sel_usage"] for r in rows]; ru = [r["read_usage"] for r in rows]
    for name, U in [("QSS selection (3.8)", su), ("strong reading with index (3.8)", ru)]:
        cached = sum(u["cached"] for u in U) / max(1, sum(u["prompt"] for u in U))
        print(f"{name}: cached share of prompt tokens {cached:.3f}; measured $/1k {1000 * np.mean([usd(u, 'gemini-3.8-flash') for u in U]):.2f}; "
              f"uncached equivalent $/1k {1000 * np.mean([usd({**u, 'cached': 0}, 'gemini-3.8-flash') for u in U]):.2f}")
    print("fresh selection identical to the original:", round(np.mean([r["sel_same"] for r in rows]), 3))


if __name__ == "__main__":
    main()
