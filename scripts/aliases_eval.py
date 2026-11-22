"""Write-time alias enrichment (strong model, once per history) and its effect on cheap query-time components:
cosine full-set recall with alias-enriched keys, and TOC selection (cheap / strong) with alias-enriched index lines.
python scripts/aliases_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm  # noqa: E402
import mm2  # noqa: E402
from gem import embed, pmap, spend  # noqa: E402
from run_mm import episodes  # noqa: E402
from run_mm2 import usd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/mm2/aliases"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    eps = episodes("memeye", "all")
    caps = {s: mm.build_captions(eps, s) for s in ["short", "dense"]}
    write_usd = 0.0
    for e in eps:
        f = OUT / f"aliases_{e.name}.json"
        if f.exists():
            mm2.ALIASES[e.name] = json.load(open(f))["aliases"]
            write_usd += json.load(open(f))["usd"]
            continue
        al, us = mm2.build_aliases(e, caps["short"])
        u = sum(usd([x], "gemini-3.8-flash") for x in us)
        write_usd += u
        mm2.ALIASES[e.name] = al
        json.dump({"aliases": al, "usd": u}, open(f, "w"), indent=0)
    n_img = sum(1 for e in eps for r in e.rounds if r.images)
    n_cov = sum(1 for e in eps for r in e.rounds if r.images and mm2.ALIASES[e.name].get(r.rid))
    print(f"write-time aliases: {n_cov}/{n_img} image rounds covered; total write cost ${write_usd:.3f}")
    # cosine with alias-enriched keys
    res = {}
    for name, enrich in [("plain", False), ("alias", True)]:
        full = {k: [] for k in [5, 10, 20, 40]}
        for e in eps:
            items = []
            for r in e.rounds:
                txt = mm.round_plain(r, caps["dense"])
                if enrich and mm2.ALIASES[e.name].get(r.rid):
                    txt += "\nAlso referred to as: " + "; ".join(mm2.ALIASES[e.name][r.rid])
                items.append(mm.DOC_FMT.format(txt))
            M = embed(items)
            ix = mm.RoundIndex.__new__(mm.RoundIndex)
            ix.ep, ix.M = e, {"k": M}
            r2i = e.rid2idx
            for qa in e.qas:
                gold = {r2i[c] for c in qa.clue if c in r2i}
                if not gold:
                    continue
                order = list(np.argsort(-ix.scores(qa, "k")))
                for k in full:
                    full[k].append(float(gold <= set(order[:k])))
        res[name] = {k: round(float(np.mean(v)), 3) for k, v in full.items()}
        print(f"cosine full-set recall, {name} keys:", res[name])
    json.dump(res, open(OUT / "cosine_fullset.json", "w"))
    print("spend total", spend().get("TOTAL"))


if __name__ == "__main__":
    main()
