"""Retrieval-only diagnostics on multimodal benchmarks: gold-clue recall of different round keys and fusions.

python scripts/retrieval_mm.py --bench memeye --split dev
Writes results/mm/retrieval_<bench>_<split>.json
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mm  # noqa: E402
from run_mm import episodes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KS = [1, 3, 5, 10, 20]


def zs(s):
    return (s - s.mean()) / (s.std() + 1e-9)


def rrf(*ss, c=60):
    out = 0
    for s in ss:
        rank = np.empty_like(s)
        rank[np.argsort(-s)] = np.arange(len(s))
        out = out + 1.0 / (c + rank + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="memeye")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--styles", nargs="*", default=["short", "dense"])
    ap.add_argument("--qids", default=None)
    args = ap.parse_args()
    eps = episodes(args.bench, args.split)
    if args.qids:
        keep = set(json.load(open(args.qids))["qids"])
        for e in eps:
            e.qas = [q for q in e.qas if q.qid in keep]
    caps = {s: mm.build_captions(eps, s) for s in args.styles}
    keys = ["text", "inter", "img"] + [f"cap:{s}" for s in args.styles]
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    gap = collections.defaultdict(list)
    perq = collections.defaultdict(dict)  # key -> qid -> coverage@10
    for e in eps:
        ix = mm.RoundIndex(e, caps, keys)
        has_img = np.array([bool(r.images) for r in e.rounds])
        r2i = e.rid2idx
        for qa in e.qas:
            gold = {r2i[c] for c in qa.clue if c in r2i}
            if not gold:
                continue
            S = {k: ix.scores(qa, k) for k in keys}
            S["rrf(cap:dense,inter)"] = rrf(S["cap:dense"], S["inter"])
            S["rrf(cap:dense,img)"] = rrf(S["cap:dense"], S["img"])
            zi = np.full(len(has_img), -9.0)
            if has_img.sum() > 1:
                zi[has_img] = zs(S["img"][has_img])   # z-normalise image scores among image rounds only
            S["maxz(text,img)"] = np.maximum(zs(S["text"]), zi)
            S["mean(cap:dense,inter)"] = (zs(S["cap:dense"]) + zs(S["inter"])) / 2
            grp = [qa.y or qa.qtype, "ALL", "gold_has_img" if any(has_img[g] for g in gold) else "gold_text_only"]
            for name, s in S.items():
                order = np.argsort(-s)
                perq[name][qa.qid] = len(gold & set(order[:10].tolist())) / len(gold)
                for k in KS:
                    top = set(order[:k].tolist())
                    for g in grp:
                        res[(name, g)][f"cov@{k}"].append(len(gold & top) / len(gold))
                        res[(name, g)][f"full@{k}"].append(float(gold <= top))
            # modality gap: raw cosine of query to image-bearing vs text-only rounds under the img/inter keys
            for k in ["img", "inter", "cap:dense", "text"]:
                s = S[k]
                g_img = [s[g] for g in gold if has_img[g]]
                g_txt = [s[g] for g in gold if not has_img[g]]
                gap[(k, "gold_img")] += g_img
                gap[(k, "gold_txt")] += g_txt
                gap[(k, "nongold_img")] += [s[i] for i in range(len(s)) if has_img[i] and i not in gold]
                gap[(k, "nongold_txt")] += [s[i] for i in range(len(s)) if not has_img[i] and i not in gold]
    out = {f"{n}||{g}": {m: round(float(np.mean(v)), 3) for m, v in d.items()} | {"n": len(d["cov@1"])}
           for (n, g), d in res.items()}
    out["_gap"] = {f"{k}||{w}": (round(float(np.mean(v)), 3) if v else None, len(v)) for (k, w), v in gap.items()}
    json.dump(out, open(ROOT / f"results/mm/retrieval_{args.bench}_{args.split}.json", "w"), indent=1)
    json.dump(perq, open(ROOT / f"results/mm/retrieval_{args.bench}_{args.split}_perq_cov10.json", "w"))
    rng = np.random.default_rng(0)
    for a, b in [("cap:dense", "inter"), ("inter", "img"), ("cap:dense", "img"), ("cap:dense", "text")]:
        q = sorted(perq[a])
        d = np.array([perq[a][x] - perq[b][x] for x in q])
        bs = rng.choice(d, size=(10000, len(d))).mean(1)
        print(f"  cov@10 {a} - {b}: {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] n={len(d)}")
    names = sorted({n for n, _ in res})
    for g in ["ALL", "Y1", "Y2", "Y3", "gold_has_img", "gold_text_only"]:
        print(f"--- {g}")
        for n in names:
            d = out.get(f"{n}||{g}")
            if d:
                print(f"  {n:24s} n={d['n']:4d} " + " ".join(f"cov@{k}={d[f'cov@{k}']:.3f}" for k in KS) +
                      f" | full@10={d['full@10']:.3f}")
    print("--- raw cosine (modality gap)")
    for kk, v in sorted(out["_gap"].items()):
        print(f"  {kk:30s} mean={v[0]} n={v[1]}")


if __name__ == "__main__":
    main()
