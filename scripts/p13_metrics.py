"""Phase 13: standard metrics for every method on the question sets of Table 2. No API calls.
Answer level (open-ended benchmarks MemEye and MemLens): SQuAD-style token F1 and sentence BLEU-1 against the gold
answer. VisualMem is multiple choice, so its only answer metric is exact match, which is its accuracy in Table 2.
Evidence level (methods that hand raw rounds to the reader): Recall@10 and nDCG@10 (binary relevance) of the returned
round list against the gold evidence rounds; text-memory systems return derived records, not rounds, so these metrics
do not apply to them. VGS uses the rounds of its lightweight-reader path (standard or widened selection per question).
python scripts/p13_metrics.py  -> results/p13/standard_metrics.json"""
import ast
import json
import math
import re
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from p11_analyze import bl, ours, TAU, L, M  # noqa: E402

SYSTEMS = ["rag", "simplemem", "omni", "mirix", "vmem", "mma", "telemem", "vgs"]
ROUND_SYSTEMS = {"rag", "vmem", "mma", "vgs"}
OUT = ROOT / "results/p13"


def toks(s):
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def f1(pred, gold):
    p, g = toks(pred), toks(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or common == 0:
        return 0.0
    P, R = common / len(p), common / len(g)
    return 2 * P * R / (P + R)


def bleu1(pred, gold):
    p, g = toks(pred), toks(gold)
    if not p or not g:
        return 0.0
    cg = Counter(g)
    prec = sum(min(c, cg[w]) for w, c in Counter(p).items()) / len(p)
    bp = 1.0 if len(p) > len(g) else math.exp(1 - len(g) / len(p))
    return bp * prec


def rank_metrics(ranked, gold, k=10):
    gold = set(gold)
    if not gold:
        return None
    top = list(dict.fromkeys(ranked))[:k]
    hits = [1.0 if r in gold else 0.0 for r in top]
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return {"r10": sum(hits) / len(gold), "ndcg10": dcg / idcg}


def ids(rec):
    r = rec.get("retrieved")
    return ast.literal_eval(r) if isinstance(r, str) else list(r or [])


def gold_rounds(bench):
    """qid -> gold evidence round ids, from the benchmark loaders (no API calls)."""
    from run_mm import episodes
    if bench == "memeye":
        eps = episodes("memeye", "all")
    elif bench == "memlens":
        eps = episodes("memlens", "32k", sorted(json.load(open(ROOT / "results/memlens_test.json"))))
    else:
        eps = episodes("visualmem", "all")
    return {qa.qid: list(qa.clue) for e in eps for qa in e.qas}


def vgs_lite_rounds(bench):
    """Rounds handed to the lightweight reader by VGS: standard-path selection if v_hat >= tau, else widened selection."""
    if bench == "memeye":
        from analyze_p2 import load_all
        V = json.load(open(ROOT / "results/p5/visibility_hat.json"))
        std = load_all("gemini-2.5-flash-lite")["toc38:40|read:medium"]
        wid = L(M / "memeye_all/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl")
    elif bench == "memlens":
        V = json.load(open(ROOT / "results/p5/visibility_hat.json"))
        std = L(M / "memlens_32k/toc38-40__read-medium__gemini-2.5-flash-lite.jsonl")
        wid = L(M / "memlens_32k_memlens_test/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl")
    else:
        V = json.load(open(ROOT / "results/p10/vm_visibility_hat.json"))
        std = L(M / "visualmem_all/toc38-40__read-medium__gemini-2.5-flash-lite__xcache.jsonl")
        wid = L(M / "visualmem_all_vm_lovis/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600__xcache.jsonl")
    out = {}
    for q in set(std) | set(wid):
        src = std if V.get(q, 0) >= TAU else wid
        if q in src:
            out[q] = ids(src[q])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for bench in ["memeye", "memlens", "visualmem"]:
        G = gold_rounds(bench)
        O = ours(bench)
        R40 = O["lite"][0]
        gold_ans = {q: r["gold"] for q, r in R40.items()}
        retrieved = {"rag": {q: ids(r) for q, r in R40.items()}, "vgs": vgs_lite_rounds(bench)}
        for s in ["vmem", "mma"]:        # re-run retrieval (P13_RETRIEVAL_ONLY=1 scripts/p12_vmem.py / p12_mma.py)
            p = OUT / "retrieval" / bench / f"{s}.jsonl"
            if p.exists():
                retrieved[s] = {r["qid"]: r["retrieved"] for r in map(json.loads, open(p)) if "retrieved" in r}
        for reader in ["lite", "strong"]:
            base, vgs = O[reader]
            qs = sorted(q for q in base if q in vgs)                 # question set of Table 2
            if bench == "memlens":
                qs = sorted(set(qs) & set(json.load(open(ROOT / "results/p11_memlens_sub150.json"))))
            for s in SYSTEMS:
                X = base if s == "rag" else vgs if s == "vgs" else bl(bench, s, reader)
                q2 = [q for q in qs if q in X and X[q].get("score") is not None]
                if not q2:
                    continue
                d = {"n": len(q2), "acc": float(np.mean([X[q]["score"] for q in q2]))}
                if bench != "visualmem":
                    d["f1"] = float(np.mean([f1(X[q]["pred"], gold_ans[q]) for q in q2]))
                    d["bleu1"] = float(np.mean([bleu1(X[q]["pred"], gold_ans[q]) for q in q2]))
                if s in retrieved and reader == "lite":
                    M_ = [rank_metrics(retrieved[s][q], G.get(q, [])) for q in q2 if q in retrieved[s]]
                    M_ = [m for m in M_ if m]
                    if M_:
                        d["r10"] = float(np.mean([m["r10"] for m in M_]))
                        d["ndcg10"] = float(np.mean([m["ndcg10"] for m in M_]))
                        d["n_ret"] = len(M_)
                res[f"{bench}|{reader}|{s}"] = d
                print(f"{bench:9s} {reader:6s} {s:9s} " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                                                for k, v in d.items()))
    json.dump(res, open(OUT / "standard_metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
