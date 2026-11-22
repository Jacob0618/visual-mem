"""Phase 16 gate analyses (no API calls): history-level visibility (question-independent), calibration of v_hat, and a
learned router (logistic regression, numpy) trained on development data (MemEye + MemLens dev) and tested on held-out
data (MemLens test, VisualMem).  python scripts/p16_gate2.py -> results/p16/gate2.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "results/p16"))
import p16_compose as P  # noqa: E402
from load import captions, load  # noqa: E402  (cache-only)
import mm2  # noqa: E402

E = ROOT / "results/p16/explore"


def L(p):
    return {json.loads(x)["qid"]: json.loads(x) for x in open(p)}


def history_visibility(bench_name):
    """qid -> share of each round's content words that its compact index line shows, averaged over the history."""
    eps = load(bench_name)
    caps = captions(eps)[0]
    mm2.NO_ASSIST = mm2.SHORT_IDS = mm2.TELEGRAPH = True
    out = {}
    try:
        for e in eps:
            vals = []
            for r in e.rounds:
                full = set(mm2._words(r.user + " " + (r.assistant or "") + " " + " ".join(caps.get(p, "") for p in r.images)))
                if full:
                    line = set(mm2._words(mm2.toc_line(r, {p: caps.get(p, "") for p in r.images}, e)))
                    vals.append(len(full & line) / len(full))
            h = float(np.mean(vals)) if vals else 1.0
            for qa in e.qas:
                out[qa.qid] = h
    finally:
        mm2.NO_ASSIST = mm2.SHORT_IDS = mm2.TELEGRAPH = False
    return out


def feats(q, V, H, R):
    return [V.get(q, 1.0), H.get(q, 1.0), R[q]["top10_user_len"] / 1000.0 if q in R else 0.2]


def logreg(X, y, l2=1e-2, it=3000, lr=0.5):
    X = np.c_[np.ones(len(X)), (X - X.mean(0)) / (X.std(0) + 1e-9)]
    mu, sd = None, None
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]])
    return w


def main():
    res = {}
    data = {}
    for bench, name in [("memeye", "memeye"), ("memlens", "memlens_test"), ("visualmem", "visualmem")]:
        out, V = P.compose(bench)
        H = history_visibility(name)
        R = json.load(open(ROOT / f"results/p16/ranks_{bench}.json"))
        data[bench] = (out["std_l"], out["wid_l"], V, H, R)
        hv = np.array(list(H.values()))
        res[f"{bench}_history_visibility_mean"] = float(hv.mean())
    # MemLens dev for training the router
    Vd = json.load(open(ROOT / "results/p5/visibility_hat.json"))
    std_d = L(E / "memlens/toc38-40__read-medium__gemini-2.5-flash-lite__cnoassist+shortid+telegraph.jsonl")
    wid_d = L(E / "memlens/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600__cnoassist+shortid+telegraph.jsonl")
    Hd = history_visibility("memlens_dev")
    Rd = {}
    # calibration: mean (widened - standard) by v_hat quintile, per benchmark (lightweight reader)
    for bench, (std, wid, V, H, R) in data.items():
        qs = sorted(set(std) & set(wid))
        v = np.array([V.get(q, 1.0) for q in qs]); d = np.array([wid[q]["score"] - std[q]["score"] for q in qs])
        edges = np.quantile(v, [0, .2, .4, .6, .8, 1.0])
        res[f"{bench}_calibration"] = [[float(edges[i]), float(edges[i + 1]), float(d[(v >= edges[i]) & (v <= edges[i + 1])].mean())] for i in range(5)]
        # history-level routing: whole history to the standard path when its index keeps >= tau_h of the round words
        for tau_h in (0.5, 0.6, 0.7):
            acc = np.mean([(std if H.get(q, 1.0) >= tau_h else wid)[q]["score"] for q in qs])
            res[f"{bench}_history_gate_{tau_h}"] = float(acc)
        res[f"{bench}_question_gate"] = float(np.mean([(std if V.get(q, 1.0) >= P.TAU else wid)[q]["score"] for q in qs]))
        res[f"{bench}_std"] = float(np.mean([std[q]["score"] for q in qs])); res[f"{bench}_wid"] = float(np.mean([wid[q]["score"] for q in qs]))
    # learned router: train on MemEye + MemLens dev, test on MemLens test and VisualMem
    Xtr, ytr = [], []
    std, wid, V, H, R = data["memeye"]
    for q in set(std) & set(wid):
        if std[q]["score"] != wid[q]["score"]:
            Xtr.append(feats(q, V, H, R)); ytr.append(float(wid[q]["score"] > std[q]["score"]))
    for q in set(std_d) & set(wid_d):
        if std_d[q]["score"] != wid_d[q]["score"]:
            Xtr.append([Vd.get(q, 1.0), Hd.get(q, 1.0), 0.2]); ytr.append(float(wid_d[q]["score"] > std_d[q]["score"]))
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    w = logreg(Xtr, ytr)
    for bench in ("memlens", "visualmem"):
        std, wid, V, H, R = data[bench]
        qs = sorted(set(std) & set(wid))
        X = np.array([feats(q, V, H, R) for q in qs]); Xn = np.c_[np.ones(len(X)), (X - mu) / sd]
        p = 1 / (1 + np.exp(-Xn @ w))
        res[f"{bench}_learned_router"] = float(np.mean([(wid if pi >= 0.5 else std)[q]["score"] for q, pi in zip(qs, p)]))
        res[f"{bench}_learned_share_std"] = float(np.mean(p < 0.5))
    json.dump(res, open(ROOT / "results/p16/gate2.json", "w"), indent=1)
    for k, v in res.items():
        print(k, v if not isinstance(v, float) else round(v, 3))


if __name__ == "__main__":
    main()
