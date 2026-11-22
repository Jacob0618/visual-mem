"""Phase 8 (review response), no API calls.
(1) The deployable visibility estimate v_hat against oracle index visibility (MemLens-32k, 699 questions).
(2) The VGS gate per path on the registered MemLens test part (466 questions, lightweight reader).
(3) Full-set recall under alternative gold definitions (MemLens-32k), and an audit of the strong-reader cases in which
    top-40 retrieval is right and the annotated evidence is wrong (200-question subset).
python scripts/p8_offline.py > results/p8_offline.md
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm  # noqa: E402
import mm2  # noqa: E402
from analyze import boot_ci  # noqa: E402
from data_memlens import _raw, load_memlens  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/mm2"


def L(p):
    R = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("error"):
            raise SystemExit(f"{p}: failed records")
        R[r["qid"]] = r
    return R


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def oracle_visibility(e, qa, caps):
    q = set(mm2._words(qa.question.split("\n", 1)[-1]))
    vals = []
    for rid in qa.clue:
        r = e.rounds[e.rid2idx[rid]]
        full = set(mm2._words(r.user + " " + r.assistant + " " + " ".join(caps.get(p, "") for p in r.images)))
        line = set(mm2._words(mm2.toc_line(r, caps, e)))
        m = q & full
        if m:
            vals.append(len(m & line) / len(m))
    return float(np.mean(vals)) if vals else None


def gold_variants():
    """Per question: rounds with an evidence-flagged turn, rounds with a needle image, and the answer sessions."""
    out = {}
    for it in _raw("32k"):
        if it["question_type"] == "answer_refusal":
            continue
        ans = set(it["answer_session_ids"])
        text, img = set(), set()
        for sid, date, turns in zip(it["haystack_session_ids"], it["haystack_dates"], it["haystack_sessions"]):
            j, t = 0, 0
            while t < len(turns):
                if turns[t]["role"] != "user":
                    pair = [turns[t]]
                    t += 1
                else:
                    pair = [turns[t]] + ([turns[t + 1]] if t + 1 < len(turns) and turns[t + 1]["role"] == "assistant" else [])
                    t += len(pair)
                j += 1
                if sid in ans:
                    rid = f"{sid}:R{j}"
                    if any(x.get("has_answer") for x in pair):
                        text.add(rid)
                    if any(im["file"].startswith("needle_images/") for x in pair for im in (x.get("images") or [])):
                        img.add(rid)
        out[it["question_id"]] = {"text": text, "img": img, "sess": ans}
    return out


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def main():
    eps = load_memlens("32k")
    E = {e.qas[0].qid: e for e in eps}
    caps = mm.build_captions(eps, "short")
    vhat = json.load(open(ROOT / "results/p5/visibility_hat.json"))
    test = set(json.load(open(ROOT / "results/memlens_test.json")))

    # (1) v_hat against oracle visibility
    ov = {q: oracle_visibility(E[q], E[q].qas[0], caps) for q in E}
    qs = [q for q in E if ov[q] is not None and q in vhat]
    a, b = np.array([vhat[q] for q in qs]), np.array([ov[q] for q in qs])
    print(f"### (1) Visibility estimate vs oracle index visibility (MemLens-32k, {len(qs)} questions)\n")
    print(f"mean oracle visibility {b.mean():.3f}; mean v_hat {a.mean():.3f}; Spearman rho {spearman(a, b):.3f}; "
          f"AUC of v_hat for oracle visibility >= 0.5: {auc(a, b >= 0.5):.3f}; share of v_hat >= 0.7: {np.mean(a >= 0.7):.3f}\n")

    # (2) gate per path on the registered test part
    R40 = L(D / "memlens_32k/rag-40__read-medium__gemini-2.5-flash-lite.jsonl")
    Q2 = L(D / "memlens_32k/toc38-40__read-medium__gemini-2.5-flash-lite.jsonl")
    Q6 = L(D / "memlens_32k/toc38-40__read-medium__gemini-2.5-flash-lite__snip600.jsonl")
    Q6N = L(D / "memlens_32k_memlens_test/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl")
    T = sorted(q for q in test if q in Q6N)
    V = {q: (Q2[q] if vhat[q] >= 0.7 else Q6N[q]) for q in T}
    print(f"### (2) VGS gate per path, MemLens test part ({len(T)} questions, lightweight reader)\n")
    print("| path (v_hat) | n | RAG@40 | QSS 200 | QSS 600 | QSS 600 + notes | VGS | VGS - RAG@40 [95% CI] |\n|---|---:|---:|---:|---:|---:|---:|---|")
    for name, sub in [("high (>= 0.7)", [q for q in T if vhat[q] >= 0.7]), ("low (< 0.7)", [q for q in T if vhat[q] < 0.7]), ("all", T)]:
        m = lambda Dd: np.mean([Dd[q]["score"] for q in sub])
        d = np.array([V[q]["score"] - R40[q]["score"] for q in sub]); lo, hi = boot_ci(d)
        print(f"| {name} | {len(sub)} | {m(R40):.3f} | {m(Q2):.3f} | {m(Q6):.3f} | {m(Q6N):.3f} | {m(V):.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    qq = sorted(T, key=lambda q: vhat[q])
    print("\nGain of the notes path over QSS 200 by v_hat quartile (test part):")
    for k, chunk in enumerate(np.array_split(np.array(qq), 4)):
        d = np.array([Q6N[q]["score"] - Q2[q]["score"] for q in chunk]); lo, hi = boot_ci(d)
        print(f"  Q{k + 1}: v_hat {vhat[chunk[0]]:.2f}-{vhat[chunk[-1]]:.2f}, n={len(chunk)}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print()

    # (3a) FSR under alternative gold definitions
    G = gold_variants()
    sel = {"QSS 200": json.load(open(D / "select_eval_memlens32k_toc38-40.json"))["per_q"],
           "QSS 600": json.load(open(D / "select_eval_memlens32k_toc38-40_snip600.json"))["per_q"]}
    print("### (3a) Full-set recall under alternative gold definitions (MemLens-32k, 699 questions)\n")
    print("| retriever | rounds | FSR, derived gold | FSR, evidence-turn rounds | FSR, needle-image rounds | answer sessions covered |\n|---|---:|---:|---:|---:|---:|")
    idx = {}
    for name, per in sel.items():
        P = {r["qid"]: r for r in per}
        rows = defaultdict(list)
        for q, r in P.items():
            e = E[q]
            S = {e.rounds[i].rid for i in r["ids"]} if r["ids"] and isinstance(r["ids"][0], int) else set(r["ids"])
            ix = idx.setdefault(q, mm.RoundIndex(e, {"short": caps, "dense": caps}, ["cap:short"]))
            C = {e.rounds[i].rid for i in ix.topk(e.qas[0], "cap:short", max(1, len(S)))}
            g = G[q]
            for tag, SS in [(name, S), (f"cosine, same size as {name}", C)]:
                rows[tag].append((len(SS), set(e.qas[0].clue) <= SS if e.qas[0].clue else None,
                                  g["text"] <= SS if g["text"] else None, g["img"] <= SS if g["img"] else None,
                                  all(any(x.startswith(s + ":") for x in SS) for s in g["sess"])))
        for tag, v in rows.items():
            f = lambda k: np.mean([x[k] for x in v if x[k] is not None])
            print(f"| {tag} | {np.mean([x[0] for x in v]):.1f} | {f(1):.3f} | {f(2):.3f} | {f(3):.3f} | {f(4):.3f} |")
    v = []
    for q, e in E.items():
        ix = idx.setdefault(q, mm.RoundIndex(e, {"short": caps, "dense": caps}, ["cap:short"]))
        SS = {e.rounds[i].rid for i in ix.topk(e.qas[0], "cap:short", 40)}
        g = G[q]
        v.append((40, set(e.qas[0].clue) <= SS if e.qas[0].clue else None, g["text"] <= SS if g["text"] else None,
                  g["img"] <= SS if g["img"] else None, all(any(x.startswith(s + ":") for x in SS) for s in g["sess"])))
    f = lambda k: np.mean([x[k] for x in v if x[k] is not None])
    print(f"| cosine top-40 | 40.0 | {f(1):.3f} | {f(2):.3f} | {f(3):.3f} | {f(4):.3f} |\n")

    # (3b) audit: strong reader, RAG@40 right and annotated evidence wrong
    S40 = L(D / "memlens_32k_memlens_sub200/rag-40__read-low__gemini-3.8-flash.jsonl")
    ORC = L(D / "memlens_32k_memlens_sub200/oracle__read-low__gemini-3.8-flash.jsonl")
    cases = [q for q in S40 if S40[q]["score"] >= 1 and ORC[q]["score"] < 1]
    rev = [q for q in S40 if S40[q]["score"] < 1 and ORC[q]["score"] >= 1]
    stats = defaultdict(int)
    for q in cases:
        e = E[q]; qa = e.qas[0]; gold = set(qa.clue)
        ans = [w for w in norm(qa.answer) if len(w) > 2 and w not in mm2._STOP]
        retrieved = set(S40[q]["retrieved"])
        def has(rids):
            txt = " ".join(norm(" ".join(e.rounds[e.rid2idx[r]].user + " " + e.rounds[e.rid2idx[r]].assistant for r in rids)))
            return bool(ans) and all(w in txt.split() for w in ans)
        stats["type:" + qa.x] += 1
        if has(gold):
            stats["answer words in gold rounds"] += 1
        if has(retrieved - gold):
            stats["answer words in retrieved non-gold rounds"] += 1
        if any(r.split(":")[0] in {g.split(":")[0] for g in gold} for r in retrieved - gold):
            stats["retrieved other rounds of an answer session"] += 1
        if "oracle abstained" not in stats:
            stats["oracle abstained"] += 0
        if re.search(r"not (stated|mentioned|specified|provided)|cannot|unable|no (information|record)", ORC[q]["pred"] or "", re.I):
            stats["oracle abstained"] += 1
    print(f"### (3b) Strong reader on the 200-question subset: RAG@40 right, annotated evidence wrong ({len(cases)} cases; reverse: {len(rev)})\n")
    for k, v in sorted(stats.items()):
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
