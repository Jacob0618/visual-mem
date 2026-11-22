"""Phase 5: visibility-gated combination of per-question results (no API calls).
Gate: query-time visibility estimate v_hat (results/p5/visibility_hat.json) >= tau -> high path, else low path.
python scripts/gate_eval.py --bench memlens --split dev
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import boot_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def L(p):
    p = ROOT / p
    if not p.exists():
        return {}
    D = {json.loads(l)["qid"]: json.loads(l) for l in open(p)}
    bad = [q for q, r in D.items() if r.get("error")]
    if bad:
        raise SystemExit(f"{p}: {len(bad)} failed records (e.g. {D[bad[0]]['error'][:80]}); re-run before analysis")
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="memlens")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--reader", default="gemini-2.5-flash-lite")
    a = ap.parse_args()
    V = json.load(open(ROOT / "results/p5/visibility_hat.json"))
    rd = a.reader
    if a.bench == "memlens" and rd == "gemini-3.8-flash":
        keep = set(json.load(open(ROOT / f"results/memlens_{a.split}.json")))
        sub = f"results/mm2/memlens_32k_memlens_{a.split}/"
        f = lambda n: {q: r for q, r in L(sub + n).items() if q in keep}
        R = {"RAG@40": f(f"rag-40__read-low__{rd}.jsonl"), "QSS (200)": f(f"toc38-40__read-low__{rd}.jsonl"),
             "QSS-long (600)": f(f"toc38-40__read-low__{rd}__snip600.jsonl")}
    elif a.bench == "memlens":
        keep = set(json.load(open(ROOT / f"results/memlens_{a.split}.json")))
        sub = f"results/mm2/memlens_32k_memlens_{a.split}/"
        base = "results/mm2/memlens_32k/"
        f = lambda d, n: {q: r for q, r in L(d + n).items() if q in keep}
        R = {"RAG@40": f(base, f"rag-40__read-medium__{rd}.jsonl"),
             "RAG@40 + thinking": {**f(base, f"rag-40__read-medium__{rd}__think1024.jsonl"), **f(sub, f"rag-40__read-medium__{rd}__think1024.jsonl")},
             "QSS (200)": f(base, f"toc38-40__read-medium__{rd}.jsonl"),
             "QSS-long (600)": f(base, f"toc38-40__read-medium__{rd}__snip600.jsonl"),
             "QSS-long + notes": f(sub, f"toc38-40__readn-medium__{rd}__snip600.jsonl"),
             "RAG@20 + notes": f(sub, f"rag-20__readn-medium__{rd}.jsonl")}
    else:
        base = "results/mm2/memeye_all/"
        R = {"RAG@40": {}, "QSS (200)": L(base + f"toc38-40__read-medium__{rd}.jsonl"),
             "QSS-long (600)": L(base + f"toc38-40__read-medium__{rd}__snip600.jsonl"),
             "QSS-long + notes": L(base + f"toc38-40__readn-medium__{rd}__snip600.jsonl"),
             "RAG@20 + notes": L(base + f"rag-20__readn-medium__{rd}.jsonl")}
        sys.path.insert(0, str(ROOT / "scripts"))
        from analyze_p2 import load_all
        D = load_all(rd)
        R["RAG@40"] = D["rag:40|read:medium"]
        R["RAG@40 + thinking"] = D.get("rag:40|read:medium +think1024", {})
    R = {k: v for k, v in R.items() if v}
    hi = lambda q: V.get(q, 1.0) >= a.tau

    def gate(high, low):
        if high not in R or low not in R:
            return {}
        qs = set(R[high]) & set(R[low])
        return {q: (R[high][q] if hi(q) else R[low][q]) for q in qs}
    R["VGS = gate(QSS | QSS-long + notes)"] = gate("QSS (200)", "QSS-long + notes")
    R["VGS-cheap = gate(QSS | RAG@20 + notes)"] = gate("QSS (200)", "RAG@20 + notes")
    R["VGS-strong = gate(QSS | QSS-long)"] = gate("QSS (200)", "QSS-long (600)")
    R = {k: v for k, v in R.items() if v}
    ref = R["RAG@40"]
    n_hi = np.mean([hi(q) for q in ref])
    print(f"### {a.bench} {a.split if a.bench == 'memlens' else 'all'} | reader {rd} | gate tau={a.tau}: share of questions on the high-visibility path {n_hi:.3f}\n")
    print("| method | n | accuracy | Δ vs RAG@40 [95% CI] | $/1k |")
    print("|---|---:|---:|---|---:|")
    for k, v in R.items():
        q = sorted(set(v) & set(ref))
        if not q:
            continue
        s = np.array([v[x]["score"] for x in q]); d = s - np.array([ref[x]["score"] for x in q]); lo, hi_ = boot_ci(d)
        print(f"| {k} | {len(q)} | {s.mean():.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi_:+.3f}] | {1000 * np.mean([v[x].get('usd', 0) for x in q]):.2f} |")


if __name__ == "__main__":
    main()
