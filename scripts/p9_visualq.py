"""Phase 9: classify each question, from its text alone, as visual (answering needs what an image shared in the
conversation shows) or not, with the cheap model; then compare set selection with RAG@40 on both groups and evaluate
visual routing (visual questions -> set path, others -> RAG@40) from stored outputs.
python scripts/p9_visualq.py > results/p9/visual_routing.md
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm2  # noqa: E402
from analyze import boot_ci  # noqa: E402
from gem import generate, pmap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p9/visualq.json"
SYS = ("You classify questions that a user asks an assistant about their own past conversations, in which the user "
       "sometimes shared photos, screenshots, or other images. Decide whether answering requires information that is "
       "shown in such an image (what it depicts, a detail in it, which image it is, how something looked), or whether "
       "the conversation text alone could contain the answer. Return JSON {\"visual\": true|false}.")
_SCHEMA = {"type": "object", "properties": {"visual": {"type": "boolean"}}, "required": ["visual"]}


def classify(q: str) -> bool:
    q = q.split("\n", 1)[1] if q.startswith("Question date:") else q
    r = generate([f"Question: {q}"], model="gemini-2.5-flash-lite", system=SYS, thinking=0, json_schema=_SCHEMA,
                 tag="p9:visualq", max_output_tokens=50)
    return bool(mm2.parse_json(r["text"]).get("visual"))


def L(p):
    if p.startswith("P2:"):
        from analyze_p2 import load_all
        return load_all("gemini-2.5-flash-lite")[p[3:]]
    return {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / p)}


def main():
    M = "results/mm2/"
    V = json.load(open(ROOT / "results/p5/visibility_hat.json"))
    test = set(json.load(open(ROOT / "results/memlens_test.json")))
    ml_q2 = L(M + "memlens_32k/toc38-40__read-medium__gemini-2.5-flash-lite.jsonl")
    ml_qn = L(M + "memlens_32k_memlens_test/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl")
    ml_vgs = {q: (ml_q2[q] if V[q] >= 0.7 else ml_qn[q]) for q in test if q in ml_q2 and q in ml_qn}
    wv = json.load(open(ROOT / "results/p8/wma_visibility_hat.json"))
    w_q2 = L(M + "wma_all_wma_personal_qids/toc38-40__read-medium__gemini-2.5-flash-lite__xcache_imgid.jsonl")
    w_qn = {**L(M + "wma_all_wma_dev_qids/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600__xcache_imgid.jsonl"),
            **L(M + "wma_all_wma_test_qids/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600__xcache_imgid.jsonl")}
    w_vgs = {q: (w_q2[q] if wv[q] >= 0.7 else w_qn[q]) for q in w_q2 if q in w_qn}
    BENCH = {
        "MemEye (QSS = VGS path)": (L("P2:rag:40|read:medium"), L("P2:toc38:40|read:medium")),
        "MemLens test (VGS)": (L(M + "memlens_32k/rag-40__read-medium__gemini-2.5-flash-lite.jsonl"), ml_vgs),
        "Mem-Gallery (QSS)": (L(M + "memgal_all/rag-40__read-medium__gemini-2.5-flash-lite.jsonl"), L(M + "memgal_all/toc38-40__read-medium__gemini-2.5-flash-lite.jsonl")),
        "WorldMemArena (VGS)": (L(M + "wma_all_wma_personal_qids/rag-40__read-medium__gemini-2.5-flash-lite_imgid.jsonl"), w_vgs),
    }
    lab = json.load(open(OUT)) if OUT.exists() else {}
    todo = {q: r["question"] for R, S in BENCH.values() for q, r in S.items() if q not in lab and q in R}
    if todo:
        ks = list(todo)
        for q, v in zip(ks, pmap(lambda k: classify(todo[k]), ks, workers=16, desc="visualq")):
            lab[q] = v
        json.dump(lab, open(OUT, "w"))
    print("| benchmark | share visual | visual: RAG@40 -> set path | text: RAG@40 -> set path | routed (visual -> set path) vs RAG@40 | $/1k RAG / set / routed |")
    print("|---|---:|---|---|---|---|")
    for name, (R, S) in BENCH.items():
        qs = sorted(q for q in S if q in R)
        vis = [q for q in qs if lab[q]]; txt = [q for q in qs if not lab[q]]
        def cmp(sub):
            if not sub:
                return "--"
            d = np.array([S[q]["score"] - R[q]["score"] for q in sub]); lo, hi = boot_ci(d)
            return f"{np.mean([R[q]['score'] for q in sub]):.3f} -> {np.mean([S[q]['score'] for q in sub]):.3f} ({d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], n={len(sub)})"
        rt = {q: (S[q] if lab[q] else R[q]) for q in qs}
        d = np.array([rt[q]["score"] - R[q]["score"] for q in qs]); lo, hi = boot_ci(d)
        cost = lambda X: 1000 * np.mean([X[q].get("usd", 0) for q in qs])
        print(f"| {name} | {len(vis) / len(qs):.2f} | {cmp(vis)} | {cmp(txt)} | {np.mean([rt[q]['score'] for q in qs]):.3f} ({d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]) | "
              f"{cost(R):.2f} / {cost(S):.2f} / {cost(rt):.2f} |")


if __name__ == "__main__":
    main()
