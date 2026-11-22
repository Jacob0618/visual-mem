"""Phase 16: id lists of the selection controls for the compact-index VGS, paired with QSS over the same 100-round
candidate pool (MemLens histories have at most 83 rounds, so the pool is the whole history): the model ranking of the pool
(QSS instruction) cut at 8 and at the size QSS chose, and cosine top-k of that size. No API calls.
python scripts/p16_ids.py memeye memlens visualmem"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "results/p16/final"
C = "cnoassist+shortid+telegraph"
SEL = {"memeye": f"select_eval_rankr38-100_{C}.json", "visualmem": f"select_eval_visualmem_rankr38-100_{C}.json",
       "memlens": f"select_eval_memlens32k_memlens_test_rankr38-100_{C}.json",
       "camroll": f"select_eval_camroll_rankr38-100_{C}.json"}
QSS = {"memeye": f"ptoc38-100-40__read-medium__gemini-2.5-flash-lite__{C}.jsonl",
       "visualmem": f"ptoc38-100-40__read-medium__gemini-2.5-flash-lite__{C}.jsonl",
       "memlens": f"toc38-40__read-medium__gemini-2.5-flash-lite__{C}.jsonl",
       "camroll": f"ptoc38-100-40__read-medium__gemini-2.5-flash-lite__{C}.jsonl"}
RAG = {"memeye": ROOT / "results/p14/runs/memeye/rag-40__read-medium__gemini-2.5-flash-lite.jsonl",
       "visualmem": ROOT / "results/p14/runs/visualmem/rag-40__read-medium__gemini-2.5-flash-lite.jsonl",
       "memlens": F / "memlens/rag-40__read-medium__gemini-2.5-flash-lite.jsonl",
       "camroll": ROOT / "results/p17/camroll/rag-40__read-medium__gemini-2.5-flash-lite.jsonl"}


def L(p):
    return {json.loads(x)["qid"]: json.loads(x) for x in open(p)}


def main():
    out = ROOT / "results/p16/ids"
    out.mkdir(parents=True, exist_ok=True)
    for bench in sys.argv[1:] or ["memeye", "memlens", "visualmem"]:
        rank = {r["qid"]: r["ids"] for r in json.load(open(ROOT / "results/mm2" / SEL[bench]))["per_q"]}
        qss, rag = L(F / bench / QSS[bench]), L(RAG[bench])
        qs = sorted(q for q in qss if q in rank and q in rag and not qss[q].get("error"))
        size = {q: max(1, len(qss[q]["retrieved"])) for q in qs}
        json.dump({q: rank[q][:8] for q in qs}, open(out / f"{bench}_rank8.json", "w"))
        json.dump({q: rank[q][:size[q]] for q in qs}, open(out / f"{bench}_rankqss.json", "w"))
        json.dump({q: rag[q]["retrieved"][:size[q]] for q in qs}, open(out / f"{bench}_cosqss.json", "w"))
        print(f"{bench}: {len(qs)} questions, mean QSS size {sum(size.values()) / len(qs):.2f}, above 40: {sum(size[q] > 40 for q in qs)}")


if __name__ == "__main__":
    main()
