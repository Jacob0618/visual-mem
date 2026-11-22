"""Phase 14: id lists of the ablation controls on the re-runs (MemEye, VisualMem), paired with QSS over the same
100-round candidate pool: the model ranking of the pool (QSS instruction) cut at 8 and at the size that QSS chose, and
cosine top-k of the same size. No API calls.  python scripts/p14_ids_v2.py memeye visualmem"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/p14/runs"


def main():
    for bench in sys.argv[1:] or ["memeye", "visualmem"]:
        sel = ROOT / ("results/mm2/select_eval_rankr38-100.json" if bench == "memeye" else f"results/mm2/select_eval_{bench}_rankr38-100.json")
        rank = {r["qid"]: r["ids"] for r in json.load(open(sel))["per_q"]}
        qss = {json.loads(l)["qid"]: json.loads(l) for l in open(R / bench / "ptoc38-100-40__read-medium__gemini-2.5-flash-lite.jsonl")}
        rag = {json.loads(l)["qid"]: json.loads(l) for l in open(R / bench / "rag-40__read-medium__gemini-2.5-flash-lite.jsonl")}
        qs = sorted(q for q in qss if q in rank and q in rag and not qss[q].get("error"))
        size = {q: max(1, len(qss[q]["retrieved"])) for q in qs}
        out = ROOT / "results/p14/ids"
        out.mkdir(parents=True, exist_ok=True)
        json.dump({q: rank[q][:8] for q in qs}, open(out / f"{bench}_rank8.json", "w"))
        json.dump({q: rank[q][:size[q]] for q in qs}, open(out / f"{bench}_rankqss.json", "w"))
        json.dump({q: rag[q]["retrieved"][:size[q]] for q in qs}, open(out / f"{bench}_cosqss.json", "w"))
        over = sum(size[q] > 40 for q in qs)
        print(f"{bench}: {len(qs)} questions, mean QSS size {sum(size.values()) / len(qs):.2f}, sizes above 40: {over}")


if __name__ == "__main__":
    main()
