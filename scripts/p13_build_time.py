"""Phase 13: construction time of the RAG@40 / VGS memory on the latency samples, with the local response cache
bypassed: one short caption per image (gemini-3.8-flash, 24 parallel requests as in all runs), one embedding per
round (gemini-embedding-2), and, for persistent histories, the creation of the explicit context cache of the index.
Costs about $0.5.  python scripts/p13_build_time.py -> results/p13/build_time_ours.json"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import gem  # noqa: E402
import mm  # noqa: E402
import mm2  # noqa: E402
from run_mm import episodes  # noqa: E402

gem.FRESH_TAGS = ["caption:"]
gem._emb_cache.get_many = lambda keys: {}          # embeddings: bypass the local cache as well


def main():
    out = {}
    samples = {"memeye": ("all", None, json.load(open(ROOT / "results/p13/latency_qids_memeye.json"))),
               "memlens": ("32k", json.load(open(ROOT / "results/p13/latency_qids_memlens.json")), None),
               "visualmem": ("all", json.load(open(ROOT / "results/p13/latency_qids_visualmem.json")), None)}
    for bench, (split, tasks, keep) in samples.items():
        eps = episodes(bench, split, tasks)
        if keep:
            eps = [e for e in eps if any(q.qid in set(keep) for q in e.qas)]
        times = []
        for e in eps:
            t0 = time.perf_counter()
            caps = mm.build_captions([e], "short")
            t1 = time.perf_counter()
            mm.RoundIndex(e, {"short": caps, "dense": caps}, ["cap:short"])
            t2 = time.perf_counter()
            times.append({"history": e.name, "rounds": len(e.rounds), "images": sum(len(r.images) for r in e.rounds),
                          "caption_s": round(t1 - t0, 2), "embed_s": round(t2 - t1, 2)})
            print(bench, times[-1], flush=True)
        out[bench] = times
    json.dump(out, open(ROOT / "results/p13/build_time_ours.json", "w"), indent=1)


if __name__ == "__main__":
    main()
