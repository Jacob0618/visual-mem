"""Phase 13: MMA memory construction time on the latency samples with the embedding cache bypassed (its latency runs
reused image embeddings that earlier retrieval re-runs had cached). Costs a few cents.
.venv_bl/bin/python scripts/p13_build_time_mma.py -> results/p13/build_time_mma.json"""
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["P13_LATENCY"] = "1"
import p12_common as C  # noqa: E402
import p12_mma as M  # noqa: E402


def main():
    M.setup()
    import yaml
    import gem
    from benchmark.mma.system import MMASystem
    gem._emb_cache.get_many = lambda keys: {}
    cfg = yaml.safe_load(open(M.HARNESS / "config/methods/mma_gemini.yaml"))
    os.environ.setdefault("GEMINI_API_KEY", C.gemini_key())
    out = {}
    for bench in ["memeye", "memlens", "visualmem"]:
        os.environ["P13_QIDS"] = str(ROOT / f"results/p13/latency_qids_{bench}.json")
        times = []
        for ep in C.load_bench(bench):
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                MMASystem(cfg).process_all_sessions(M.EpisodeDataset(ep))
            times.append({"history": ep.name, "build_s": round(time.perf_counter() - t0, 2)})
            print(bench, times[-1], flush=True)
        out[bench] = times
    json.dump(out, open(ROOT / "results/p13/build_time_mma.json", "w"), indent=1)


if __name__ == "__main__":
    main()
