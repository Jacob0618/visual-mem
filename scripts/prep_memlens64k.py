"""Prepare MemLens-64k for the test split (captions + round embeddings), so later runs only pay for reading."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mm  # noqa: E402
from data_memlens import load_memlens  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
test = json.load(open(ROOT / "results/memlens_test.json"))
eps = load_memlens("64k", test)
print("episodes", len(eps), "rounds/q", sum(len(e.rounds) for e in eps) / len(eps), "images/q", sum(e.n_images for e in eps) / len(eps), flush=True)
caps = {"short": mm.build_captions(eps, "short")}; caps["dense"] = caps["short"]
for i, e in enumerate(eps):
    mm.RoundIndex(e, caps, ["cap:short"])
    if i % 50 == 0:
        print("indexed", i, flush=True)
print("done")
