"""Phase 5: recall of the per-unit verify+extract step on gold units (MemLens dev sample). Every gold round is evidence
by construction, so recall = share of gold units marked relevant.  python scripts/map_probe.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mm  # noqa: E402
import mm2  # noqa: E402
from data_memlens import load_memlens  # noqa: E402
from gem import pmap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LENIENT = (
    "\nReturn JSON {{\"relevant\": true|false, \"facts\": [\"<round id>: <one concrete fact>\", ...]}}.\n"
    "Most of this conversation may be about another topic: the evidence is often a brief aside in one sentence, or a "
    "photo, receipt, or screenshot. Set relevant=true if ANY sentence or image mentions, shows, or implies an item, event, "
    "amount, date, or state that the question asks about, even indirectly or partially (one of several items is enough). "
    "Each fact must be self-contained (what, which item, exact value, when) and cite its round id; describe what any "
    "related image shows. If nothing at all relates to the question, return relevant=false and an empty list.")


def main():
    dev = json.load(open(ROOT / "results/memlens_dev.json"))
    raw = {x["question_id"]: x for x in json.load(open(ROOT / "data/MemLens/dataset_32k.json"))}
    by = {}
    for q in dev:
        by.setdefault(raw[q]["question_type"], []).append(q)
    rng = random.Random(0)
    sample = sorted(sum((rng.sample(v, min(10, len(v))) for v in by.values()), []))
    eps = load_memlens("32k", sample)
    caps = {"short": mm.build_captions(eps, "short")}; caps["dense"] = caps["short"]
    jobs = [(e, e.qas[0], [e.rid2idx[c]]) for e in eps for c in e.qas[0].clue] + \
           [(e, e.qas[0], u) for e in eps for u in mm2.session_units(e, [e.rid2idx[c] for c in e.qas[0].clue])]
    variants = {"V0 2.5": (None, 0, None), "V1 2.5 + thinking 512": (None, 512, None), "V2 2.5 lenient prompt": (None, 0, LENIENT),
                "V3 3.8-flash": ("gemini-3.8-flash", "low", None)}
    for name, (model, th, instr) in variants.items():
        old = mm2.EVA_MAP_INSTR
        if instr:
            mm2.EVA_MAP_INSTR = instr
        out = pmap(lambda j: (len(j[2]), mm2.eva_map(j[0], j[1], j[2], "medium", caps, model=model, thinking=th, tag="p5:mapprobe")[0]),
                   jobs, workers=12, desc=name)
        mm2.EVA_MAP_INSTR = old
        rr = [ok for n, ok in out if n == 1]; rs = [ok for n, ok in out if n > 1]
        print(f"{name:24s} round units: recall {np.mean(rr):.3f} (n={len(rr)}) | session units: recall {np.mean(rs):.3f} (n={len(rs)})", flush=True)


if __name__ == "__main__":
    main()
