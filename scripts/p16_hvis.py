"""Phase 16: history-level index visibility (question-independent, computable at write time). No API calls (cache-only).
For every history: the share of each round's content words (user message, assistant reply, image descriptions) that its
compact index line shows, averaged over the rounds; with 200-character lines (standard index) and 600-character lines
(widened index); plus the mean user-message length. Stored per question (each question inherits its history's values).
python scripts/p16_hvis.py memeye memlens memlens64 memlens_dev visualmem -> results/p16/hvis_<bench>.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "results/p16"))
from load import captions, load  # noqa: E402  (cache-only loader: the API client is disabled)
import mm2  # noqa: E402

NAMES = {"memeye": "memeye", "memlens": "memlens_test", "memlens64": "memlens_test64", "memlens_dev": "memlens_dev",
         "visualmem": "visualmem", "camroll": "camroll"}


def history_stats(e, caps):
    out = {}
    for snip in (200, 600):
        mm2.SNIPPET = snip
        vals = []
        for r in e.rounds:
            c = {p: caps.get(p, "") for p in r.images}
            full = set(mm2._words(r.user + " " + (r.assistant or "") + " " + " ".join(c.values())))
            if full:
                vals.append(len(full & set(mm2._words(mm2.toc_line(r, c, e)))) / len(full))
        out[f"H{snip}"] = float(np.mean(vals)) if vals else 1.0
    out["ulen"] = float(np.mean([len(r.user) for r in e.rounds]))
    out["rounds"] = len(e.rounds)
    return out


def main():
    mm2.NO_ASSIST = mm2.SHORT_IDS = mm2.TELEGRAPH = True
    try:
        for bench in sys.argv[1:]:
            eps = load(NAMES[bench])
            caps = captions(eps)[0]
            out, per_hist = {}, []
            for e in eps:
                s = history_stats(e, caps)
                per_hist.append(s)
                for qa in e.qas:
                    out[qa.qid] = dict(s, hist=e.name)
            json.dump(out, open(ROOT / f"results/p16/hvis_{bench}.json", "w"))
            h = np.array([s["H200"] for s in per_hist]); h6 = np.array([s["H600"] for s in per_hist]); u = np.array([s["ulen"] for s in per_hist])
            print(f"{bench:12s} histories {len(per_hist):3d} | H200 mean {h.mean():.3f} min {h.min():.3f} max {h.max():.3f} | "
                  f"H600 mean {h6.mean():.3f} min {h6.min():.3f} max {h6.max():.3f} | user chars mean {u.mean():.0f} min {u.min():.0f} max {u.max():.0f}")
    finally:
        mm2.NO_ASSIST = mm2.SHORT_IDS = mm2.TELEGRAPH = False
        mm2.SNIPPET = 200


if __name__ == "__main__":
    main()
