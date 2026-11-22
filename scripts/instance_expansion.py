"""Phase 5: instance expansion for set questions. A cheap model rewrites the question into the concrete things the
user may have mentioned or shown (e.g. coffee makers -> espresso machine, moka pot, pour-over dripper ...); rounds are
scored by their best match over the question and the instances (round-level embeddings). Retrieval-only comparison.
python scripts/instance_expansion.py --bench memlens --split dev | --bench memeye
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm  # noqa: E402
import mm2  # noqa: E402
from gem import embed, generate, pmap  # noqa: E402
from run_mm import episodes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INST_SYS = ("You help search a user's long chat history with an assistant. Given a question about that history, list the "
            "concrete things the user may have mentioned in passing or shown in photos that would be needed to answer it: "
            "specific kinds, brands, or names of items, events, places, or documents (for example, for 'coffee makers': "
            "espresso machine, moka pot, pour-over dripper, French press, drip coffee maker, capsule machine). Return JSON "
            "{\"instances\": [up to 8 short phrases]}.")
_SCHEMA = {"type": "object", "properties": {"instances": {"type": "array", "items": {"type": "string"}}}, "required": ["instances"]}


def instances(question: str, model="gemini-2.5-flash-lite"):
    q = question.split("\n", 1)[1] if question.startswith("Question date:") else question
    r = generate([f"Question: {q}"], model=model, system=INST_SYS, thinking=0, json_schema=_SCHEMA, tag="p5:instances",
                 max_output_tokens=300)
    return [s for s in (mm2.parse_json(r["text"]).get("instances") or []) if isinstance(s, str)][:8]


def expanded_scores(ix, qa, key, insts):
    s = ix.scores(qa, key)
    if not insts:
        return s
    V = embed([mm.QA_QUERY_FMT.format(t) for t in insts])
    return np.maximum(s, (ix.M[key] @ V.T).max(axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="memlens")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    if a.bench == "memlens":
        eps = episodes("memlens", "32k", json.load(open(ROOT / f"results/memlens_{a.split}.json")))
        caps = {"short": mm.build_captions(eps, "short")}; caps["dense"] = caps["short"]; key = "cap:short"
    else:
        eps = episodes("memeye", "all"); caps = {s: mm.build_captions(eps, s) for s in ["short", "dense"]}; key = "cap:dense"
    jobs = [(e, qa) for e in eps for qa in e.qas]
    I = dict(pmap(lambda j: (j[1].qid, instances(j[1].question)), jobs, workers=16, desc="instances"))
    res = {}
    for e in eps:
        ix = mm.RoundIndex(e, caps, [key]); r2i = e.rid2idx
        for qa in e.qas:
            gold = {r2i[c] for c in qa.clue if c in r2i}
            if not gold:
                continue
            for name, s in [("question only", ix.scores(qa, key)), ("question + instances", expanded_scores(ix, qa, key, I[qa.qid]))]:
                o = np.argsort(-s)
                for k in [10, 20, 40]:
                    top = set(o[:k].tolist())
                    res.setdefault((name, k), []).append((float(gold <= top), len(gold & top) / len(gold)))
    print(f"== {a.bench}: instance expansion (mean {np.mean([len(v) for v in I.values()]):.1f} instances per question)")
    for (name, k), v in sorted(res.items(), key=lambda x: (x[0][1], x[0][0])):
        v = np.array(v)
        print(f"   top-{k:<3d} {name:22s} FSR {v[:, 0].mean():.3f} | coverage {v[:, 1].mean():.3f}")
    json.dump(I, open(ROOT / f"results/p5/instances_{a.bench}{'_' + a.split if a.bench == 'memlens' else ''}.json", "w"), indent=0)


if __name__ == "__main__":
    main()
