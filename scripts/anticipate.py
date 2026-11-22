"""Phase 6: anticipatory set resolution (AnSR). At ingestion, the strong model reads a user's index once and writes
questions the user may later ask (emphasis on sets / series / changes); each anticipated question is resolved with the
strong selector (QSS). At question time, a lightweight selector sees the anticipated resolutions most similar to the
real question below the index (resolution write-back without real past questions).
python scripts/anticipate.py --n 40            (MemEye; writes results/p6/anticipated_memeye.json)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm  # noqa: E402
import mm2  # noqa: E402
from gem import generate, pmap  # noqa: E402
from run_mm import episodes  # noqa: E402
from data_mm import QA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GEN_SYS = (
    "You maintain the long-term memory of a personal assistant. You see the compact index of one user's whole "
    "conversation history with the assistant (one line per round; images are summarised briefly). Anticipate the "
    "questions this user may later ask about their own history that require opening past rounds. Favour questions about "
    "groups or series of items: all items of one kind, every version of something over time, how something changed, "
    "counts, comparisons, the first, last, or a specific numbered item, using the labels the user and the conversation "
    "use (day or session numbers, names, situations). Also include some questions about single details. The questions "
    "must be diverse and cover the whole history. Return JSON {\"questions\": [...]}.")
_SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
           "required": ["questions"]}


def anticipate(e, caps, n):
    body = mm2.toc(e, caps["short"])
    r = generate([f"=== MEMORY INDEX ({len(e.rounds)} rounds) ===\n{body}\n=== END INDEX ===\n\nWrite {n} questions."],
                 model="gemini-3.8-flash", system=GEN_SYS, thinking="low", json_schema=_SCHEMA, tag="p6:anticipate",
                 max_output_tokens=4096)
    qs = [q for q in (mm2.parse_json(r["text"]).get("questions") or []) if isinstance(q, str)][:n]
    return qs, r["usage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    eps = episodes("memeye", "all")
    caps = {s: mm.build_captions(eps, s) for s in ["short", "dense"]}
    out, usages = {}, []
    for e in eps:
        qs, u = anticipate(e, caps, a.n)
        usages.append(("gemini-3.8-flash", u))

        def resolve(q):
            fake = QA(qid=f"anticip:{e.name}:{abs(hash(q)) % 10**8}", task=e.name, question=q, answer="", qtype="", clue=[])
            ids, uu = mm2.toc_select(e, fake, caps["short"], 40, "gemini-3.8-flash", "low")
            return {"question": q, "rids": [e.rounds[i].rid for i in ids]}, uu
        res = pmap(resolve, qs, workers=8, desc=e.name)
        out[e.name] = [r for r, _ in res]
        usages += [("gemini-3.8-flash", uu) for _, uu in res]
        print(e.name, len(qs), "anticipated questions", flush=True)
    from gem import PRICES
    p = PRICES["gemini-3.8-flash"]
    usd = sum(((u["prompt"] - u["cached"]) * p["in"] + u["cached"] * p["cached"] + (u["out"] + u["thoughts"]) * p["out"]) / 1e6
              for _, u in usages)
    json.dump({"n": a.n, "ingestion_usd": usd, "per_history": out}, open(ROOT / "results/p6/anticipated_memeye.json", "w"))
    print(f"ingestion cost: ${usd:.2f} for {len(eps)} histories")


if __name__ == "__main__":
    main()
