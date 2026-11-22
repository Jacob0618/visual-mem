"""Text memory benchmarks (read-only files from ~/Desktop/jevmem/data): LoCoMo and LongMemEval (S / M).

Unified unit = Unit(uid, text, date, order, sid). LoCoMo unit = one utterance (turn); LongMemEval unit = one round
(user turn + following assistant turn), the granularity LongMemEval's paper found best for QA.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

JEV = Path.home() / "Desktop/jevmem/data"
LOCOMO = JEV / "locomo10.json"
LME_S = JEV / "big/longmemeval_s_cleaned.json"
LME_M = JEV / "big/lme_m/longmemeval_m_cleaned.json"
ROOT = Path(__file__).resolve().parents[1]
LOCOMO_CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


@dataclass
class Unit:
    uid: str
    sid: str
    date: str
    order: int
    text: str           # rendered text (what is embedded and what the reader sees)


@dataclass
class TQA:
    qid: str
    question: str
    answer: str
    qtype: str
    evidence: List[str]
    episode: str
    question_date: Optional[str] = None
    abstention: bool = False


@dataclass
class TEpisode:
    name: str
    units: List[Unit]
    qas: List[TQA] = field(default_factory=list)


def load_locomo(include_adversarial: bool = False) -> List[TEpisode]:
    raw = json.load(open(LOCOMO))
    eps = []
    for conv in raw:
        c = conv["conversation"]
        idxs = sorted(int(k.split("_")[1]) for k in c if k.startswith("session_") and not k.endswith("date_time"))
        units, k = [], 0
        for i in idxs:
            date = c.get(f"session_{i}_date_time", "")
            for t in c[f"session_{i}"]:
                txt = f"[{date}] {t['speaker']}: {t['text']}"
                if t.get("blip_caption"):  # LoCoMo images are only released as BLIP captions (+search query)
                    txt += f" [shares a photo: {t['blip_caption']}]"
                units.append(Unit(uid=t["dia_id"], sid=f"S{i}", date=date, order=k, text=txt))
                k += 1
        qas = []
        for j, q in enumerate(conv["qa"]):
            cat = LOCOMO_CAT.get(q["category"], str(q["category"]))
            if cat == "adversarial" and not include_adversarial:
                continue
            ans = q.get("answer", q.get("adversarial_answer", ""))
            qas.append(TQA(qid=f"{conv['sample_id']}:{j}", question=q["question"], answer=str(ans), qtype=cat,
                           evidence=list(q.get("evidence") or []), episode=conv["sample_id"],
                           abstention=(cat == "adversarial")))
        eps.append(TEpisode(name=conv["sample_id"], units=units, qas=qas))
    return eps


def lme_episode(q: dict) -> TEpisode:
    units, k = [], 0
    for sid, date, sess in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
        i = 0
        while i < len(sess):
            t = sess[i]
            if t["role"] == "user":
                nxt = sess[i + 1] if i + 1 < len(sess) and sess[i + 1]["role"] == "assistant" else None
                txt = f"[{date}] User: {t['content']}" + (f"\nAssistant: {nxt['content']}" if nxt else "")
                has = bool(t.get("has_answer") or (nxt and nxt.get("has_answer")))
                units.append(Unit(uid=f"{sid}#{i}" + ("*" if has else ""), sid=sid, date=date, order=k, text=txt))
                i += 2 if nxt else 1
            else:  # assistant-first turn
                units.append(Unit(uid=f"{sid}#{i}" + ("*" if t.get("has_answer") else ""), sid=sid, date=date,
                                  order=k, text=f"[{date}] Assistant: {t['content']}"))
                i += 1
            k += 1
    qid = q["question_id"]
    ev = [u.uid for u in units if u.uid.endswith("*")]
    tq = TQA(qid=qid, question=q["question"], answer=str(q["answer"]), qtype=q["question_type"], evidence=ev,
             episode=qid, question_date=q.get("question_date"), abstention=qid.endswith("_abs"))
    return TEpisode(name=qid, units=units, qas=[tq])


def lme_subset_ids(n_per_type: int = 17, seed: int = 0, path: Path = LME_S) -> List[str]:
    """Stratified question-id subset (fixed seed) shared by S and M so the scale effect is a paired comparison."""
    d = json.load(open(path))
    by = defaultdict(list)
    for q in d:
        by[("abs" if q["question_id"].endswith("_abs") else q["question_type"])].append(q["question_id"])
    rng = random.Random(seed)
    out = []
    for t in sorted(by):
        ids = sorted(by[t])
        out += rng.sample(ids, min(n_per_type, len(ids)))
    return sorted(out)


def load_lme(which: str = "S", ids: Optional[List[str]] = None) -> List[TEpisode]:
    if which == "S":
        d = json.load(open(LME_S))
        keep = set(ids) if ids else None
        return [lme_episode(q) for q in d if keep is None or q["question_id"] in keep]
    # LongMemEval-M is 2.7 GB: stream it and keep only the requested ids (cached as a small subset file)
    assert ids, "LME-M requires an explicit id subset"
    sub = ROOT / f"data/lme_m_subset_{len(ids)}.json"
    if not sub.exists():
        import ijson
        keep, out = set(ids), []
        with open(LME_M, "rb") as f:
            for q in ijson.items(f, "item"):
                if q["question_id"] in keep:
                    out.append(q)
                    if len(out) == len(keep):
                        break
        json.dump(out, open(sub, "w"))
    d = json.load(open(sub))
    return [lme_episode(q) for q in d]


if __name__ == "__main__":
    import collections
    eps = load_locomo()
    print("LoCoMo", len(eps), "convs", sum(len(e.units) for e in eps), "turns", sum(len(e.qas) for e in eps), "qas",
          collections.Counter(q.qtype for e in eps for q in e.qas))
    ids = lme_subset_ids()
    print("LME subset", len(ids))
