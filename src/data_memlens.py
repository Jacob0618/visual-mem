"""MemLens loader (arXiv:2605.14906; HF xiyuRenBill/MEMLENS, CC-BY-4.0).

Every question has its own haystack, so every question becomes one Episode with a single QA. Sessions are sorted by
date; a round is one user turn plus the following assistant turn. Gold rounds: rounds of answer sessions that contain
a has_answer turn or a needle image (the image that carries the evidence, e.g. a receipt). Abstention questions
(answer_refusal) are excluded.
"""
from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List

from data_mm import QA, Episode, Round

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/MemLens"


def _date(s: str) -> datetime:
    try:
        return datetime.strptime(s.split(" (")[0] + " " + s.split(") ")[-1], "%Y/%m/%d %H:%M")
    except Exception:  # noqa: BLE001
        return datetime(2100, 1, 1)


@lru_cache(maxsize=4)
def _raw(length: str):
    return json.load(open(DATA / f"dataset_{length}.json"))


def load_memlens(length: str = "32k", qids: List[str] | None = None) -> List[Episode]:
    keep = set(qids) if qids else None
    eps = []
    for it in _raw(length):
        if it["question_type"] == "answer_refusal" or (keep is not None and it["question_id"] not in keep):
            continue
        ans = set(it["answer_session_ids"])
        sessions = sorted(zip(it["haystack_session_ids"], it["haystack_dates"], it["haystack_sessions"]),
                          key=lambda x: _date(x[1]))
        rounds, clue, order = [], [], 0
        for sid, date, turns in sessions:
            j, t = 0, 0
            while t < len(turns):
                if turns[t]["role"] != "user":   # stray assistant turn: attach as its own round
                    pair = [turns[t]]
                    t += 1
                else:
                    pair = [turns[t]] + ([turns[t + 1]] if t + 1 < len(turns) and turns[t + 1]["role"] == "assistant" else [])
                    t += len(pair)
                j += 1
                user = " ".join(x["content"] for x in pair if x["role"] == "user")
                asst = " ".join(x["content"] for x in pair if x["role"] == "assistant")
                ims = [im for x in pair for im in (x.get("images") or [])]
                paths = [str(DATA / "release_images" / im["file"]) for im in ims]
                rid = f"{sid}:R{j}"
                rounds.append(Round(rid=rid, sid=sid, date=date, order=order, user=user, assistant=asst, images=paths,
                                    given_captions=[im.get("blip_caption", "") for im in ims]))
                order += 1
                if sid in ans and (any(x.get("has_answer") for x in pair)
                                   or any(im["file"].startswith("needle_images/") for im in ims)):
                    clue.append(rid)
        q = f"Question date: {it['question_date']}\n{it['question']}"
        qa = QA(qid=it["question_id"], task=it["question_id"], question=q, answer=str(it["answer"]),
                qtype=f"{it['question_type']}|{it.get('question_subtype') or ''}", clue=clue,
                x=it["question_type"], y=it.get("question_subtype"))
        eps.append(Episode(name=it["question_id"], bench="memlens", rounds=rounds, qas=[qa]))
    return eps
