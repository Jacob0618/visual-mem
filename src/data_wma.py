"""WorldMemArena (Liu et al., 2026; arXiv 2605.29341), lifelong part (personal + project), as an external benchmark for
the frozen VGS design. Each QA checkpoint asks questions after a prefix of the sessions (its covered_sessions); we turn
every (sample, checkpoint) into one Episode whose rounds are the covered sessions (user turn + following assistant turn,
images attached to the turns). The benchmark annotates evidence as session-level memory points, not rounds, so QA.clue
is empty and the evidence sessions are kept in EVIDENCE_SESSIONS for session-level analyses.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from data_mm import QA, Episode, Round

DATA = Path(__file__).resolve().parents[1] / "data/WorldMemArena/lifelong"
EVIDENCE_SESSIONS: Dict[str, List[str]] = {}


def _date(ts: str) -> str:
    try:
        d = datetime.strptime(ts.strip(), "%b %d, %Y, %H:%M:%S")
        return d.strftime("%Y-%m-%d (%a) %H:%M")
    except Exception:  # noqa: BLE001
        return ts


def sample_files() -> List[Path]:
    return sorted(DATA.glob("personal/*.json")) + sorted(DATA.glob("project/*/*.json"))


def load_wma(qids: Optional[List[str]] = None) -> List[Episode]:
    keep = set(qids) if qids else None
    eps = []
    for f in sample_files():
        d = json.load(open(f))
        sid_of_mp = {}
        for blk in d.get("memory_points", []):
            for mp in blk.get("memory_points", []):
                sid_of_mp[mp["memory_id"]] = mp.get("_session_id") or blk.get("session_id")
        sessions = {s["_v2_session_id"]: s for s in d["sessions"]}
        order_all = [s["_v2_session_id"] for s in d["sessions"]]
        for ck in d["qa_checkpoints"]:
            covered = [s for s in order_all if s in set(ck["covered_sessions"])]
            rounds, order = [], 0
            for sid in covered:
                turns = sessions[sid]["dialogue"]
                j, t = 0, 0
                while t < len(turns):
                    if turns[t]["role"] != "user":
                        pair = [turns[t]]
                        t += 1
                    else:
                        pair = [turns[t]] + ([turns[t + 1]] if t + 1 < len(turns) and turns[t + 1]["role"] == "assistant" else [])
                        t += len(pair)
                    j += 1
                    atts = [a for x in pair for a in (x.get("attachments") or []) if a.get("file_path")]
                    paths = [str(f.parent / a["file_path"]) for a in atts]
                    rounds.append(Round(rid=f"{sid}:R{j}", sid=sid, date=_date(pair[0].get("timestamp", "")), order=order,
                                        user=" ".join(x["content"] for x in pair if x["role"] == "user"),
                                        assistant=" ".join(x["content"] for x in pair if x["role"] == "assistant"),
                                        images=paths, image_ids=[a.get("image_id") or Path(a["file_path"]).stem for a in atts],
                                        given_captions=[a.get("caption", "") for a in atts]))
                    order += 1
            qas = []
            for k, q in enumerate(ck["questions"]):
                qid = f"{d['sample_id']}:{ck['checkpoint_id']}:{k}"
                if keep is not None and qid not in keep:
                    continue
                EVIDENCE_SESSIONS[qid] = sorted({sid_of_mp.get(ev.get("memory_id"), "") for ev in q.get("evidence", [])} - {""})
                qas.append(QA(qid=qid, task=f"{d['sample_id']}:{ck['checkpoint_id']}", question=q["question"],
                              answer=str(q["answer"]), qtype=q.get("question_type_abbrev") or q.get("question_type", ""),
                              clue=[], x=q.get("question_type"), y=q.get("difficulty")))
            if qas:
                eps.append(Episode(name=f"{d['sample_id']}:{ck['checkpoint_id']}", bench="wma", rounds=rounds, qas=qas))
    return eps
