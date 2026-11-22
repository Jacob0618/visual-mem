"""Unified loaders for the multimodal memory benchmarks (MemEye, Mem-Gallery).

Each task/scenario -> Episode(rounds=[Round...], qas=[QA...]). Rounds are in chronological order.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
MEMEYE = ROOT / "data/MemEye/data"
MEMGAL = ROOT / "data/Mem-Gallery/data"

MEMEYE_TASKS = ["Brand_Memory_Test", "Card_Playlog_Test", "Cartoon_Entertainment_Companion",
                "Home_Renovation_Interior_Design", "Multi-Scene_Visual_Case_Archive_Assistant",
                "Outdoor_Navigation_Route_Memory_Assistant", "Personal_Health_Dashboard_Assistant",
                "Social_Chat_Memory_Test"]


@dataclass
class Round:
    rid: str
    sid: str
    date: str
    order: int
    user: str
    assistant: str
    images: List[str] = field(default_factory=list)       # absolute paths
    image_ids: List[str] = field(default_factory=list)
    given_captions: List[str] = field(default_factory=list)  # dataset-provided captions (not used by default)


@dataclass
class QA:
    qid: str
    task: str
    question: str
    answer: str
    qtype: str                      # MemEye "X3|Y2"; Mem-Gallery point code
    clue: List[str]
    x: Optional[str] = None
    y: Optional[str] = None
    question_image: Optional[str] = None
    mcq: Optional[list] = None      # MemEye: list of rotations [{A..D, answer}]
    latest_clue: Optional[str] = None
    stale_clues: Optional[list] = None
    option_images: Optional[list] = None   # VisualMem image-choice questions: [(label, path), ...]


@dataclass
class Episode:
    name: str
    bench: str
    rounds: List[Round]
    qas: List[QA]
    profile: dict = field(default_factory=dict)
    persona_images: List[tuple] = field(default_factory=list)  # (label, path) e.g. MemEye Social faces/avatars

    @property
    def rid2idx(self):
        return {r.rid: i for i, r in enumerate(self.rounds)}

    @property
    def n_images(self):
        return sum(len(r.images) for r in self.rounds)


_MANIFEST = ROOT / "data/_api_images/manifest.json"
_CANON = json.load(open(_MANIFEST)) if _MANIFEST.exists() else {}


def _resolve(p: str, dialog_dir: Path, image_dir: Path) -> str:
    """Resolve a dataset image path; return its canonical API copy (scripts/prep_images.py) when available."""
    for base in (dialog_dir, image_dir):
        c = os.path.normpath(os.path.join(base, p))
        if os.path.exists(c):
            return _CANON[c]["canon"] if c in _CANON else c
    raise FileNotFoundError(p)


def _rounds(d, dialog_dir, image_dir):
    out, k = [], 0
    for s in d["multi_session_dialogues"]:
        for r in s["dialogues"]:
            imgs = [_resolve(p, dialog_dir, image_dir) for p in (r.get("input_image") or [])]
            out.append(Round(rid=r["round"], sid=s["session_id"], date=str(s.get("date", "")), order=k,
                             user=r.get("user", "") or "", assistant=r.get("assistant", "") or "", images=imgs,
                             image_ids=list(r.get("image_id") or []),
                             given_captions=list(r.get("image_caption") or [])))
            k += 1
    return out


def load_memeye(task: str) -> Episode:
    dd, idir = MEMEYE / "dialog", MEMEYE / "image"
    op = json.load(open(dd / f"{task}_Open.json"))
    mc = json.load(open(dd / f"{task}.json"))
    rounds = _rounds(op, dd, idir)
    mc_by_q = {}
    for i, q in enumerate(mc["human-annotated QAs"]):
        mc_by_q[q.get("question_id", f"Q{i + 1}")] = q
    qas = []
    for i, q in enumerate(op["human-annotated QAs"]):
        qid = q.get("question_id", f"Q{i + 1}")
        m = mc_by_q.get(qid)
        x = q["point"][0][0] if q.get("point") else None
        y = q["point"][1][0] if q.get("point") else None
        qi = q.get("question_image")
        qas.append(QA(qid=f"{task}:{qid}", task=task, question=q["question"], answer=str(q["answer"]),
                      qtype=f"{x}|{y}", x=x, y=y, clue=list(q.get("clue") or []),
                      question_image=_resolve(qi, dd, idir) if qi else None,
                      mcq=(m or {}).get("options"), latest_clue=q.get("latest_clue_round"),
                      stale_clues=q.get("stale_clue_rounds")))
    persona_imgs = []
    for p in op.get("personas", []) or []:
        for key in ("face_path", "avatar_path"):
            if p.get(key):
                persona_imgs.append((f"{p.get('name')} ({key.split('_')[0]})", _resolve(p[key], dd, idir)))
    return Episode(name=task, bench="memeye", rounds=rounds, qas=qas, profile=op.get("character_profile", {}),
                   persona_images=persona_imgs)


def memgal_tasks() -> List[str]:
    return sorted(p.stem for p in (MEMGAL / "dialog").glob("*.json"))


def load_memgal(task: str) -> Episode:
    dd, idir = MEMGAL / "dialog", MEMGAL / "image"
    d = json.load(open(dd / f"{task}.json"))
    rounds = _rounds(d, dd, idir)
    qas = []
    for i, q in enumerate(d["human-annotated QAs"]):
        qi = q.get("question_image")
        qas.append(QA(qid=f"{task}:Q{i + 1}", task=task, question=q["question"], answer=str(q["answer"]),
                      qtype=q["point"], clue=list(q.get("clue") or []),
                      question_image=_resolve(qi, dd, idir) if qi else None))
    return Episode(name=task, bench="memgal", rounds=rounds, qas=qas, profile=d.get("character_profile", {}))


if __name__ == "__main__":
    import collections
    for t in MEMEYE_TASKS:
        e = load_memeye(t)
        r2i = e.rid2idx
        bad = sum(1 for q in e.qas for c in q.clue if c not in r2i)
        print(f"MemEye {t[:30]:30s} rounds={len(e.rounds):4d} imgs={e.n_images:4d} qas={len(e.qas):3d} "
              f"bad_clues={bad} mcq={sum(q.mcq is not None for q in e.qas)} qimg={sum(q.question_image is not None for q in e.qas)} "
              f"persona_imgs={len(e.persona_images)} cells={dict(collections.Counter(q.qtype for q in e.qas))}")
    tot = collections.Counter()
    for t in memgal_tasks():
        e = load_memgal(t)
        r2i = e.rid2idx
        tot["bad"] += sum(1 for q in e.qas for c in q.clue if c not in r2i)
        tot["qas"] += len(e.qas)
        tot["imgs"] += e.n_images
        tot["qimg"] += sum(q.question_image is not None for q in e.qas)
    print("Mem-Gallery", dict(tot))
