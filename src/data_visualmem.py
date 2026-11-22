"""VisualMem (Nguyen et al., NeurIPS 2026; viettmab/visualmem): personal visual memory whose decisive evidence is in
images. One Episode per persona: every event is a session, a round is a user turn with the following assistant turn,
and the event image is attached to its turn. All questions are asked after the whole history (as in the benchmark
pipeline). Questions are multiple choice; image-choice questions carry their option images. Gold rounds = the image
rounds of the event that the question belongs to. Images are downscaled to 1,024 pixels on the longer side."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from data_mm import QA, Episode, Round

DATA = Path(__file__).resolve().parents[1] / "data"


def _img(rel: str) -> str:
    """Images are used downscaled to 1,024 pixels on the longer side (data/visualmem_data_1024, same relative paths)."""
    r = Path(rel).relative_to("visualmem_data")
    return str((DATA / "visualmem_data_1024" / r).with_suffix(".jpg"))


def load_visualmem(qids: Optional[List[str]] = None, personas: Optional[List[int]] = None) -> List[Episode]:
    keep = set(qids) if qids else None
    eps = []
    for p in json.load(open(DATA / "VisualMem/data.json")):
        if personas is not None and p["idx"] not in personas:
            continue
        rounds, order, img_round = [], 0, {}
        for e in sorted(p["events"], key=lambda e: (e["date"], e["event_id"])):
            sid = f"E{e['event_id']}"
            turns = e["conversation"]
            j, t = 0, 0
            while t < len(turns):
                pair = [turns[t]] + ([turns[t + 1]] if turns[t]["role"] == "user" and t + 1 < len(turns)
                                     and turns[t + 1]["role"] == "assistant" else [])
                t += len(pair)
                j += 1
                imgs = [x["image_path"] for x in pair if x.get("image_path")]
                r = Round(rid=f"{sid}:R{j}", sid=sid, date=e["date"], order=order,
                          user=" ".join(x["content"] for x in pair if x["role"] == "user"),
                          assistant=" ".join(x["content"] for x in pair if x["role"] == "assistant"),
                          images=[_img(x) for x in imgs], image_ids=[f"{sid}_img{k + 1}" for k in range(len(imgs))])
                rounds.append(r)
                if imgs:
                    img_round.setdefault(sid, []).append(r.rid)
                order += 1
        qas = []
        for e in p["events"]:
            sid = f"E{e['event_id']}"
            for q in e.get("questions") or []:
                qid = f"vm{p['idx']}_{e['event_id']}_{q['id']}"
                if keep is not None and qid not in keep:
                    continue
                ch = q["choices"]
                img_choices = all(isinstance(v, str) and v.endswith((".jpg", ".jpeg", ".png")) for v in ch.values())
                if img_choices:
                    text = q["question"] + "\nThe options are the images labelled A to D below. Answer with the letter only."
                    opts = [(k, _img(v)) for k, v in sorted(ch.items())]
                else:
                    text = q["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in sorted(ch.items())) + \
                        "\nAnswer with the letter of the correct option only."
                    opts = None
                qimg = _img(q["image_path"]) if q.get("image_path") else None
                qas.append(QA(qid=qid, task=f"visualmem_{p['idx']}", question=text, answer=q["ground_truth"],
                              qtype=q.get("type") or q.get("question_type") or "text", clue=img_round.get(sid, []), x=q.get("type") or q.get("question_type") or "text",
                              y=e.get("topic"), option_images=opts, question_image=qimg))
        if qas:
            eps.append(Episode(name=f"visualmem_{p['idx']}", bench="visualmem", rounds=rounds, qas=qas))
    return eps


def mc_score(pred: str, gold: str) -> float:
    import re
    m = re.search(r"\b([ABCD])\b", (pred or "").strip().upper())
    return float(bool(m) and m.group(1) == gold.strip().upper())
