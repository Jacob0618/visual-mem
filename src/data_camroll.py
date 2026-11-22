"""CamRoll-yfcc20 (Personal AI Agent for Camera Roll VQA, arXiv 2606.05275; thaoshibe/camroll-yfcc20): 20 personal
camera rolls sampled from YFCC100M, 500 to 1,000 photos each over several years.
One Episode per camera roll. A round is one photo: its text is the capture time and its image is the photo, downscaled
to 1,024 pixels on the longer side. The photos of one calendar day form a session. Questions are multiple choice and are
asked after the whole camera roll; the released option "a" is the correct one, so options are shuffled with a seed
derived from the question id. Gold rounds are the photos listed in image_ref (episodic questions); semantic questions
about the album owner have none. The owner's reference photo is given to every reader as a persona image."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import zipfile
from pathlib import Path
from typing import List, Optional

from data_mm import QA, Episode, Round

DATA = Path(__file__).resolve().parents[1] / "data"
SRC = DATA / "CamRoll"
OUT = DATA / "camroll_1024"


def users() -> List[str]:
    return sorted(p.name for p in SRC.iterdir() if p.is_dir() and (p / "album_data.json").exists())


def prepare(user: str) -> None:
    """Extract images.zip and write 1,024-pixel JPEGs to data/camroll_1024/<user>/ (idempotent)."""
    from PIL import Image, ImageOps
    dst = OUT / user
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SRC / user / "images.zip") as z:
        for n in z.namelist():
            if not n.lower().endswith((".jpg", ".jpeg", ".png")) or Path(n).name.startswith("."):
                continue
            o = dst / (Path(n).stem + ".jpg")
            if o.exists():
                continue
            with z.open(n) as f:
                im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            im.thumbnail((1024, 1024))
            im.save(o, "JPEG", quality=90)
    o = dst / "_profile.jpg"
    if not o.exists():
        im = ImageOps.exif_transpose(Image.open(SRC / user / "profile.jpg")).convert("RGB")
        im.thumbnail((1024, 1024))
        im.save(o, "JPEG", quality=90)


def load_camroll(qids: Optional[List[str]] = None, only_users: Optional[List[str]] = None) -> List[Episode]:
    keep = set(qids) if qids else None
    eps = []
    for ui, user in enumerate(users()):
        if only_users is not None and user not in only_users:
            continue
        meta = [(r["image"], r["datetaken"]) for r in csv.DictReader(open(SRC / user / "metadata.csv"))]
        meta = [(n, t) for n, t in meta if (OUT / user / (Path(n).stem + ".jpg")).exists()]
        meta.sort(key=lambda x: (x[1], x[0]))
        rounds, rid_of, day_no, pos = [], {}, {}, {}
        for order, (name, taken) in enumerate(meta):
            day, clock = taken[:10], taken[11:16]
            d = day_no.setdefault(day, len(day_no) + 1)
            pos[d] = pos.get(d, 0) + 1
            rid = f"D{d}:P{pos[d]}"
            rid_of[name] = rid
            rounds.append(Round(rid=rid, sid=f"D{d}", date=day, order=order, user=f"Photo taken at {clock}.", assistant="",
                                images=[str(OUT / user / (Path(name).stem + ".jpg"))], image_ids=[f"D{d}_P{pos[d]}"]))
        if not rounds:                       # camera roll not prepared yet (images.zip missing or incomplete)
            continue
        album = json.load(open(SRC / user / "album_data.json"))
        qas = []
        for kind in ("semantic", "episodic"):
            for i, q in enumerate(album.get(f"{kind}_qa") or []):
                qid = f"cr{ui:02d}_{kind[0]}{i:02d}"
                if keep is not None and qid not in keep:
                    continue
                opts = sorted(q["answer"].items())                     # [("a", correct), ("b", ...), ...]
                random.Random(int(hashlib.md5(qid.encode()).hexdigest()[:8], 16)).shuffle(opts)
                letters = "ABCD"[: len(opts)]
                text = q["question"] + "\n" + "\n".join(f"{L}. {v}" for L, (_, v) in zip(letters, opts)) + \
                    "\nAnswer with the letter of the correct option only."
                gold = next(L for L, (k, _) in zip(letters, opts) if k == "a")
                clue = [rid_of[n] for n in (q.get("image_ref") or []) if n in rid_of]
                qas.append(QA(qid=qid, task=f"camroll_{user}", question=text, answer=gold, qtype=kind, clue=clue, x=kind,
                              y=str(bool(q.get("detail_type")))))
        if qas:
            eps.append(Episode(name=f"camroll_{user}", bench="camroll", rounds=rounds, qas=qas,
                               persona_images=[("album owner (reference photo)", str(OUT / user / "_profile.jpg"))]))
    return eps


if __name__ == "__main__":
    import sys
    for u in users():
        if (SRC / u / "images.zip").exists():
            try:
                prepare(u)
            except zipfile.BadZipFile:
                print(u, "images.zip incomplete", flush=True)
                continue
            print(u, len(list((OUT / u).glob("*.jpg"))) - 1, "photos", flush=True)
    if "--stats" in sys.argv:
        E = load_camroll()
        import collections
        print("users", len(E), "rounds", sum(len(e.rounds) for e in E), "questions", sum(len(e.qas) for e in E),
              dict(collections.Counter(q.qtype for e in E for q in e.qas)))
