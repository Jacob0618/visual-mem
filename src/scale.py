"""Scale stress test: grow a MemEye history with distractor sessions from Mem-Gallery (another user's multi-topic
image chats). Distractor sessions are re-dated uniformly inside the target's date span (seeded) and interleaved, so
the target questions/clues are unchanged while the haystack grows to ~N extra images.
"""
from __future__ import annotations

import copy
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_mm import Episode, Round, load_memgal, memgal_tasks  # noqa: E402


def _d(s):
    y, m, dd = (int(x) for x in s[:10].split("-"))
    return date(y, m, dd)


def distractor_sessions(seed: int = 0):
    sess = []
    for t in memgal_tasks():
        e = load_memgal(t)
        by = {}
        for r in e.rounds:
            by.setdefault(r.sid, []).append(r)
        for sid, rs in by.items():
            sess.append((f"{t[:12]}/{sid}", rs))
    random.Random(seed).shuffle(sess)
    return sess


def grow(ep: Episode, n_images: int, seed: int = 0) -> Episode:
    if n_images <= 0:
        return ep
    tsess = []
    for r in ep.rounds:
        if not tsess or tsess[-1][0] != r.sid:
            tsess.append((r.sid, r.date, []))
        tsess[-1][2].append(r)
    tdates = [_d(dt) for _, dt, _ in tsess]
    d0, d1 = min(tdates), max(tdates)
    span = max(1, (d1 - d0).days)
    rng = random.Random(seed + 1)
    picked, k = [], 0
    for j, (sid, rs) in enumerate(distractor_sessions(seed)):
        if k >= n_images:
            break
        nd = d0 + timedelta(days=rng.randint(0, span))
        new = []
        for r in rs:
            rr = copy.copy(r)
            rr.sid, rr.rid, rr.date = f"DX{j}_{r.sid}", f"DX{j}_{r.rid}", nd.isoformat()
            rr.image_ids = [f"DX{j}/{x}" for x in (r.image_ids or [f"img{i + 1}" for i in range(len(r.images))])]
            new.append(rr)
        # insert just before the first target session (in original order) dated after nd
        ins = next((i for i, td in enumerate(tdates) if td > nd), len(tdates))
        picked.append((ins, -1, rng.random(), new))
        k += sum(len(r.images) for r in rs)
    allsess = [(i, 0, 0.0, rs) for i, (_, _, rs) in enumerate(tsess)] + picked
    allsess.sort(key=lambda x: (x[0], x[1], x[2]))
    rounds, o = [], 0
    for _, _, _, rs in allsess:
        for r in rs:
            rr = copy.copy(r)
            rr.order = o
            o += 1
            rounds.append(rr)
    g = Episode(name=f"{ep.name}+dx{n_images}", bench=ep.bench, rounds=rounds, qas=ep.qas, profile=ep.profile,
                persona_images=ep.persona_images)
    return g
