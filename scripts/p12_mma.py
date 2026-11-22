"""Phase 12: MMA (Lu et al., 2026; arXiv 2602.16493), confidence-aware multimodal memory, as implemented in the
official MemEye evaluation framework (github.com/MinghoKwok/MemEye, benchmark/mma), on our benchmarks.
  .venv_bl/bin/python scripts/p12_mma.py --bench memeye
MMA stores every user and assistant turn as a memory entry (user entries keep their images), embeds text with
MiniLM and images with a joint text-image encoder, scores each entry's confidence from source credibility, temporal
decay, and consensus with similar entries, and retrieves the top 10 entries by (blended similarity x confidence).
We run its released build and retrieval code with the MemEye authors' Gemini configuration and give its formatted
context, with the images of the retrieved entries, to our readers. Substitution: the joint encoder is
gemini-embedding-2 instead of CLIP ViT-B/32, which could not be downloaded here.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "baselines/MemEye_harness"
sys.path.insert(0, str(ROOT / "scripts"))
import p12_common as C  # noqa: E402
import p12_budget as B  # noqa: E402

SYSTEM = "mma"
JOINT = "gemini-embedding-2"


class GeminiJointEmbedder:
    """embed_text / embed_image in one joint space (the interface MMA expects from CLIP/SigLIP)."""
    _lock = threading.Lock()

    def _embed(self, item):
        import gem
        key = gem._emb_item_key(JOINT, None, [item])
        if key not in gem._emb_cache.get_many([key]):
            B.charge(SYSTEM, 0.0002 if isinstance(item, gem.Img) else len(item) / 4 * 0.15e-6, "build:joint-embeddings")
        return gem.embed([item], model=JOINT)[0].astype(np.float32).tolist()

    def embed_text(self, text):
        return self._embed(text)

    def embed_image(self, path):
        import gem
        return self._embed(gem.Img(str(path)))


class EpisodeDataset:
    """The dataset interface of the MemEye framework (session_order / get_session / rounds) over our Episode."""

    def __init__(self, ep):
        self.rounds, self._sessions, self._order, self.idx_of = {}, {}, [], {}
        for i, r in enumerate(ep.rounds):
            if r.sid not in self._sessions:
                self._sessions[r.sid] = {"date": r.date or "", "dialogues": []}
                self._order.append(r.sid)
            rid = f"R{i}"
            self.idx_of[rid] = i
            self._sessions[r.sid]["dialogues"].append({"round": rid})
            self.rounds[rid] = {"user": r.user, "assistant": r.assistant, "images": [str(p) for p in r.images]}

    def session_order(self):
        return list(self._order)

    def get_session(self, sid):
        return self._sessions[sid]


def setup():
    sys.path.insert(0, str(HARNESS))
    import torch
    from transformers import PreTrainedModel  # noqa: F401
    import sentence_transformers  # noqa: F401
    torch.backends.mps.is_available = lambda: False
    import benchmark.embeddings as E
    E.get_multimodal_embedder = lambda **kw: GeminiJointEmbedder()
    import benchmark.mma.confidence as conf
    conf._cosine = lambda a, b: float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)) if len(a) and len(b) else 0.0


def run_episode(args):
    bench, ep_name, readers = args
    setup()
    import io
    import contextlib
    import mm
    from benchmark.mma.system import MMASystem
    eps = [e for e in C.load_bench(bench) if e.name == ep_name]
    ep = eps[0]
    caps = mm.build_captions([ep], "short")
    cfg = yaml.safe_load(open(HARNESS / "config/methods/mma_gemini.yaml"))
    if os.getenv("P16_MMA_TOPK"):                       # retrieval budget in memory entries (turns)
        cfg["retrieval_top_k"] = int(os.environ["P16_MMA_TOPK"])
    os.environ.setdefault("GEMINI_API_KEY", C.gemini_key())
    outp = C.out_path(bench, SYSTEM)
    done = C.done_keys(outp)
    todo = [q for q in ep.qas if any((q.qid, rd) not in done for rd in readers)]
    if not todo:
        return []
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        system = MMASystem(cfg)
        ds = EpisodeDataset(ep)
        system.process_all_sessions(ds)
    build_s = time.time() - t0
    out = []
    from concurrent.futures import ThreadPoolExecutor

    def one(qa):
        recs = []
        try:
            t_r = time.perf_counter()
            entries = system._retrieve(qa.question, query_image_paths=[str(qa.question_image)] if qa.question_image else None)
            ctx = system._format_context(entries)
            t_ret = time.perf_counter() - t_r
            ranked = list(dict.fromkeys(ep.rounds[ds.idx_of[e.round_id]].rid for e in entries   # MMA's own order
                                        if getattr(e, "round_id", None) in ds.idx_of))
            if C.RETRIEVAL_ONLY:
                return [{"qid": qa.qid, "reader": "none", "system": SYSTEM, "retrieved": ranked,
                         "n_entries": len(entries), "n_images": len({p for e in entries for p in (e.image_paths or [])})}]
            imgs = []
            for e in entries:
                for p in e.image_paths or []:
                    if p not in imgs:
                        imgs.append(p)
            for rd in readers:
                if (qa.qid, rd) in done:
                    continue
                pred, r_cost = C.read_context(SYSTEM, ep, qa, ctx, imgs, rd)
                recs.append({"qid": qa.qid, "reader": rd, "system": SYSTEM, "pred": pred,
                             "score": C.judge(SYSTEM, bench, qa, pred), "n_entries": len(entries),
                             "n_images": len(imgs), "ctx_chars": len(ctx), "usd_query": 0.0, "usd_read": r_cost,
                             "usd_build_share": 0.0, "usd": r_cost, "build_seconds": build_s, "retrieved": ranked,
                             "t_retrieve_s": round(t_ret, 3), "t_read_s": round(getattr(C.TL, "read_s", 0.0), 3),
                             "warm": qa is todo[0]})
        except B.BudgetExceeded as e:
            recs.append({"qid": qa.qid, "error": f"budget: {e}"})
        except Exception as e:  # noqa: BLE001
            recs.append({"qid": qa.qid, "error": repr(e)[:300]})
        return recs

    with ThreadPoolExecutor(1 if C.LATENCY else 6) as ex:
        for recs in ex.map(one, todo):
            out += recs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["memeye", "memlens", "visualmem", "camroll"])
    ap.add_argument("--readers", default="lite,strong")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    B.check(SYSTEM)
    eps = C.load_bench(a.bench)
    outp = C.out_path(a.bench, SYSTEM)
    readers = a.readers.split(",")
    jobs = [(a.bench, e.name, readers) for e in eps]
    with mp.get_context("spawn").Pool(a.procs) as pool, open(outp, "a") as f:
        for i, recs in enumerate(pool.imap_unordered(run_episode, jobs), 1):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"  episodes done {i}/{len(jobs)} | spent ${B.spent(SYSTEM):.2f}", flush=True)
            if any("budget" in r.get("error", "") for r in recs):
                print("STOPPED at the budget cap", flush=True)
                pool.terminate()
                break


if __name__ == "__main__":
    main()
