"""Phase 12: V-Mem (Kang et al., 2026; arXiv 2608.01543; github.com/Dingyi-Kang/V-Mem) on our benchmarks.
  .venv_bl/bin/python scripts/p12_vmem.py --bench memeye
V-Mem keeps the raw conversation: rounds hold user, assistant (caption folded in, its default) and image units,
indexed with MiniLM + BM25 and SigLIP2 image vectors; retrieval routes by query/target modality (its 2x2 routing,
hypothetical captions for text->image queries, image keywords for image+text queries), top-12 rounds with +/-1
neighbours. We run its released retrieval code unchanged and read the returned raw rounds with our readers, exactly
as RAG@40 and VGS are read. Two substitutions: SigLIP2 (the same checkpoint) runs in-process via transformers instead
of a vLLM server -- here replaced by gemini-embedding-2, see encode_image -- and its anchor LLM is
gemini-2.5-flash-lite (the counterpart of the gpt-4o-mini backbone in its paper).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VMEM = ROOT / "baselines/V-Mem"
sys.path.insert(0, str(ROOT / "scripts"))
import p12_common as C  # noqa: E402
import p12_budget as B  # noqa: E402

SYSTEM = "vmem"
ANCHOR_MODEL = "gemini-2.5-flash-lite"
IMG_EMB = "gemini-embedding-2"
_emb_lock = threading.Lock()


def _charge_embeddings(items):
    """Charge newly embedded items (images ~$0.0002 each, text by length) to the ledger."""
    import gem
    norm = [[it] if isinstance(it, (str, gem.Img)) else list(it) for it in items]
    keys = [gem._emb_item_key(IMG_EMB, None, it) for it in norm]
    have = gem._emb_cache.get_many(list(set(keys)))
    new = [it for k, it in zip(keys, norm) if k not in have]
    cost = sum(0.0002 if isinstance(p, gem.Img) else len(p) / 4 * 0.15e-6 for it in new for p in it)
    B.charge(SYSTEM, cost, "build:image-embeddings")


def encode_image(path):
    """V-Mem's image lane encoder. SigLIP2 (1.5 GB) could not be downloaded here (~80 KB/s), so the image lane uses
    gemini-embedding-2, a joint text-image space like SigLIP; the lane serves only questions that carry an image."""
    import gem
    item = gem.Img(str(path))
    with _emb_lock:
        _charge_embeddings([item])
        return gem.embed([item], model=IMG_EMB)[0].astype(np.float32)


def encode_text(text):
    import gem
    with _emb_lock:
        return gem.embed([text], model=IMG_EMB)[0].astype(np.float32)


def save_siglip_cache():
    pass


def setup():
    os.environ["OPENAI_API_KEY"] = C.gemini_key()
    os.environ["OPENAI_BASE_URL"] = C.GEMINI_OAI
    sys.path.insert(0, str(VMEM))
    import torch
    from transformers import PreTrainedModel  # noqa: F401  (resolve transformers' lazy import before the patch)
    import sentence_transformers  # noqa: F401
    torch.backends.mps.is_available = lambda: False      # keep MiniLM on the CPU (memory)
    import visual_embeddings
    visual_embeddings.encode_image = encode_image
    visual_embeddings.encode_text = encode_text
    visual_embeddings.encode_images = lambda paths: (list(paths), np.stack([encode_image(p) for p in paths]))
    import gem
    gem.embed  # noqa: B018  (gem is imported in this environment)
    C.patch_openai(SYSTEM)


def build_memory(ep, caps):
    """Our episode -> V-Mem Rounds/Units (its Mem-Gallery loader with fold=True), then its three indexes."""
    from model import Round, Unit
    from indexes import VectorIndex, KeywordIndex, VisualIndex
    from memory import Memory
    import embeddings
    rounds, idx_of, sess_no, pos = [], {}, {}, {}
    for i, r in enumerate(ep.rounds):
        sid = sess_no.setdefault(r.sid, len(sess_no) + 1)
        pos[sid] = pos.get(sid, 0) + 1
        rid = f"S{sid}:R{pos[sid]}"
        idx_of[rid] = i
        units = []
        if r.user:
            units.append(Unit(unit_id=f"{rid}::user", round_id=rid, kind="user", text=r.user))
        capt = [caps.get(p, "") for p in r.images]
        if r.assistant or capt:
            folded = (r.assistant + "".join(f"\n[Image: {c}]" for c in capt if c)).strip()
            units.append(Unit(unit_id=f"{rid}::assistant", round_id=rid, kind="assistant", text=folded))
        for k, p in enumerate(r.images):
            units.append(Unit(unit_id=f"{rid}::img::{k}", round_id=rid, kind="image", text=capt[k],
                              image_id=f"{rid}_img{k}", url=str(p)))
        if units:
            rounds.append(Round(round_id=rid, session_id=sid, index=pos[sid], date=r.date or "", units=units))
    vec_ids, texts, metas = [], [], []
    kindex = KeywordIndex()
    for rnd in rounds:
        for u in rnd.units:
            vec_ids.append(u.unit_id)
            texts.append(u.index_text)
            metas.append({"round_id": u.round_id, "kind": u.kind, "image_id": u.image_id})
            kindex.add(u.unit_id, u.index_text)
    vindex = VectorIndex()
    vindex.build(vec_ids, embeddings.encode(texts), metas)
    img_units = [u for r in rounds for u in r.units if u.kind == "image" and u.url]
    vvisual = None
    if img_units and any(q.question_image for q in ep.qas):     # the image lane only serves image-bearing questions
        import gem
        items = [gem.Img(u.url) for u in img_units]
        _charge_embeddings(items)
        vecs = gem.embed(items, model=IMG_EMB, workers=4)
        vvisual = VisualIndex()
        vvisual.build([u.unit_id for u in img_units], [v.astype(np.float32) for v in vecs],
                      [{"round_id": u.round_id, "kind": "image", "image_id": u.image_id} for u in img_units])
    return Memory({r.round_id: r for r in rounds}, vindex, kindex, vvisual), idx_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["memeye", "memlens", "visualmem", "camroll"])
    ap.add_argument("--readers", default="lite,strong")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit_eps", type=int, default=0)
    a = ap.parse_args()
    setup()
    import mm
    from llm import LLM
    import run_memgallery_final as RMF
    from run_memgallery_final import route_and_retrieve
    if os.getenv("P16_VMEM_TOPK"):                      # retrieval budget: top-k rounds before its +/-1 neighbour windows
        _retrieve, _k = RMF.retrieve, int(os.environ["P16_VMEM_TOPK"])
        RMF.retrieve = lambda mem, q, top_k=12, **kw: _retrieve(mem, q, top_k=_k, **kw)
    llm = LLM(model=ANCHOR_MODEL)
    eps = C.load_bench(a.bench)[: a.limit_eps or None]
    caps = mm.build_captions(eps, "short")
    outp = C.out_path(a.bench, SYSTEM)
    done = C.done_keys(outp)
    readers = a.readers.split(",")
    lock = threading.Lock()
    from concurrent.futures import ThreadPoolExecutor
    n_done = 0
    for ei, ep in enumerate(eps):
        todo = [q for q in ep.qas if any((q.qid, rd) not in done for rd in readers)]
        if not todo:
            continue
        t0 = time.time()
        mem, idx_of = build_memory(ep, caps)
        build_s = time.time() - t0

        def one(qa):
            C.STATE["stage"] = "query"
            qimg = str(qa.question_image) if qa.question_image else ""
            qcap = caps.get(qa.question_image, "") if qa.question_image else ""
            tr = {}
            spent0 = B.spent(SYSTEM, "query")
            t_r = time.perf_counter()
            rids = route_and_retrieve(mem, llm, qa.question, qcap, qimg, tr)
            rids = [r if isinstance(r, str) else getattr(r, "round_id", r) for r in rids]
            t_ret = time.perf_counter() - t_r
            ranked = [ep.rounds[idx_of[r]].rid for r in dict.fromkeys(rids) if r in idx_of]   # V-Mem's own order
            idxs = sorted({idx_of[r] for r in rids if r in idx_of})
            q_cost = max(0.0, B.spent(SYSTEM, "query") - spent0)
            if C.RETRIEVAL_ONLY:
                return [{"qid": qa.qid, "reader": "none", "system": SYSTEM, "retrieved": ranked, "n_rounds": len(idxs),
                         "route": tr.get("route"), "usd_query": q_cost}]
            recs = []
            for rd in readers:
                if (qa.qid, rd) in done:
                    continue
                pred, r_cost = C.read_rounds(SYSTEM, ep, qa, idxs, rd, caps)
                recs.append({"qid": qa.qid, "reader": rd, "system": SYSTEM, "pred": pred,
                             "score": C.judge(SYSTEM, a.bench, qa, pred), "route": tr.get("route"),
                             "n_rounds": len(idxs), "usd_query": q_cost, "usd_read": r_cost, "usd_build_share": 0.0,
                             "usd": q_cost + r_cost, "build_seconds": build_s, "retrieved": ranked,
                             "t_retrieve_s": round(t_ret, 3), "t_read_s": round(getattr(C.TL, "read_s", 0.0), 3),
                             "warm": qa is todo[0]})
            return recs

        with ThreadPoolExecutor(a.workers) as ex, open(outp, "a") as f:
            for recs in ex.map(one, todo):
                with lock:
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
                    f.flush()
        save_siglip_cache()
        n_done += 1
        print(f"  [{ei + 1}/{len(eps)}] {ep.name}: {len(todo)} questions, build {build_s:.0f}s, "
              f"spent ${B.spent(SYSTEM):.2f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except B.BudgetExceeded as e:
        save_siglip_cache()
        print("STOPPED:", e, flush=True)
