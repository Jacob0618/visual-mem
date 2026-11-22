"""Selection-only evaluation of set-level recall (no answering): full-set recall, coverage and #rounds loaded,
compared with cosine top-k at the SAME per-question size. Writes results/mm2/select_eval_<name>.json

python scripts/select_eval.py --sel toc:40 --sel toc38:40 --sel scan:40:30 --sel scan38:40:30
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm  # noqa: E402
import mm2  # noqa: E402
ALIAS = {"3.8f": "gemini-3.8-flash", "3.5fl": "gemini-3.5-flash-lite", "3.1fl": "gemini-3.1-flash-lite",
         "2.5fl": "gemini-2.5-flash-lite", "2.5f": "gemini-2.5-flash", "3.6f": "gemini-3.6-flash",
         "2.5flt": "gemini-2.5-flash-lite#think", "2.5ft": "gemini-2.5-flash#think", "3.8nt": "gemini-3.8-flash#nothink", "qwen": "qwen/qwen3-235b-a22b-2507",
         "gpt5m": "openai/gpt-5-mini", "qwenvl": "qwen/qwen3-vl-235b-a22b-instruct", "gpt41m": "openai/gpt-4.1-mini",
         "qwent": "qwen/qwen3-235b-a22b-thinking-2507", "ds": "deepseek/deepseek-chat-v3.1", "gpt5": "openai/gpt-5"}


def th_for(m):
    return 0 if m.startswith("gemini-2.5") else ("low" if m.startswith("gemini") else None)


def resolve(alias):
    """'gemini-2.5-flash-lite#think' -> (model, thinking budget 2048)"""
    m = ALIAS[alias]
    if m.endswith("#nothink"):
        return m.split("#")[0], 0
    return (m.split("#")[0], 2048) if m.endswith("#think") else (m, th_for(m))
from gem import pmap  # noqa: E402
from run_mm import episodes  # noqa: E402
from run_mm2 import usd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KEY = "cap:dense"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", action="append", required=True)
    ap.add_argument("--reader", default="gemini-2.5-flash-lite")
    ap.add_argument("--grow", type=int, default=0)
    ap.add_argument("--qids", default=None)
    ap.add_argument("--toc_mode", default="full")
    ap.add_argument("--bench", default="memeye")
    ap.add_argument("--snippet", type=int, default=200, help="user-message characters per index line")
    ap.add_argument("--snippet_mode", default="head", help="head | query (head + question-matched window)")
    ap.add_argument("--compact", default="", help="index compression flags: comma list of noassist, shortid, telegraph")
    ap.add_argument("--index", default="lines", help="index layout: lines (one line per round) | iac (image-anchored)")
    ap.add_argument("--iac_kw", type=int, default=12, help="IAC: key words per session header")
    ap.add_argument("--iac_b", type=int, default=20, help="IAC: user words per image-round line")
    ap.add_argument("--mask_p", type=float, default=0.0, help="visibility manipulation: mask question words in index lines")
    ap.add_argument("--length", default="32k", help="MemLens context length")
    ap.add_argument("--limit_q", type=int, default=0, help="evaluate a fixed random subset of this many questions")
    ap.add_argument("--coords", action="store_true", help="narrative coordinates + session headers in the index")
    ap.add_argument("--aliases", action="store_true", help="write-time alias phrases (results/mm2/aliases) in the index")
    ap.add_argument("--demos", action="store_true", help="in-context distillation: teacher (3.8) selections from other tasks")
    ap.add_argument("--usage_src", default="toc38-40", help="whose resolutions are written back (select_eval file)")
    ap.add_argument("--usage_frac", type=float, default=1.0, help="fraction of past questions written back")
    ap.add_argument("--usage", default=None, help="usage write-back of strong-selector resolutions: loo | online")
    ap.add_argument("--usage_maxsim", type=float, default=0.9)
    a = ap.parse_args()
    mm2.TOC_MODE = a.toc_mode
    mm2.SNIPPET = a.snippet
    mm2.SNIPPET_MODE = a.snippet_mode
    _cf = set(x for x in a.compact.split(",") if x)
    mm2.NO_ASSIST, mm2.SHORT_IDS, mm2.TELEGRAPH = "noassist" in _cf, "shortid" in _cf, "telegraph" in _cf
    mm2.MASK_P = a.mask_p
    mm2.INDEX_MODE, mm2.IAC_KW, mm2.IAC_B = a.index, a.iac_kw, a.iac_b
    mm2.COORDS = a.coords
    if a.aliases:
        for f in (ROOT / "results/mm2/aliases").glob("aliases_*.json"):
            mm2.ALIASES[f.stem[len("aliases_"):]] = json.load(open(f))["aliases"]
        mm2.USE_ALIASES = True
    global KEY
    eps = episodes(a.bench, a.length if a.bench == "memlens" else "all")
    if a.limit_q:
        import random
        allq = sorted(qa.qid for e in eps for qa in e.qas)
        keep_q = set(random.Random(0).sample(allq, min(a.limit_q, len(allq))))
        for e in eps:
            e.qas = [q for q in e.qas if q.qid in keep_q]
        eps = [e for e in eps if e.qas]
    if a.bench in ("memlens", "wma", "visualmem", "camroll"):
        KEY = "cap:short"
    if a.grow:
        import scale
        eps = [scale.grow(e, a.grow) for e in eps]
    if a.qids:
        _j = json.load(open(a.qids))
        keep = set(_j["qids"] if isinstance(_j, dict) else _j)
        for e in eps:
            e.qas = [q for q in e.qas if q.qid in keep]
        eps = [e for e in eps if e.qas]
    caps = {s: mm.build_captions(eps, s) for s in (["short"] if a.bench in ("memlens", "wma", "visualmem", "camroll") else ["short", "dense"])}
    if a.bench in ("memlens", "wma", "visualmem", "camroll"):
        caps["dense"] = caps["short"]
    idx = {e.name: mm.RoundIndex(e, caps, [KEY]) for e in eps}
    if a.snippet_mode == "gist":
        mm2.build_gists(eps, caps["short"])
    if a.demos:
        T = {r["qid"]: r for r in json.load(open(ROOT / "results/mm2/select_eval_toc38-40.json"))["per_q"]}
        W = {r["qid"]: r for r in json.load(open(ROOT / "results/mm2/select_eval_toc-40.json"))["per_q"]}
        for q in T:
            T[q]["weak_full"] = W.get(q, {}).get("full", 1.0)
        mm2.DEMOS = mm2.build_demos(eps, caps["short"], T)
        mm2.USE_DEMOS = True
        print("demo block tokens ~", {k: len(v) // 4 for k, v in mm2.DEMOS.items()})
    if a.usage == "anticip":
        mm2.USAGE = mm2.build_anticipated_usage(eps, str(ROOT / "results/p6/anticipated_memeye.json"))
        mm2.USE_USAGE = True
    elif a.usage:
        T = {r["qid"]: r["ids"] for r in json.load(open(ROOT / f"results/mm2/select_eval_{a.usage_src}.json"))["per_q"]}
        mm2.USAGE = mm2.build_usage(eps, T, mode=a.usage, max_sim=a.usage_maxsim, frac=a.usage_frac)
        mm2.USE_USAGE = True
        print("usage items per question:", round(float(np.mean([len(v) for v in mm2.USAGE.values()])), 1))
    for sel in a.sel:
        sp = sel.split(":")

        def one(job):
            e, qa = job
            r2i = e.rid2idx
            gold = {r2i[c] for c in qa.clue if c in r2i}
            s = idx[e.name].scores(qa, KEY)
            if sp[0] in ("rank38", "rankr38"):   # rank38:P ranking control (cut offline); rankr38 = recall-oriented prompt
                pre = idx[e.name].topk(qa, KEY, int(sp[1]))
                ids, u = mm2.rank_select(e, qa, caps["short"], pre, "gemini-3.8-flash", "low", recall=sp[0] == "rankr38")
                us = [("gemini-3.8-flash", u)]
            elif sp[0].startswith("grade"):   # grade38:P:t | grade@alias:P:t
                m = ALIAS[sp[0].split("@")[1]].split("#")[0] if "@" in sp[0] else "gemini-3.8-flash"
                pre = idx[e.name].topk(qa, KEY, int(sp[1]))
                ids, u, _ = mm2.grade_select(e, qa, caps["short"], pre, m, th_for(m), threshold=int(sp[2]))
                us = [(m, u)]
            elif sp[0] == "q2q":       # q2q:m:k  union of evidence sets of the m most similar past questions + cosine top-k
                past = mm2.USAGE.get(qa.qid, [])[: int(sp[1])]
                ids = [r2i[x] for p_ in past for x in p_["rids"] if x in r2i]
                for i in [int(j) for j in np.argsort(-s)[: int(sp[2])]]:
                    ids.append(i)
                ids = list(dict.fromkeys(ids))
                us = []
            elif sp[0] == "psc":       # psc:C:N  predicate-guided set completion (cheap model)
                ids, uu, _ = mm2.psc_select(e, qa, caps["short"], idx[e.name], KEY, int(sp[1]), int(sp[2]), a.reader, 0)
                us = [(a.reader, u) for u in uu]
            elif sp[0] == "scu":     # scu:K:V:N  K temperature samples of the cheap TOC selector, keep >= V votes
                ids, uu = mm2.toc_select_sampled(e, qa, caps["short"], int(sp[3]), a.reader, 0, int(sp[1]), int(sp[2]))
                us = [(a.reader, u) for u in uu]
            elif sp[0] == "pscu":    # pscu:K:V:P:N  sampled selection over a cosine-prefiltered pool of P rounds
                pre = idx[e.name].topk(qa, KEY, int(sp[3]))
                ids, uu = mm2.toc_select_sampled(e, qa, caps["short"], int(sp[4]), a.reader, 0, int(sp[1]), int(sp[2]),
                                                 subset=pre)
                us = [(a.reader, u) for u in uu]
            elif sp[0] == "tocf":    # tocf:N  count-first TOC selection (cheap model)
                ids, u = mm2.toc_select(e, qa, caps["short"], int(sp[1]), a.reader, 0, system=mm2.TOC_SELECT_CF_SYS,
                                        schema=mm2._SEL_CF_SCHEMA)
                us = [(a.reader, u)]
            elif sp[0] == "ctoc38":   # ctoc38:k:N  cascade: lite selection U cosine top-k -> 3.8 selects within the pool
                lite_ids, u1 = mm2.toc_select(e, qa, caps["short"], int(sp[2]), "gemini-2.5-flash-lite", 0)
                pool = list(dict.fromkeys(lite_ids + idx[e.name].topk(qa, KEY, int(sp[1]))))
                ids, u2 = mm2.toc_select(e, qa, caps["short"], int(sp[2]), "gemini-3.8-flash", "low", subset=sorted(pool))
                us = [("gemini-2.5-flash-lite", u1), ("gemini-3.8-flash", u2)]
            elif sp[0].startswith("ptoc"):
                m = "gemini-3.8-flash" if sp[0] == "ptoc38" else a.reader
                pre = idx[e.name].topk(qa, KEY, int(sp[1]))
                ids, u = mm2.toc_select(e, qa, caps["short"], int(sp[2]), m, th_for(m), subset=pre)
                us = [(m, u)]
            elif sp[0].startswith("toc"):
                m, tth = resolve(sp[0].split("@")[1]) if "@" in sp[0] else (
                    ("gemini-3.8-flash", "low") if sp[0] == "toc38" else (a.reader, 0))
                ids, u = mm2.toc_select(e, qa, caps["short"], int(sp[1]), m, tth)
                us = [(m, u)]
            else:
                m = ALIAS[sp[0].split("@")[1]] if "@" in sp[0] else ("gemini-3.8-flash" if sp[0] == "scan38" else a.reader)
                ids, uu = mm2.scan_select(e, qa, caps["short"], int(sp[1]), int(sp[2]), m, th_for(m), order_scores=s)
                us = [(m, u) for u in uu]
            k = max(1, len(ids))
            rag = set(int(i) for i in np.argsort(-s)[:k])
            return {"qid": qa.qid, "task": qa.task, "ids": [e.rounds[i].rid for i in ids],
                    "gold": sorted(e.rounds[i].rid for i in gold), "n": len(ids), "n_gold": len(gold),
                    "full": float(gold <= set(ids)) if gold else None, "rag_full_same_k": float(gold <= rag) if gold else None,
                    "cov": len(gold & set(ids)) / len(gold) if gold else None, "usd": sum(usd([u], mm_) for mm_, u in us),
                    "in_tok": sum(u["prompt"] for _, u in us)}

        jobs = [(e, qa) for e in eps for qa in e.qas]
        R = pmap(one, jobs, workers=12, desc=sel)
        G = [r for r in R if r["full"] is not None]
        s = {"sel": sel, "bench": a.bench, "length": a.length if a.bench == "memlens" else None, "toc_mode": a.toc_mode, "coords": a.coords, "aliases": a.aliases, "demos": a.demos, "usage": a.usage, "usage_maxsim": a.usage_maxsim if a.usage else None, "usage_src": a.usage_src if a.usage else None, "usage_frac": a.usage_frac if a.usage else None, "grow": a.grow, "n_q": len(R), "mean_loaded": round(float(np.mean([r["n"] for r in R])), 2),
             "full": round(float(np.mean([r["full"] for r in G])), 3),
             "rag_full_same_k": round(float(np.mean([r["rag_full_same_k"] for r in G])), 3),
             "cov": round(float(np.mean([r["cov"] for r in G])), 3),
             "full_ge4": round(float(np.mean([r["full"] for r in G if r["n_gold"] >= 4])), 3),
             "rag_full_ge4": round(float(np.mean([r["rag_full_same_k"] for r in G if r["n_gold"] >= 4])), 3),
             "usd_1k": round(1000 * float(np.mean([r["usd"] for r in R])), 3),
             "in_tok": int(np.mean([r["in_tok"] for r in R]))}
        print(json.dumps(s))
        name = (f"{a.bench}{a.length}_" if a.bench == "memlens" else f"{a.bench}_" if a.bench != "memeye" else "") + (f"{Path(a.qids).stem}_" if a.qids else "") + (f"n{a.limit_q}_" if a.limit_q else "") + sel.replace(":", "-") + (f"_snip{a.snippet}" if a.snippet != 200 else "") + ("_qsnip" if a.snippet_mode == "query" else "") + ("_gist" if a.snippet_mode == "gist" else "") + (f"_c{a.compact.replace(',', '+')}" if a.compact else "") + (f"_iac{a.iac_kw}-{a.iac_b}" if a.index == "iac" else "") + (f"_mask{a.mask_p}" if a.mask_p else "") + (f"_dx{a.grow}" if a.grow else "") + (f"_toc{a.toc_mode}" if a.toc_mode != "full" else "") + ("_coords" if a.coords else "") + ("_aliases" if a.aliases else "") + ("_demos" if a.demos else "") + (f"_usage{a.usage}{a.usage_maxsim}" if a.usage else "") + (f"_src{a.usage_src}" if a.usage and a.usage_src != "toc38-40" else "") + (f"_frac{a.usage_frac}" if a.usage and a.usage_frac < 1 else "")
        json.dump({"summary": s, "per_q": R}, open(ROOT / f"results/mm2/select_eval_{name}.json", "w"))


if __name__ == "__main__":
    main()
