"""Phase-2 runner: selector | reader-mode (see src/mm2.py), with total usage over all stages.

python scripts/run_mm2.py --reader gemini-2.5-flash-lite --cond "toc:20|read:medium" --cond "rag:20|zoom:medium"
Outputs results/mm2/<bench>_<split>/<slug>__<reader>.jsonl and a line in results/mm2/summary.jsonl
"""
import argparse
import collections
import json
import sys
import time
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
from gem import OR_PRICES, PRICES, pmap, spend  # noqa: E402
from run_mm import episodes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KEY = "cap:dense"


def usd(usages, model):
    if model in OR_PRICES:
        return sum((u.get("cost_usd") if u.get("cost_usd") is not None else
                    (u["prompt"] * OR_PRICES[model]["in"] + u["out"] * OR_PRICES[model]["out"]) / 1e6) for u in usages)
    p = PRICES[model]
    return sum(((u["prompt"] - u["cached"]) * p["in"] + u["cached"] * p["cached"] + (u["out"] + u["thoughts"]) * p["out"])
               for u in usages) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="memeye")
    ap.add_argument("--split", default="all")
    ap.add_argument("--cond", action="append", required=True)
    ap.add_argument("--reader", default="gemini-2.5-flash-lite")
    ap.add_argument("--thinking", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--qids", default=None)
    ap.add_argument("--grow", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--qids_file", default=None, help="JSON list of question ids (MemLens subsets)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--seed", type=int, default=None, help="decoding seed for answer calls (replicates)")
    ap.add_argument("--xcache", action="store_true", help="serve static per-user index prefixes from explicit context caches")
    ap.add_argument("--fresh_sel", action="store_true", help="new QSS selection draw: bypass the local cache for selection calls")
    ap.add_argument("--snippet", type=int, default=200, help="user-message characters per index line")
    ap.add_argument("--snippet_mode", default="head", help="head | query (head + question-matched window)")
    ap.add_argument("--compact", default="", help="index compression flags: comma list of noassist, shortid, telegraph")
    ap.add_argument("--index", default="lines", help="index layout: lines (one line per round) | iac (image-anchored)")
    ap.add_argument("--iac_kw", type=int, default=12, help="IAC: key words per session header")
    ap.add_argument("--iac_b", type=int, default=20, help="IAC: user words per image-round line")
    ap.add_argument("--mask_p", type=float, default=0.0, help="visibility manipulation: mask question words in index lines")
    ap.add_argument("--toc_mode", default="full", help="index content ablation: full | text (no image descriptions) | cap")
    ap.add_argument("--key", default=None, help="retrieval key (default cap:dense; MemLens uses cap:short)")
    ap.add_argument("--judge2", default=None, help="second judge model stored as score2 (e.g. gemini-3.8-flash)")
    ap.add_argument("--usage_src", default="toc38-40", help="whose resolutions are written back (select_eval file)")
    ap.add_argument("--usage_frac", type=float, default=1.0, help="fraction of past questions written back")
    ap.add_argument("--usage", default=None, help="resolution write-back from strong-selector resolutions: loo | online")
    ap.add_argument("--sel_thinking", default=None, help="thinking for READER-model selection calls (default: off for "
                    "gemini-2.5, i.e. --thinking only affects answering)")
    a = ap.parse_args()
    mm2.SEED = a.seed
    mm2.XCACHE = a.xcache
    if a.fresh_sel:
        import gem as _gem
        _gem.FRESH_TAGS = ["p2:toc_select"]
    mm2.SNIPPET = a.snippet
    mm2.SNIPPET_MODE = a.snippet_mode
    _cf = set(x for x in a.compact.split(",") if x)
    mm2.NO_ASSIST, mm2.SHORT_IDS, mm2.TELEGRAPH = "noassist" in _cf, "shortid" in _cf, "telegraph" in _cf
    mm2.MASK_P = a.mask_p
    mm2.TOC_MODE = a.toc_mode
    mm2.INDEX_MODE, mm2.IAC_KW, mm2.IAC_B = a.index, a.iac_kw, a.iac_b
    th = (int(a.thinking or 0) if a.reader.startswith("gemini-2.5") else (a.thinking or "low"))
    sth_r = (int(a.sel_thinking or 0) if a.reader.startswith("gemini-2.5") else (a.sel_thinking or "low"))
    if a.qids_file:
        a.tasks = json.load(open(a.qids_file))
    eps = episodes(a.bench, a.split, a.tasks)
    if a.limit:
        for e in eps:
            e.qas = e.qas[: a.limit]
    if a.grow:
        import scale
        eps = [scale.grow(e, a.grow) for e in eps]
    if a.qids:
        keep = set(json.load(open(a.qids))["qids"])
        for e in eps:
            e.qas = [q for q in e.qas if q.qid in keep]
        eps = [e for e in eps if e.qas]
    global KEY
    KEY = a.key or ("cap:short" if a.bench in ("memlens", "wma", "visualmem", "camroll") else KEY)
    caps = {s: mm.build_captions(eps, s) for s in (["short"] if a.bench in ("memlens", "wma", "visualmem", "camroll") else ["short", "dense"])}
    if a.bench in ("memlens", "wma", "visualmem", "camroll"):
        caps["dense"] = caps["short"]
    idx = {e.name: mm.RoundIndex(e, caps, [KEY]) for e in eps}
    if a.snippet_mode == "gist":
        mm2.build_gists(eps, caps["short"])
    if a.usage == "anticip":
        mm2.USAGE = mm2.build_anticipated_usage(eps, str(ROOT / "results/p6/anticipated_memeye.json"))
        mm2.USE_USAGE = True
    elif a.usage:
        T = {r["qid"]: r["ids"] for r in json.load(open(ROOT / f"results/mm2/select_eval_{a.usage_src}.json"))["per_q"]}
        mm2.USAGE = mm2.build_usage(eps, T, mode=a.usage, frac=a.usage_frac)
        mm2.USE_USAGE = True
    out = Path(a.outdir) if a.outdir else ROOT / (f"results/mm2/{a.bench}_{a.split}" + (f"_dx{a.grow}" if a.grow else "")
                                                  + (f"_{Path(a.qids_file).stem}" if a.qids_file else ""))
    out.mkdir(parents=True, exist_ok=True)

    _groups = {}
    _gran = {}
    import threading
    _gran_lock = threading.Lock()

    def granular_for(e):
        with _gran_lock:
            if e.name not in _gran:
                from granular_index import GranularIndex
                _gran[e.name] = GranularIndex(e, caps["short"])
            return _gran[e.name]

    def groups_for(e):
        """write-time group index, built once per history; its cost is reported separately (amortised)."""
        if e.name not in _groups:
            g, u = mm2.build_groups(e, caps["short"])
            _groups[e.name] = (g, u)
            json.dump({"groups": [{"name": x["name"], "rounds": [e.rounds[i].rid for i in x["rounds"]]} for x in g],
                       "usage": u, "usd": usd([u], "gemini-3.8-flash")},
                      open(out / f"groups_{e.name}.json", "w"), indent=0)
        return _groups[e.name]

    for cond in a.cond:
        t0 = time.time()
        sel, rmode = cond.split("|")
        if sel.startswith("grp"):
            pmap(groups_for, eps, workers=8)

        def run(job):
            try:
                return run1(job)
            except Exception as ex:  # noqa: BLE001  one failed question must not kill the run
                e, qa = job
                return {"qid": qa.qid, "task": qa.task, "qtype": qa.qtype, "x": qa.x, "y": qa.y, "cond": cond,
                        "reader": a.reader, "error": f"{type(ex).__name__}: {str(ex)[:200]}", "score": 0.0,
                        "retrieved": [], "n_loaded": 0, "n_calls": 0, "in_tok": 0, "cached_tok": 0, "img_tok": 0,
                        "out_tok": 0, "usd": 0.0, "question": qa.question, "gold": qa.answer, "pred": ""}

        def run1(job):
            e, qa = job
            _t0 = time.perf_counter()               # latency: selection = everything before reading (incl. embedding)
            usages, extra, sel_usages = [], {}, []  # sel_usages: (model, usage) for selector calls
            sp = sel.split(":")
            if sp[0] == "fc":
                ids = list(range(len(e.rounds)))
            elif sp[0] == "oracle":
                r2i = e.rid2idx
                ids = sorted({r2i[c] for c in qa.clue if c in r2i})
            elif sp[0] == "rag":
                ids = idx[e.name].topk(qa, KEY, int(sp[1]))
            elif sp[0] == "gx":
                ids = mm2.graph_expand(idx[e.name], qa, KEY, int(sp[1]), int(sp[2]), int(sp[3]))
            elif sp[0] in ("toc", "tocu", "toc38", "tocu38") or sp[0].startswith("toc@"):
                if sp[0].startswith("toc@"):
                    smodel, sth = resolve(sp[0].split("@")[1])
                else:
                    smodel, sth = (a.reader, sth_r) if sp[0] not in ("toc38", "tocu38") else ("gemini-3.8-flash", "low")
                ids, u = mm2.toc_select(e, qa, caps["short"], int(sp[1]), smodel, sth)
                sel_usages.append((smodel, u))
                extra["n_toc_selected"] = len(ids)
                if sp[0] in ("tocu", "tocu38"):
                    for i in idx[e.name].topk(qa, KEY, int(sp[2])):
                        if i not in ids:
                            ids.append(i)
            elif sp[0] in ("gran", "granf"):   # gran:k sentence-granular max-sim top-k; granf:k fused with round cosine
                gi = granular_for(e)
                q_ = qa.question.split("\n", 1)[1] if qa.question.startswith("Question date:") else qa.question
                sg = gi.round_scores(q_)
                if sp[0] == "granf":
                    rr = np.zeros(len(e.rounds))
                    for o in (np.argsort(-sg), np.argsort(-idx[e.name].scores(qa, KEY))):
                        rr[o] += 1.0 / (60 + np.arange(len(o)))
                    sg = rr
                ids = [int(i) for i in np.argsort(-sg)[: int(sp[1])]]
            elif sp[0] == "ctoc38":   # ctoc38:k:N  cascade: lite selection U cosine top-k -> 3.8 selects within the pool
                lite_ids, u1 = mm2.toc_select(e, qa, caps["short"], int(sp[2]), "gemini-2.5-flash-lite", 0)
                pool = list(dict.fromkeys(lite_ids + idx[e.name].topk(qa, KEY, int(sp[1]))))
                ids, u2 = mm2.toc_select(e, qa, caps["short"], int(sp[2]), "gemini-3.8-flash", "low", subset=sorted(pool))
                sel_usages += [("gemini-2.5-flash-lite", u1), ("gemini-3.8-flash", u2)]
                extra["pool_size"] = len(pool)
            elif sp[0] in ("toc38s", "toc38n"):   # Phase 5: QSS set completed within sessions (s: whole session; n: +-w)
                qs, u = mm2.toc_select(e, qa, caps["short"], int(sp[1]), "gemini-3.8-flash", "low")
                sel_usages.append(("gemini-3.8-flash", u))
                extra["n_toc_selected"] = len(qs)
                ids = mm2.session_closure(e, qs, None if sp[0] == "toc38s" else int(sp[2]))
            elif sp[0] == "sess":   # sess:k  all rounds of the sessions that contain the cosine top-k rounds
                top = idx[e.name].topk(qa, KEY, int(sp[1]))
                ids = [j for u in mm2.session_units(e, top) for j in u]
            elif sp[0] == "utoc38":   # utoc38:N:k  (post hoc) union of the QSS set and the cosine top-k
                qs, u = mm2.toc_select(e, qa, caps["short"], int(sp[1]), "gemini-3.8-flash", "low")
                sel_usages.append(("gemini-3.8-flash", u))
                ids = list(dict.fromkeys(qs + idx[e.name].topk(qa, KEY, int(sp[2]))))
                extra["n_toc_selected"] = len(qs)
            elif sp[0].startswith("grade"):   # grade38:P:t | grade@alias:P:t  graded relevance + dynamic cut-off
                gm = ALIAS[sp[0].split("@")[1]].split("#")[0] if "@" in sp[0] else "gemini-3.8-flash"
                pre = idx[e.name].topk(qa, KEY, int(sp[1]))
                ids, u, _ = mm2.grade_select(e, qa, caps["short"], pre, gm, th_for(gm), threshold=int(sp[2]))
                sel_usages.append((gm, u))
                extra["n_grade_selected"] = len(ids)
            elif sp[0] == "grp":   # grp:N:k  write-time groups (3.8, once per history) + query-time group pick (reader)
                g, gu = groups_for(e)
                fb = idx[e.name].topk(qa, KEY, int(sp[2]))
                ids, u = mm2.group_select(e, qa, g, int(sp[1]), a.reader, sth_r, fallback=fb)
                sel_usages.append((a.reader, u))
                extra["n_groups"] = len(g)
            elif sp[0] in ("ptoc38", "ptoc"):   # ptoc38:P:N  cosine prefilter to P rounds, then TOC selection
                smodel, sth = ("gemini-3.8-flash", "low") if sp[0] == "ptoc38" else (a.reader, sth_r)
                pre = idx[e.name].topk(qa, KEY, int(sp[1]))
                ids, u = mm2.toc_select(e, qa, caps["short"], int(sp[2]), smodel, sth, subset=pre)
                sel_usages.append((smodel, u))
                extra["n_toc_selected"] = len(ids)
            elif sp[0] == "scu":   # scu:K:V:N  K temperature samples of the READER-model TOC selection, >= V votes
                ids, us_ = mm2.toc_select_sampled(e, qa, caps["short"], int(sp[3]), a.reader, sth_r, int(sp[1]), int(sp[2]))
                sel_usages += [(a.reader, u) for u in us_]
                extra["n_scu_selected"] = len(ids)
            elif sp[0] == "q2q":   # q2q:m:k  evidence sets of the m most similar resolved past questions + cosine top-k
                r2i = e.rid2idx
                past = mm2.USAGE.get(qa.qid, [])[: int(sp[1])]
                ids = list(dict.fromkeys([r2i[x] for p_ in past for x in p_["rids"] if x in r2i]
                                         + idx[e.name].topk(qa, KEY, int(sp[2]))))
            elif sp[0] == "pscu":  # pscu:K:V:P:N  sampled reader-model selection over a cosine-prefiltered pool of P rounds
                pre = idx[e.name].topk(qa, KEY, int(sp[3]))
                ids, us_ = mm2.toc_select_sampled(e, qa, caps["short"], int(sp[4]), a.reader, sth_r, int(sp[1]), int(sp[2]),
                                                  subset=pre)
                sel_usages += [(a.reader, u) for u in us_]
                extra["n_scu_selected"] = len(ids)
            elif sp[0] in ("scan", "scan38"):   # scan:N:chunk
                smodel, sth = (a.reader, sth_r) if sp[0] == "scan" else ("gemini-3.8-flash", "low")
                ids, us_ = mm2.scan_select(e, qa, caps["short"], int(sp[1]), int(sp[2]), smodel, sth,
                                           order_scores=idx[e.name].scores(qa, KEY))
                sel_usages += [(smodel, u) for u in us_]
                extra["n_scan_selected"] = len(ids)
            else:
                raise ValueError(sel)
            _t1 = time.perf_counter()
            mode, res = rmode.split(":")
            with_toc = mode.endswith("+toc")
            base = mode.replace("+toc", "")
            if base == "hint":
                pred, us = mm2.hint_read(e, qa, ids, res, caps, a.reader, th, tag=f"p2:{cond}")
                extra["n_hint"] = len(ids)
                ids = list(range(len(e.rounds)))   # the reader sees everything
            elif base in ("readb", "readc", "read0"):   # raw + caption / caption only / no visual information at all
                pred, us = mm2.read(e, qa, ids, res, caps, a.reader, th, with_toc, tag=f"p2:{cond}",
                                    rep={"readb": "both", "readc": "cap", "read0": "none"}[base])
            elif base == "read":
                pred, us = mm2.read(e, qa, ids, res, caps, a.reader, th, with_toc, tag=f"p2:{cond}")
            elif base in ("eva", "evar", "evau", "evaur") or base.startswith("readn"):   # per-unit verify + extract
                rk = int(base[5:]) if base.startswith("readn") and base[5:].isdigit() else None   # readn<k>: raw top-k only
                wide = None
                if base.startswith("readnw"):   # Phase 9: notes for the selected set + raw rounds of (set U cosine top-K)
                    wide = list(dict.fromkeys(ids + idx[e.name].topk(qa, KEY, int(base[6:]))))
                    extra["n_wide_raw"] = len(wide)
                pred, us, info = mm2.eva_read(e, qa, ids, res, caps, a.reader, th, tag=f"p5:{cond}",
                                              with_raw=base.endswith("r"), fallback_idxs=idx[e.name].topk(qa, KEY, 10),
                                              unit="round" if base.startswith("evau") or base.startswith("readn") else "session",
                                              all_raw=base.startswith("readn"), raw_idxs=(wide if wide else (ids[:rk] if rk else None)))
                sel_usages += [(mm2.EVA_MAP_MODEL, u) for u in info.pop("map_usages")]
                extra.update(info)
            elif base in ("zoom2", "zoom2f"):
                pred, us, zooms = mm2.zoom2_read(e, qa, ids, res, caps, a.reader, th, with_toc, forced=(base == "zoom2f"),
                                                 tag=f"p2:{cond}")
                extra["zooms"] = zooms
            elif base in ("zoom3", "zoom3f"):
                pred, us, zooms = mm2.zoom3_read(e, qa, ids, res, caps, a.reader, th, with_toc, forced=(base == "zoom3f"),
                                                 tag=f"p2:{cond}")
                extra["zooms"] = zooms
            elif base == "zoomb":
                pred, us, zooms = mm2.blind_zoom_read(e, qa, ids, res, caps, a.reader, th, idx[e.name], KEY, with_toc,
                                                      tag=f"p2:{cond}")
                extra["zooms"] = zooms
            elif base in ("zoom", "zoomf"):
                pred, us, zooms = mm2.zoom_read(e, qa, ids, res, caps, a.reader, th, with_toc, tag=f"p2:{cond}",
                                                forced=(base == "zoomf"))
                extra["zooms"] = zooms
            elif base == "mr":
                pred, us, nn = mm2.map_reduce(e, qa, ids, res, caps, a.reader, th, with_toc, tag=f"p2:{cond}")
                extra["n_notes"] = nn
            else:
                raise ValueError(rmode)
            _t2 = time.perf_counter()
            extra["t_select_s"], extra["t_read_s"] = round(_t1 - _t0, 3), round(_t2 - _t1, 3)
            usages += us
            sel_usd = sum(usd([u], m) for m, u in sel_usages)
            usages_all = usages + [u for _, u in sel_usages]
            if a.bench == "memlens":
                j = mm2.memlens_judge(qa, pred)
                if a.judge2:
                    extra["score2"] = mm2.memlens_judge(qa, pred, model=a.judge2)["score"]
            elif a.bench in ("visualmem", "camroll"):   # multiple choice: exact match of the option letter
                from data_visualmem import mc_score
                j = {"score": mc_score(pred, qa.answer), "reason": "exact letter match"}
            else:
                j = mm.judge(qa.question, qa.answer, pred)
            return {"qid": qa.qid, "task": qa.task, "qtype": qa.qtype, "x": qa.x, "y": qa.y, "cond": cond,
                    "reader": a.reader, "retrieved": [e.rounds[i].rid for i in ids], "n_loaded": len(ids),
                    **mm.clue_stats(e, qa, ids), "question": qa.question, "gold": qa.answer, "pred": pred,
                    "score": j["score"], "judge_reason": j["reason"], "n_calls": len(usages_all),
                    "in_tok": sum(u["prompt"] for u in usages_all), "cached_tok": sum(u["cached"] for u in usages_all),
                    "img_tok": sum(u["img_tokens"] for u in usages_all),
                    "out_tok": sum(u["out"] + u["thoughts"] for u in usages_all),
                    "usd": usd(usages, a.reader) + sel_usd, "usd_select": sel_usd, **extra}

        jobs = [(e, qa) for e in eps for qa in e.qas]
        first = {}
        for i, (e, _) in enumerate(jobs):
            first.setdefault(e.name, i)
        fi = list(first.values())
        warm = pmap(run, [jobs[i] for i in fi], workers=a.workers)
        rest_i = [i for i in range(len(jobs)) if i not in set(fi)]
        rest = pmap(run, [jobs[i] for i in rest_i], workers=a.workers, desc=cond)
        recs = [None] * len(jobs)
        for i, r in zip(fi + rest_i, warm + rest):
            recs[i] = r
        for i in fi:
            recs[i]["warm"] = True                  # first question of a history: may include cache creation
        slug = cond.replace("|", "__").replace(":", "-") + f"__{a.reader}" + (f"__think{a.thinking}" if a.thinking else "") + (f"__selthink{a.sel_thinking}" if a.sel_thinking else "") + (f"__snip{a.snippet}" if a.snippet != 200 else "") + ("__qsnip" if a.snippet_mode == "query" else "") + ("__gist" if a.snippet_mode == "gist" else "") + (f"__c{a.compact.replace(',', '+')}" if a.compact else "") + (f"__iac{a.iac_kw}-{a.iac_b}" if a.index == "iac" else "") + (f"__mask{a.mask_p}" if a.mask_p else "") + (f"__toc{a.toc_mode}" if a.toc_mode != "full" else "") + (f"__usage{a.usage}" if a.usage else "") + (f"-src{a.usage_src}" if a.usage and a.usage_src != "toc38-40" else "") + (f"-frac{a.usage_frac}" if a.usage and a.usage_frac < 1 else "") + (f"__seed{a.seed}" if a.seed is not None else "") + ("__xcache" if a.xcache else "") + a.tag
        with open(out / f"{slug}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by = collections.defaultdict(list)
        for r in recs:
            by[r["x"]].append(r["score"])
            by[r["y"]].append(r["score"])
        s = {"bench": a.bench, "split": a.split, "grow": a.grow, "cond": cond, "reader": a.reader, "think": a.thinking,
             "n": len(recs), "score": round(float(np.mean([r["score"] for r in recs])), 4),
             "by": {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items()) if k},
             "full": round(float(np.mean([r["full"] for r in recs if r.get("n_gold")] or [0])), 3),
             "cov": round(float(np.mean([r["cov"] for r in recs if r.get("n_gold")] or [0])), 3),
             "n_loaded": round(float(np.mean([r["n_loaded"] for r in recs])), 1),
             "in_tok": int(np.mean([r["in_tok"] for r in recs])), "usd_1k": round(1000 * float(np.mean([r["usd"] for r in recs])), 3),
             "errors": sum(1 for r in recs if r.get("error")),
             "secs": round(time.time() - t0, 1), "spend_total": spend().get("TOTAL")}
        print(json.dumps(s))
        with open(ROOT / "results/mm2/summary.jsonl", "a") as f:
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
