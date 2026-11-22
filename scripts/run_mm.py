"""Run multimodal-memory conditions on MemEye / Mem-Gallery and write per-question records.

Example:
  python scripts/run_mm.py --bench memeye --split dev --cond "fc|raw:high" --cond "inter:10|raw:high"
Condition = "<selection>|<representation>", see src/mm.py.
Outputs: results/mm/<bench>_<split>/<slug>.jsonl (+ a summary line appended to results/mm/summary.jsonl)
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mm  # noqa: E402
from data_mm import MEMEYE_TASKS, load_memeye, load_memgal, memgal_tasks  # noqa: E402
from gem import pmap, spend  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def episodes(bench, split, tasks=None):
    if bench == "memlens":   # split = context length ("32k", "128k"); tasks = optional question ids
        from data_memlens import load_memlens
        return load_memlens(split if split in ("32k", "64k", "128k", "256k") else "32k", tasks)
    if bench == "visualmem":  # VisualMem personas; tasks = optional question ids
        from data_visualmem import load_visualmem
        return load_visualmem(tasks)
    if bench == "camroll":    # CamRoll-yfcc20 camera rolls; tasks = optional question ids
        from data_camroll import load_camroll
        return load_camroll(tasks)
    if bench == "wma":       # WorldMemArena lifelong; tasks = optional question ids
        from data_wma import load_wma
        return load_wma(tasks)
    if bench == "memeye":
        sp = json.load(open(ROOT / "results/memeye_split.json"))
        names = tasks or (sp[split] if split in ("dev", "test") else MEMEYE_TASKS)
        return [load_memeye(t) for t in names]
    names = tasks or memgal_tasks()
    return [load_memgal(t) for t in names]


def parse_sel(sel):
    """'fc' | 'oracle' | '<key>:<k>' | 'fc+<key>:<k>' (all rounds; top-k by key marked for the 'hi' resolution)
    | 'sess+<key>:<k>' (whole sessions of the top-k rounds) | 'nbr<w>+<key>:<k>' (top-k rounds +- w neighbours)."""
    if sel in ("fc", "oracle", "oracle_sess"):
        return sel, None
    if "+" in sel:
        sel = sel.split("+", 1)[1]
    key, k = sel.rsplit(":", 1)
    return key, int(k)


def expand(e, top, sel):
    if sel.startswith("sess+"):
        sids = {e.rounds[i].sid for i in top}
        return [i for i, r in enumerate(e.rounds) if r.sid in sids]
    if sel.startswith("nbr"):
        w = int(sel[3:].split("+")[0])
        return sorted({j for i in top for j in range(max(0, i - w), min(len(e.rounds), i + w + 1))})
    return top


def slug(cond):
    return cond.replace("|", "__").replace(":", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="memeye")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--cond", action="append", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max questions per episode (0=all)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--seed", type=int, default=None, help="reader sampling seed (replicates for test-retest)")
    ap.add_argument("--reader", default=mm.READER)
    ap.add_argument("--thinking", default=None, help="gemini-2.5: int budget (0=off); gemini-3.x: level")
    ap.add_argument("--qids", default=None, help="json file with {'qids': [...]} restricting the questions")
    ap.add_argument("--grow", type=int, default=0, help="add ~N distractor images (Mem-Gallery sessions), see scale.py")
    args = ap.parse_args()

    eps = episodes(args.bench, args.split, args.tasks)
    if args.grow:
        import scale
        eps = [scale.grow(e, args.grow) for e in eps]
    if args.qids:
        keep = set(json.load(open(args.qids))["qids"])
        for e in eps:
            e.qas = [q for q in e.qas if q.qid in keep]
        eps = [e for e in eps if e.qas]
    styles = sorted({p for c in args.cond for p in [mm.parse_rep(c.split("|")[1])[1]] if p} |
                    {k[4:] for c in args.cond for k in [parse_sel(c.split("|")[0])[0]] if k.startswith("cap:")})
    caps = {s: mm.build_captions(eps, s) for s in styles}
    keys = sorted({parse_sel(c.split("|")[0])[0] for c in args.cond} - {"fc", "oracle", "oracle_sess"})
    idx = {e.name: mm.RoundIndex(e, caps, keys) for e in eps} if keys else {}
    outdir = Path(args.outdir or ROOT / f"results/mm/{args.bench}_{args.split}")
    outdir.mkdir(parents=True, exist_ok=True)

    for cond in args.cond:
        t0 = time.time()
        sel, rep = cond.split("|")
        key, k = parse_sel(sel)
        jobs = []
        for e in eps:
            qs = e.qas[: args.limit] if args.limit else e.qas
            for qa in qs:
                hi = []
                if key == "fc":
                    ids = list(range(len(e.rounds)))
                elif key == "oracle":
                    r2i = e.rid2idx
                    ids = sorted({r2i[c] for c in qa.clue if c in r2i})
                elif key == "oracle_sess":  # gold clue rounds + every other round of their sessions
                    r2i = e.rid2idx
                    sids = {e.rounds[r2i[c]].sid for c in qa.clue if c in r2i}
                    ids = [i for i, r in enumerate(e.rounds) if r.sid in sids]
                elif sel.startswith("fc+"):
                    ids, hi = list(range(len(e.rounds))), idx[e.name].topk(qa, key, k)
                elif sel.startswith("dq+"):  # LLM query decomposition, union capped at k rounds
                    ids = None
                else:
                    ids = expand(e, idx[e.name].topk(qa, key, k), sel)
                jobs.append((e, qa, ids, hi))

        def run(job):
            e, qa, ids, hi = job
            if ids is None:
                ids = mm.decomposed_topk(idx[e.name], qa, key, k)
            th = ("low" if args.thinking is None else args.thinking) if not args.reader.startswith("gemini-2.5") else \
                int(args.thinking or 0)
            a = mm.answer(e, qa, ids, rep, caps, tag=f"mm:{cond}", hi_idxs=hi, seed=args.seed, reader=args.reader,
                          thinking=th)
            j = mm.judge(qa.question, qa.answer, a["pred"])
            sel_ids = hi or ids
            return {"qid": qa.qid, "task": qa.task, "qtype": qa.qtype, "x": qa.x, "y": qa.y, "cond": cond,
                    "retrieved": [e.rounds[i].rid for i in sel_ids], "clue": qa.clue, **mm.clue_stats(e, qa, sel_ids),
                    "question": qa.question, "gold": qa.answer, "pred": a["pred"], "score": j["score"],
                    "judge_reason": j["reason"], "usage": a["usage"], "latency_s": a["latency_s"]}

        # warm implicit prompt caching: one question per episode first (identical history prefix for fc)
        first = {}
        for i, (e, _, _, _) in enumerate(jobs):
            first.setdefault(e.name, i)
        warm = pmap(run, [jobs[i] for i in first.values()], workers=args.workers)
        rest_i = [i for i in range(len(jobs)) if i not in set(first.values())]
        rest = pmap(run, [jobs[i] for i in rest_i], workers=args.workers, desc=cond)
        recs = [None] * len(jobs)
        for i, r in zip(first.values(), warm):
            recs[i] = r
        for i, r in zip(rest_i, rest):
            recs[i] = r
        sfx = (f"__seed{args.seed}" if args.seed is not None else "") + \
              (f"__{args.reader}" if args.reader != mm.READER else "") + \
              (f"__think{args.thinking}" if args.thinking is not None else "")
        with open(outdir / f"{slug(cond)}{sfx}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summarize(recs, cond + sfx, args.bench, args.split, time.time() - t0)


def summarize(recs, cond, bench, split, secs=0.0):
    sc = np.array([r["score"] for r in recs])
    by = collections.defaultdict(list)
    for r in recs:
        if r.get("x"):
            by[r["x"]].append(r["score"])
            by[r["y"]].append(r["score"])
        else:
            by[r["qtype"]].append(r["score"])
    cov = [r["cov"] for r in recs if r.get("n_gold")]
    full = [r["full"] for r in recs if r.get("n_gold")]
    pt = np.mean([r["usage"]["prompt"] for r in recs])
    it = np.mean([r["usage"]["img_tokens"] for r in recs])
    s = {"bench": bench, "split": split, "cond": cond, "n": len(recs), "score": round(float(sc.mean()), 4),
         "by": {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items())},
         "clue_cov": round(float(np.mean(cov)), 3) if cov else None,
         "clue_full": round(float(np.mean(full)), 3) if full else None,
         "prompt_tok": int(pt), "img_tok": int(it), "secs": round(secs, 1), "spend_total_usd": spend().get("TOTAL")}
    print(json.dumps(s))
    with open(ROOT / "results/mm/summary.jsonl", "a") as f:
        f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
