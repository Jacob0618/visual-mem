"""Phase 11, step 2 for systems that run in their own environment (MIRIX): read stored memory contexts with our
readers (same prompt as scripts/p11_baselines.py) and judge them.
  .venv/bin/python scripts/p11_read_ctx.py --system mirix --bench memeye
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
PRICES = {"gemini-3.8-flash": (0.75, 3.75), "gemini-2.5-flash-lite": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50)}
READERS = {"lite": ("gemini-2.5-flash-lite", 0, "medium"), "strong": ("gemini-3.8-flash", "low", "low")}


def stage_usd(system, bench, stage, since=0.0):
    tot = 0.0
    for l in open(ROOT / "cache/usage_bl.jsonl"):
        r = json.loads(l)
        if r["tag"] == f"bl:{system}:{bench}:{stage}" and r["t"] >= since:
            pi, po = PRICES.get(r["model"], (0.75, 3.75))
            tot += (r["prompt"] - r.get("cached", 0)) * pi / 1e6 + r.get("cached", 0) * pi / 10 / 1e6 + (r["out"] + r["thoughts"]) * po / 1e6
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--bench", required=True, choices=["memeye", "memlens", "visualmem"])
    ap.add_argument("--readers", default="lite,strong")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--since", type=float, default=0.0, help="count construction calls from this time on (clean run)")
    ap.add_argument("--ctx", default=None, help="context file (default results/baselines/<bench>/<system>_ctx.jsonl)")
    ap.add_argument("--out", default=None, help="output file (default results/baselines/<bench>/<system>.jsonl)")
    ap.add_argument("--nojudge", action="store_true", help="latency runs: do not judge")
    a = ap.parse_args()
    import mm
    import mm2
    from analyze_p2 import p1_usd
    from gem import Img, generate
    from run_mm import episodes
    C = {json.loads(l)["qid"]: json.loads(l) for l in open(a.ctx or ROOT / f"results/baselines/{a.bench}/{a.system}_ctx.jsonl")}
    C = {q: r for q, r in C.items() if not r.get("error")}
    eps = episodes(a.bench, "32k" if a.bench == "memlens" else "all", list(C) if a.bench != "memeye" else None)
    QA = {q.qid: (e, q) for e in eps for q in e.qas if q.qid in C}
    n = len(QA)
    share_build = stage_usd(a.system, a.bench, "build", a.since) / max(1, n)
    share_query = stage_usd(a.system, a.bench, "query", a.since) / max(1, n)
    outp = Path(a.out) if a.out else ROOT / f"results/baselines/{a.bench}/{a.system}.jsonl"
    done = set()
    if outp.exists():
        done = {(json.loads(l)["qid"], json.loads(l).get("reader")) for l in open(outp) if not json.loads(l).get("error")}

    def one(item):
        qid, rd = item
        ep, qa = QA[qid]
        model, thinking, res = READERS[rd]
        ctx = C[qid]["ctx"]
        parts = [f"User profile: {json.dumps(ep.profile, ensure_ascii=False)[:1500]}\n"] if ep.profile else []
        parts.append(f"\n=== RETRIEVED MEMORY ===\n{ctx or '(no memory retrieved)'}\n")
        for p in C[qid].get("images", []):
            parts += ["\n[Image from memory]", Img(p, res)]
        parts.append("\n=== END OF MEMORY ===\n")
        parts = mm._merge(parts + mm2._question_parts(qa) + ["\nAnswer:"])
        import time
        t1 = time.perf_counter()
        r = generate(parts, model=model, system=mm.system_prompt(ep.bench), thinking=thinking, tag=f"bl:{a.system}:read",
                     max_output_tokens=2048)
        t_read = time.perf_counter() - t1
        pred = r["text"].strip()
        if a.nojudge:
            score = None
        elif a.bench == "memlens":
            score = mm2.memlens_judge(qa, pred)["score"]
        elif a.bench == "visualmem":
            from data_visualmem import mc_score
            score = mc_score(pred, qa.answer)
        else:
            score = mm.judge(qa.question, qa.answer, pred)["score"]
        r_usd = p1_usd({"usage": r["usage"]}, model)
        return {"qid": qid, "reader": rd, "system": a.system, "pred": pred, "score": score, "ctx_chars": len(ctx),
                "usd_query": share_query, "usd_read": r_usd, "usd_build_share": share_build,
                "usd": share_query + r_usd + share_build, "t_read_s": round(t_read, 3),
                **{k: C[qid][k] for k in ("t_retrieve_s", "t_build_s", "warm") if k in C[qid]}}

    todo = [(q, rd) for q in QA for rd in a.readers.split(",") if (q, rd) not in done]
    print(f"{a.system} on {a.bench}: {len(todo)} readings", flush=True)
    with ThreadPoolExecutor(a.workers) as ex, open(outp, "a") as f:
        for rec in ex.map(one, todo):
            f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
