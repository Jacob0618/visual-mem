"""Phase 9 (exploration): how complementary are the set view (QSS) and the wide view (RAG@40) per question?
Upper bound (oracle), agreement, and simple combination rules, from stored per-question outputs. No API calls."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import boot_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/mm2"
ABST = re.compile(r"(not (stated|mentioned|specified|provided|available|contain|include|recorded|shown|visible)|no (information|record|mention|evidence)|"
                  r"cannot (be )?(determine|answer|find)|unable to|does not (say|state|mention|specify|contain|include|appear)|isn't (stated|mentioned)|"
                  r"unknown|insufficient|not enough)", re.I)
SET = json.load(open(ROOT / "results/p6/setness.json"))


def L(p):
    if p.startswith("P2:"):   # MemEye phase-2 conditions through the shared loader
        from analyze_p2 import load_all
        return load_all("gemini-2.5-flash-lite")[p[3:]]
    return {json.loads(l)["qid"]: json.loads(l) for l in open(D / p)}


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split())


def agree(a, b):
    a, b = norm(a), norm(b)
    return bool(a) and (a == b or (len(a) > 3 and a in b) or (len(b) > 3 and b in a))


def abst(p):
    return bool(ABST.search(p or "")) and len(p or "") < 300


BENCH = {
    "MemEye (dev)": ("P2:rag:40|read:medium", "P2:toc38:40|read:medium", None),
    "MemLens test (32k)": ("memlens_32k/rag-40__read-medium__gemini-2.5-flash-lite.jsonl", "memlens_32k/toc38-40__read-medium__gemini-2.5-flash-lite.jsonl", "results/memlens_test.json"),
    "Mem-Gallery": ("memgal_all/rag-40__read-medium__gemini-2.5-flash-lite.jsonl", "memgal_all/toc38-40__read-medium__gemini-2.5-flash-lite.jsonl", None),
    "WorldMemArena-personal": ("wma_all_wma_personal_qids/rag-40__read-medium__gemini-2.5-flash-lite.jsonl", "wma_all_wma_personal_qids/toc38-40__read-medium__gemini-2.5-flash-lite__xcache.jsonl", None),
}


def main():
    print("| benchmark | n | RAG@40 | QSS | oracle max | agree | when they disagree: QSS better / RAG better | rule: prefer non-abstaining view | rule: multi->QSS, single->RAG |")
    print("|---|---:|---:|---:|---:|---:|---|---|---|")
    for name, (fr, fq, sub) in BENCH.items():
        R, Q = L(fr), L(fq)
        qs = sorted(set(R) & set(Q))
        if sub:
            keep = set(json.load(open(ROOT / sub)))
            qs = [q for q in qs if q in keep]
        r = np.array([R[q]["score"] for q in qs]); s = np.array([Q[q]["score"] for q in qs])
        ag = np.array([agree(R[q]["pred"], Q[q]["pred"]) for q in qs])
        dis = ~ag
        qb, rb = np.mean(s[dis] > r[dis]), np.mean(r[dis] > s[dis])
        # rule 1: prefer the view that does not abstain; otherwise QSS
        f1 = np.array([R[q]["score"] if (abst(Q[q]["pred"]) and not abst(R[q]["pred"])) else Q[q]["score"] for q in qs])
        d1 = f1 - r; l1, h1 = boot_ci(d1)
        # rule 2: router
        if all(q in SET for q in qs):
            f2 = np.array([Q[q]["score"] if SET[q] else R[q]["score"] for q in qs]); d2 = f2 - r; l2, h2 = boot_ci(d2)
            r2 = f"{f2.mean():.3f} ({d2.mean():+.3f} [{l2:+.3f}, {h2:+.3f}])"
        else:
            r2 = "--"
        print(f"| {name} | {len(qs)} | {r.mean():.3f} | {s.mean():.3f} | {np.maximum(r, s).mean():.3f} | {ag.mean():.2f} | {qb:.2f} / {rb:.2f} | "
              f"{f1.mean():.3f} ({d1.mean():+.3f} [{l1:+.3f}, {h1:+.3f}]) | {r2} |")


if __name__ == "__main__":
    main()
