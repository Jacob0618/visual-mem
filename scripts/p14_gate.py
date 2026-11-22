"""Phase 14: how good is the visibility gate? Per-question outcomes of both lightweight-reader paths (standard: QSS with
200-character lines; widened: 600-character lines + per-member notes) are combined under different routing policies:
always standard, always widened, the visibility gate (v_hat >= tau -> standard), a message-length gate (mean user-message
length of the 10 most similar rounds >= L -> widened), random routing with the gate's share, and a per-question oracle.
Thresholds of the two gates are chosen on the development data (MemEye + MemLens development split). No API calls.
python scripts/p14_gate.py [--v2]   -> results/p14/gate.json"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze import boot_ci  # noqa: E402
from p11_analyze import L, M  # noqa: E402
from p8_offline import auc  # noqa: E402

TAU = 0.7
OUT = ROOT / "results/p14"


def rounds_text(bench, qids=None):
    """qid -> mean length (characters) of the user messages of its 10 most similar rounds (cosine order of RAG@40)."""
    from run_mm import episodes
    if bench == "memeye":
        eps = episodes("memeye", "all")
    elif bench == "memlens":
        eps = episodes("memlens", "32k", qids)
    else:
        eps = episodes("visualmem", "all")
    return {e.name: e for e in eps}, {qa.qid: e.name for e in eps for qa in e.qas}


def length_feature(bench, rag, qids):
    E, ep_of = rounds_text(bench, sorted(qids) if bench == "memlens" else None)
    out = {}
    for q in qids:
        r = rag.get(q)
        if r is None or q not in ep_of:
            continue
        e = E[ep_of[q]]
        ids = r["retrieved"] if isinstance(r["retrieved"], list) else eval(r["retrieved"])
        r2 = {x.rid: x for x in e.rounds}
        lens = [len(r2[i].user) for i in ids[:10] if i in r2]
        out[q] = float(np.mean(lens)) if lens else 0.0
    return out


def load_v1():
    """bench -> dict(std, wid, rag, vhat) with per-question records (stored phase 5-10 runs, lightweight reader)."""
    from analyze_p2 import load_all
    V = json.load(open(ROOT / "results/p5/visibility_hat.json"))
    test = set(json.load(open(ROOT / "results/memlens_test.json")))
    D = {}
    P2 = load_all("gemini-2.5-flash-lite")
    D["memeye"] = {"std": P2["toc38:40|read:medium"], "wid": L(M / "memeye_all/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl"),
                   "rag": P2["rag:40|read:medium"], "V": V}
    std_ml = L(M / "memlens_32k/toc38-40__read-medium__gemini-2.5-flash-lite.jsonl")
    rag_ml = L(M / "memlens_32k/rag-40__read-medium__gemini-2.5-flash-lite.jsonl")
    wid_t = L(M / "memlens_32k_memlens_test/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl")
    wid_d = L(M / "memlens_32k_memlens_dev/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600.jsonl")
    D["memlens"] = {"std": {q: r for q, r in std_ml.items() if q in test}, "wid": wid_t,
                    "rag": {q: r for q, r in rag_ml.items() if q in test}, "V": V}
    D["memlens_dev"] = {"std": {q: r for q, r in std_ml.items() if q in wid_d}, "wid": wid_d,
                        "rag": {q: r for q, r in rag_ml.items() if q in wid_d}, "V": V}
    Vv = json.load(open(ROOT / "results/p10/vm_visibility_hat.json"))
    D["visualmem_low"] = {"std": L(M / "visualmem_all/toc38-40__read-medium__gemini-2.5-flash-lite__xcache.jsonl"),
                          "wid": L(M / "visualmem_all_vm_lovis/toc38-40__readn-medium__gemini-2.5-flash-lite__snip600__xcache.jsonl"),
                          "rag": L(M / "visualmem_all/rag-40__read-medium__gemini-2.5-flash-lite.jsonl"), "V": Vv}
    return D


def analyse(D, tau_star, len_star):
    res = {}
    for bench, d in D.items():
        qs = sorted(q for q in d["std"] if q in d["wid"] and q in d["V"])
        s = np.array([d["std"][q]["score"] for q in qs])
        w = np.array([d["wid"][q]["score"] for q in qs])
        v = np.array([d["V"][q] for q in qs])
        lf = d.get("len", {})
        ln = np.array([lf.get(q, 0.0) for q in qs])
        rag = np.array([d["rag"][q]["score"] for q in qs]) if all(q in d["rag"] for q in qs) else None
        gate = np.where(v >= tau_star, s, w)
        lgate = np.where(ln >= len_star, w, s)
        share_std = float(np.mean(v >= tau_star))
        rand = share_std * s.mean() + (1 - share_std) * w.mean()
        diff = w - s
        lab = diff > 0
        keep = diff != 0
        r = {"n": len(qs), "always_standard": float(s.mean()), "always_widened": float(w.mean()), "gate": float(gate.mean()),
             "length_gate": float(lgate.mean()), "random_same_share": float(rand), "oracle": float(np.maximum(s, w).mean()),
             "share_standard": share_std, "share_standard_length": float(np.mean(ln < len_star)),
             "auc_vhat": auc(-v[keep], lab[keep]) if keep.sum() > 5 else None,
             "auc_length": auc(ln[keep], lab[keep]) if keep.sum() > 5 else None,
             "n_differ": int(keep.sum()),
             "tau_curve": {f"{t:.2f}": float(np.where(v >= t, s, w).mean()) for t in np.arange(0, 1.0001, 0.05)}}
        for name, x in [("gate_vs_standard", gate - s), ("gate_vs_widened", gate - w), ("gate_vs_length", gate - lgate)]:
            lo, hi = boot_ci(x)
            r[name] = [float(x.mean()), float(lo), float(hi)]
        if rag is not None:
            lo, hi = boot_ci(gate - rag)
            r["gate_vs_rag"] = [float((gate - rag).mean()), float(lo), float(hi)]
            r["rag"] = float(rag.mean())
        res[bench] = r
    return res


def choose_length_threshold(D, benches=("memeye", "memlens_dev")):
    """Length threshold maximising the mean accuracy over the development data, as tau was chosen."""
    best = None
    grid = sorted({x for b in benches for x in D[b]["len"].values()})
    cand = np.unique(np.percentile(grid, np.arange(0, 101, 1)))
    for Lc in cand:
        accs = []
        for b in benches:
            d = D[b]
            qs = [q for q in d["std"] if q in d["wid"]]
            accs.append(np.mean([d["wid"][q]["score"] if d["len"].get(q, 0) >= Lc else d["std"][q]["score"] for q in qs]))
        m = float(np.mean(accs))
        if best is None or m > best[0] + 1e-12:
            best = (m, float(Lc))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true", help="also analyse the phase-14 re-runs (results/p14/runs)")
    ap.add_argument("--strong", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    D = load_v1()
    for b, d in D.items():
        bench = "memlens" if b.startswith("memlens") else "visualmem" if b.startswith("visualmem") else "memeye"
        d["len"] = length_feature(bench, d["rag"], set(d["std"]) & set(d["wid"]))
    len_star = choose_length_threshold(D)
    res = analyse(D, TAU, len_star)
    res["_thresholds"] = {"tau": TAU, "length_chars": len_star}
    json.dump(res, open(OUT / "gate.json", "w"), indent=1)
    print("length threshold (dev):", round(len_star, 1), "characters")
    for b, r in res.items():
        if b.startswith("_"):
            continue
        print(f"\n== {b} (n={r['n']}, {r['n_differ']} questions where the paths differ)")
        for k in ["always_standard", "always_widened", "gate", "length_gate", "random_same_share", "oracle", "rag"]:
            if k in r:
                print(f"  {k:18s} {r[k]:.3f}")
        print(f"  share standard: gate {r['share_standard']:.3f}, length {r['share_standard_length']:.3f} | "
              f"AUC v_hat {r['auc_vhat']} | AUC length {r['auc_length']}")
        for k in ["gate_vs_standard", "gate_vs_widened", "gate_vs_length", "gate_vs_rag"]:
            if k in r:
                print(f"  {k:18s} {r[k][0]:+.3f} [{r[k][1]:+.3f}, {r[k][2]:+.3f}]")
        tc = r["tau_curve"]
        print("  tau curve:", " ".join(f"{t}:{v:.3f}" for t, v in list(tc.items())[::2]))


if __name__ == "__main__":
    main()


def strong_policies(len_star):
    """Strong reader: standard path = QSS (with the cached index as context on MemEye), widened path = RAG@40."""
    V = json.load(open(ROOT / "results/p5/visibility_hat.json"))
    test = set(json.load(open(ROOT / "results/memlens_test.json")))
    out = {}
    sets = {"memeye": (L(M / "memeye_all/toc38-40__read+toc-low__gemini-3.8-flash__cnoassist+telegraph__xcache.jsonl"),
                       L(M / "memeye_all/rag-40__read-low__gemini-3.8-flash.jsonl")),
            "memlens": (L(M / "memlens_32k_memlens_test/toc38-40__read-low__gemini-3.8-flash.jsonl"),
                        L(M / "memlens_32k_memlens_test/rag-40__read-low__gemini-3.8-flash.jsonl"))}
    for bench, (std, rag) in sets.items():
        lite_rag = (load_v1()[bench]["rag"])
        lf = length_feature(bench, lite_rag, set(std) & set(rag))
        qs = sorted(q for q in std if q in rag and q in V and (bench != "memlens" or q in test))
        s = np.array([std[q]["score"] for q in qs]); w = np.array([rag[q]["score"] for q in qs])
        v = np.array([V[q] for q in qs]); ln = np.array([lf.get(q, 0.0) for q in qs])
        gate = np.where(v >= TAU, s, w); lg = np.where(ln >= len_star, w, s)
        sh = float(np.mean(v >= TAU))
        r = {"n": len(qs), "always_qss": float(s.mean()), "always_rag": float(w.mean()), "gate": float(gate.mean()),
             "length_gate": float(lg.mean()), "random_same_share": sh * s.mean() + (1 - sh) * w.mean(),
             "oracle": float(np.maximum(s, w).mean()),
             "tau_curve": {f"{t:.2f}": float(np.where(v >= t, s, w).mean()) for t in np.arange(0, 1.0001, 0.05)}}
        for name, x in [("gate_vs_qss", gate - s), ("gate_vs_rag", gate - w), ("gate_vs_length", gate - lg)]:
            lo, hi = boot_ci(x)
            r[name] = [float(x.mean()), float(lo), float(hi)]
        out[bench] = r
        print(f"strong {bench}: " + " ".join(f"{k} {r[k]:.3f}" for k in ["always_qss", "always_rag", "gate", "length_gate", "random_same_share", "oracle"])
              + " | gate-QSS {:+.3f} [{:+.3f},{:+.3f}]".format(*r["gate_vs_qss"]) + " gate-RAG {:+.3f} [{:+.3f},{:+.3f}]".format(*r["gate_vs_rag"]))
    return out


if __name__ == "__main__" and "--strong" in sys.argv:
    res = json.load(open(OUT / "gate.json"))
    res["_strong"] = strong_policies(res["_thresholds"]["length_chars"])
    json.dump(res, open(OUT / "gate.json", "w"), indent=1)


def load_v2(pool=False):
    """Phase-14 re-runs with fresh image descriptions (MemEye, VisualMem) + the stored MemLens test runs.
    pool=True: both paths read the index lines of a 100-round candidate pool."""
    R = ROOT / "results/p14/runs"
    D = {}
    pre = "ptoc38-100-40" if pool else "toc38-40"
    for bench, suf in [("memeye", ""), ("visualmem", "" if pool else "__xcache")]:
        std = R / bench / f"{pre}__read-medium__gemini-2.5-flash-lite{suf}.jsonl"
        wid = R / bench / f"{pre}__readn-medium__gemini-2.5-flash-lite__snip600{suf}.jsonl"
        rag = R / bench / "rag-40__read-medium__gemini-2.5-flash-lite.jsonl"
        if std.exists() and wid.exists() and rag.exists():
            D[bench] = {"std": L(std), "wid": L(wid), "rag": L(rag), "V": json.load(open(ROOT / f"results/p14/vhat_{bench}.json"))}
    v1 = load_v1()
    D["memlens"] = v1["memlens"]
    D["memeye_dev_v1"], D["memlens_dev"] = v1["memeye"], v1["memlens_dev"]
    return D


if __name__ == "__main__" and "--v2" in sys.argv:
    for pool in (False, True):
        D = load_v2(pool)
        for b, d in D.items():
            bench = "memlens" if b.startswith("memlens") else "visualmem" if b.startswith("visualmem") else "memeye"
            d["len"] = length_feature(bench, d["rag"], set(d["std"]) & set(d["wid"]))
        Dd = {"memeye": D["memeye_dev_v1"], "memlens_dev": D["memlens_dev"]}
        len_star = choose_length_threshold(Dd)
        res = analyse({k: v for k, v in D.items() if k in ("memeye", "memlens", "visualmem")}, TAU, len_star)
        res["_thresholds"] = {"tau": TAU, "length_chars": len_star}
        name = "gate_v2_pool.json" if pool else "gate_v2.json"
        json.dump(res, open(OUT / name, "w"), indent=1)
        print(f"\n##### {name} (length threshold {len_star:.0f} characters)")
        for b, r in res.items():
            if b.startswith("_"):
                continue
            print(f"{b:10s} n={r['n']} std {r['always_standard']:.3f} wid {r['always_widened']:.3f} gate {r['gate']:.3f} "
                  f"len {r['length_gate']:.3f} rand {r['random_same_share']:.3f} oracle {r['oracle']:.3f} rag {r.get('rag', float('nan')):.3f} | "
                  f"AUC vhat {r['auc_vhat']:.3f} len {r['auc_length']:.3f} | gate-RAG {r['gate_vs_rag'][0]:+.3f} [{r['gate_vs_rag'][1]:+.3f},{r['gate_vs_rag'][2]:+.3f}]")
