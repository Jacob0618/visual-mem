"""Phase 11: MIRIX (Wang & Chen, 2025; pip mirix==0.1.5) as a baseline. Step 1 (this script, .venv_mirix): build one
MIRIX memory per history and retrieve the memory context of every question; step 2 (scripts/p11_read_ctx.py, main
venv) reads the contexts with our readers and judges them.
  .venv_mirix/bin/python scripts/p11_mirix.py --bench memeye --procs 8
Each history is ingested session by session with mirix.add(text, image_uris=...): the text carries the dated turns and
[Image: caption] markers, and the raw images are passed so that MIRIX's own multimodal agents see them. Every worker
runs with its own HOME (MIRIX stores its SQLite database in ~/.mirix). Memory model: gemini-2.5-flash, the strongest
model in the list that MIRIX 0.1.5 supports (it fails on Gemini 3 responses).
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemini-2.5-flash"   # MIRIX 0.1.5 cannot parse some Gemini 3 responses (KeyError: role)


def _load(bench, qids):
    """Episodes as plain dicts, produced with the main environment (our loaders need its packages)."""
    import hashlib
    import subprocess
    tag = hashlib.sha1(json.dumps(sorted(qids or [])).encode()).hexdigest()[:8] if qids else "all"
    out = ROOT / f"baselines/runs/mirix/{bench}_{tag}_episodes.json"
    if not out.exists():
        code = f"""
import json, sys
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import mm
from run_mm import episodes
bench = {bench!r}; qids = {qids!r}
eps = episodes(bench, '32k' if bench == 'memlens' else 'all', qids if bench != 'memeye' else None)
caps = mm.build_captions(eps, 'short')
D = []
for e in eps:
    qs = [{{'qid': q.qid, 'question': q.question}} for q in e.qas if (not qids or q.qid in set(qids))]
    if not qs:
        continue
    sess = []
    for r in e.rounds:
        if not sess or sess[-1]['sid'] != r.sid:
            sess.append({{'sid': r.sid, 'date': r.date, 'turns': [], 'images': []}})
        cap = ' '.join('[Image: ' + caps.get(p, 'image') + ']' for p in r.images)
        sess[-1]['turns'].append('[' + r.date + '] User: ' + (r.user + (' ' + cap if cap else '')).strip() + '\\nAssistant: ' + r.assistant)
        sess[-1]['images'] += list(r.images)
    D.append({{'name': e.name, 'sessions': sess, 'questions': qs}})
json.dump(D, open({str(out)!r}, 'w'))
print(len(D), 'episodes')
"""
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(ROOT / ".venv/bin/python"), "-c", code], cwd=ROOT, check=True)
    return json.load(open(out))


def run_episode(ep):
    home = Path(ep.get("home_root") or ROOT / "baselines/runs/mirix") / ep["bench"] / re.sub(r"[^A-Za-z0-9_.-]", "_", ep["name"])
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(home)
    sys.path.insert(0, str(ROOT / "scripts"))
    import p11_mirix_patch
    p11_mirix_patch.apply()
    p11_mirix_patch.TAG["v"] = f"bl:mirix:{ep['bench']}:build"
    from mirix import Mirix
    m = Mirix(api_key=os.environ["GEMINI_API_KEY"], model_provider="google_ai", model=MODEL)
    uid = m.list_users()[0].id
    done_flag = home / "BUILT"
    t0 = time.time()
    direct = ep.get("direct", False)        # phase 15: resumable single-process mode, restarted by a driver after a hang
    added = {l.split("\t")[0]: float(l.split("\t")[1]) for l in open(home / "ADDED")} if (home / "ADDED").exists() else {}
    hung = [l.strip() for l in open(home / "HUNG")] if (home / "HUNG").exists() else []
    if not done_flag.exists():
        import shutil
        tmp = home / "image_copies"          # MIRIX deletes uploaded files by default: give it copies only
        for s in ep["sessions"]:
            if direct and (str(s["sid"]) in added or hung.count(str(s["sid"])) >= 2):
                continue                     # added before a restart, or hung twice (skipped)
            text = "\n".join(s["turns"])
            copies = []
            for k, src in enumerate(s["images"]):
                tmp.mkdir(exist_ok=True)
                dst = tmp / f"{s['sid']}_{k}.jpg"
                from PIL import Image                # some MemLens .jpg files hold PNG or WebP data; MIRIX declares the
                im = Image.open(src)                 # MIME type from the extension and Gemini rejects the mismatch
                if im.format == "JPEG":
                    shutil.copyfile(src, dst)
                else:
                    im.convert("RGB").save(dst, "JPEG", quality=95)
                copies.append(str(dst))
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
            for attempt in range(3):
                ex = ThreadPoolExecutor(1)       # watchdog: an add that hangs inside MIRIX is abandoned after 15 min
                ta = time.perf_counter()
                try:
                    ex.submit(m.add, text, image_uris=copies or None, delete_after_upload=False).result(timeout=480 if direct else 900)
                    if direct:
                        with open(home / "ADDED", "a") as f:
                            f.write(f"{s['sid']}\t{time.perf_counter() - ta:.2f}\n")
                    break
                except FutTimeout:
                    if direct:                   # the hung call keeps MIRIX's locks: record it and kill the process
                        print(f"[{ep['name']}] add hung; session {s['sid']}; restarting", flush=True)
                        with open(home / "HUNG", "a") as f:
                            f.write(str(s["sid"]) + "\n")
                        os._exit(75)
                    print(f"[{ep['name']}] add timed out; session {s['sid']} skipped", flush=True)
                    with open(home / "SKIPPED", "a") as f:
                        f.write(str(s["sid"]) + "\n")
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        print(f"[{ep['name']}] add failed: {e!r}"[:300], flush=True)
                    time.sleep(5)
                finally:
                    ex.shutdown(wait=False)
        if direct:                           # construction time = the summed durations of the completed adds
            added = {l.split("\t")[0]: float(l.split("\t")[1]) for l in open(home / "ADDED")} if (home / "ADDED").exists() else {}
            done_flag.write_text(f"{sum(added.values()):.2f}")
        else:
            done_flag.write_text(str(time.time() - t0))
    p11_mirix_patch.TAG["v"] = f"bl:mirix:{ep['bench']}:query"
    out = []
    t_build = float(done_flag.read_text() or 0) if done_flag.exists() else None
    for k, q in enumerate(ep["questions"]):
        try:
            t1 = time.perf_counter()
            ctx = m.extract_memory_for_system_prompt(q["question"], uid)
            t_ret = time.perf_counter() - t1
            ctx = re.sub(r"^Current Time:.*\n", "", ctx or "")       # live-deployment clock, not memory content
            out.append({"qid": q["qid"], "ctx": ctx, "images": [], "t_retrieve_s": round(t_ret, 3),
                        "t_build_s": t_build, "warm": k == 0})
        except Exception as e:  # noqa: BLE001
            out.append({"qid": q["qid"], "error": repr(e)[:300]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["memeye", "memlens", "visualmem"])
    ap.add_argument("--qids_file", default=None)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=None, help="context file (default results/baselines/<bench>/mirix_ctx.jsonl)")
    ap.add_argument("--home_root", default=None, help="root of the per-history MIRIX homes")
    ap.add_argument("--direct", action="store_true", help="no worker pool; resumable after a hang (scripts/p15_mirix_driver.py)")
    a = ap.parse_args()
    if "GEMINI_API_KEY" not in os.environ:
        sys.path.insert(0, str(ROOT / "src"))
        from keys import gemini_key
        os.environ["GEMINI_API_KEY"] = gemini_key()
    qids = json.load(open(a.qids_file)) if a.qids_file else None
    eps = _load(a.bench, qids)
    outp = Path(a.out) if a.out else ROOT / "results/baselines" / a.bench / "mirix_ctx.jsonl"
    outp.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["qid"] for l in open(outp)} if outp.exists() else set()
    jobs = [{**e, "bench": a.bench, "home_root": a.home_root, "direct": a.direct} for e in eps if any(q["qid"] not in done for q in e["questions"])]
    print(f"mirix on {a.bench}: {len(jobs)} episodes, {sum(len(e['questions']) for e in jobs)} questions, "
          f"{sum(len(e['sessions']) for e in jobs)} sessions to add", flush=True)
    if a.direct:
        for e in jobs:
            recs = run_episode(e)
            with open(outp, "a") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
        return
    with mp.get_context("spawn").Pool(a.procs) as pool, open(outp, "a") as f:
        for i, recs in enumerate(pool.imap_unordered(run_episode, jobs), 1):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"  episodes done {i}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
