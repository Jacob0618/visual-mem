"""Fetch the VisualMem images referenced by data.json from the remote zip (HTTP range requests, parallel)."""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from remotezip import open_remote_zip  # noqa: E402

URL = "https://huggingface.co/datasets/viettmab/visualmem/resolve/main/visualmem_data.zip"
ROOT = Path(__file__).resolve().parents[1] / "data/VisualMem"
d = json.load(open(ROOT / "data.json"))
need = set()
for p in d:
    for e in p["events"]:
        for t in e["conversation"]:
            if t.get("image_path"):
                need.add(t["image_path"])
        for q in e.get("questions") or []:
            if q.get("type") == "image":
                need.update(v for v in q["choices"].values() if isinstance(v, str) and v.endswith((".jpg", ".png", ".jpeg")))
            if q.get("query_image_path"):
                need.add(q["query_image_path"])
todo = sorted(x for x in need if not (ROOT.parent / x).exists())
print("needed", len(need), "to fetch", len(todo), flush=True)
local = threading.local()


def get(name):
    if not hasattr(local, "z"):
        local.z = open_remote_zip(URL)
    dst = ROOT.parent / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(local.z.read(name))
    return name


done = 0
with ThreadPoolExecutor(12) as ex:
    for _ in ex.map(get, todo):
        done += 1
        if done % 100 == 0:
            print("fetched", done, flush=True)
print("done", flush=True)
