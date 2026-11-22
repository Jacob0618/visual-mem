"""Run a command and record its wall-clock window under a tag in results/p8/wma_windows.json (for attributing
explicit-cache creation and storage costs to a run).  python scripts/p8_window.py TAG -- cmd args..."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tag, cmd = sys.argv[1], sys.argv[sys.argv.index("--") + 1:]
t0 = time.time()
rc = subprocess.call(cmd)
f = ROOT / "results/p8/wma_windows.json"
W = json.load(open(f)) if f.exists() else {}
W[tag] = [t0, time.time()]
json.dump(W, open(f, "w"), indent=1)
sys.exit(rc)
