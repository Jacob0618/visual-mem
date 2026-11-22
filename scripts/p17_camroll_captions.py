"""Phase 17: describe every CamRoll photo once (served from the response cache afterwards).
.venv/bin/python scripts/p17_camroll_captions.py [workers]"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mm  # noqa: E402
from data_camroll import load_camroll  # noqa: E402

eps = load_camroll()
caps = mm.build_captions(eps, "short", workers=int(sys.argv[1]) if len(sys.argv) > 1 else 40)
print("captions", len(caps), "| mean chars %.0f" % (sum(len(c) for c in caps.values()) / len(caps)), flush=True)
