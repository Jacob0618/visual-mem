#!/bin/bash
# Phase 18, variant U0 (pre-stated in RESEARCH_LOG P18-D3): strong reader reads selected U top-40 rounds without the index.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python; LG=results/p18/logs; CA="--compact noassist,shortid,telegraph"; S="--reader gemini-3.8-flash"
$PY scripts/run_mm2.py --bench memeye $S --cond "tocu:40:40|read:low" $CA --xcache --workers 8 --outdir results/p18/memeye > $LG/memeye_uni.log 2>&1
$PY scripts/run_mm2.py --bench camroll $S --cond "tocu:40:40|read:low" $CA --xcache --workers 8 --outdir results/p18/camroll > $LG/camroll_uni.log 2>&1
echo "DONE $(date +%H:%M)" > $LG/STATUS_u0
