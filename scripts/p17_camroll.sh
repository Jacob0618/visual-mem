#!/bin/bash
# Phase 17: fourth benchmark (CamRoll-yfcc20). Waits for the download and the captions, then runs the core conditions,
# the widened paths, and full context with the lightweight reader. Resumable.   scripts/p17_camroll.sh
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python; LG=results/p17/logs; CA="--compact noassist,shortid,telegraph"; CS="--compact noassist,telegraph"
while [ ! -f results/p16/logs/STATUS_camroll_refetch ]; do sleep 20; done
$PY src/data_camroll.py > $LG/camroll_prepare2.log 2>&1
while pgrep -f p17_camroll_captions.py > /dev/null; do sleep 20; done
$PY scripts/p17_camroll_captions.py 40 > $LG/camroll_captions2.log 2>&1
O=results/p17/camroll
$PY scripts/run_mm2.py --bench camroll --cond "rag:40|read:medium" --workers 10 --outdir $O > $LG/camroll_rag_lite.log 2>&1
$PY scripts/run_mm2.py --bench camroll --cond "toc38:40|read:medium" $CA --xcache --workers 8 --outdir $O > $LG/camroll_std_lite.log 2>&1
$PY scripts/run_mm2.py --bench camroll --reader gemini-3.8-flash --cond "rag:40|read:low" --workers 8 --outdir $O > $LG/camroll_rag_strong.log 2>&1
$PY scripts/run_mm2.py --bench camroll --reader gemini-3.8-flash --cond "toc38:40|read+toc:low" $CS --xcache --workers 8 --outdir $O > $LG/camroll_std_strong.log 2>&1
echo "DONE $(date +%H:%M)" > $LG/STATUS_camroll_core
$PY scripts/run_mm2.py --bench camroll --cond "toc38:40|readn:medium" --snippet 600 $CA --xcache --workers 8 --outdir $O > $LG/camroll_wid_lite.log 2>&1
$PY scripts/run_mm2.py --bench camroll --reader gemini-3.8-flash --cond "tocu:40:40|read+toc:low" $CS --xcache --workers 8 --outdir $O > $LG/camroll_uni_strong.log 2>&1
$PY scripts/run_mm2.py --bench camroll --cond "fc|read:medium" --workers 6 --outdir $O > $LG/camroll_fc_lite.log 2>&1
echo "DONE $(date +%H:%M)" > $LG/STATUS_camroll_paths
