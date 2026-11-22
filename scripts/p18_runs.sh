#!/bin/bash
# Phase 18: strong reader WITHOUT the index in its context (opened rounds only), reusing the lightweight path's selection.
#   scripts/p18_runs.sh test   (gate-chosen paths + CamRoll annotated evidence)
#   scripts/p18_runs.sh rest   (the other strong paths and the long-history points; run only if the variant is adopted)
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python; LG=results/p18/logs; CA="--compact noassist,shortid,telegraph"; S="--reader gemini-3.8-flash"
mkdir -p $LG
case $1 in
  test)
    $PY scripts/run_mm2.py --bench memeye $S --cond "toc38:40|read:low" $CA --xcache --workers 8 --outdir results/p18/memeye > $LG/memeye_std.log 2>&1
    $PY scripts/run_mm2.py --bench camroll $S --cond "toc38:40|read:low" $CA --xcache --workers 8 --outdir results/p18/camroll > $LG/camroll_std.log 2>&1
    $PY scripts/run_mm2.py --bench visualmem $S --cond "tocu:40:40|read:low" $CA --xcache --workers 8 --outdir results/p18/visualmem > $LG/visualmem_uni.log 2>&1
    $PY scripts/run_mm2.py --bench camroll $S --cond "oracle|read:low" --workers 8 --outdir results/p18/camroll > $LG/camroll_oracle.log 2>&1
    echo "DONE $(date +%H:%M)" > $LG/STATUS_test ;;
  rest)
    $PY scripts/run_mm2.py --bench memeye $S --cond "tocu:40:40|read:low" $CA --xcache --workers 8 --outdir results/p18/memeye > $LG/memeye_uni.log 2>&1
    $PY scripts/run_mm2.py --bench visualmem $S --cond "toc38:40|read:low" $CA --xcache --workers 8 --outdir results/p18/visualmem > $LG/visualmem_std.log 2>&1
    $PY scripts/run_mm2.py --bench camroll $S --cond "tocu:40:40|read:low" $CA --xcache --workers 8 --outdir results/p18/camroll > $LG/camroll_uni.log 2>&1
    for g in 250 1000; do
      $PY scripts/run_mm2.py --bench memeye --grow $g $S --cond "toc38:40|read:low" $CA --xcache --workers 6 --outdir results/p18/long/dx$g > $LG/long_$g.log 2>&1
    done
    echo "DONE $(date +%H:%M)" > $LG/STATUS_rest ;;
esac
