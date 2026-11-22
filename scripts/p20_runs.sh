#!/bin/bash
# Phase 20: strong reader on MemEye (development benchmark) with a small union: selected rounds U top-k similar rounds.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python; LG=results/p20/logs; CS="--compact noassist,telegraph"
for k in 10 20; do
  $PY scripts/run_mm2.py --bench memeye --reader gemini-3.8-flash --cond "tocu:40:$k|read+toc:low" $CS --xcache --workers 8 --outdir results/p20/memeye > $LG/memeye_u$k.log 2>&1
done
echo "DONE $(date +%H:%M)" > $LG/STATUS_memeye
