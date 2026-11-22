#!/bin/bash
# Phase 18: CamRoll column of the ablation table (lightweight reader): QSS over a 100-round cosine pool and the three
# selection controls (cosine at the QSS size, model ranking cut at 8 and at the QSS size).
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python; LG=results/p18/logs; CA="--compact noassist,shortid,telegraph"; O=results/p16/final/camroll
mkdir -p $O $LG
$PY scripts/run_mm2.py --bench camroll --cond "ptoc38:100:40|read:medium" $CA --workers 8 --outdir $O > $LG/camroll_pool.log 2>&1
$PY scripts/p16_ids.py camroll > $LG/camroll_abl_ids.log 2>&1
for k in cosqss rank8 rankqss; do
  $PY scripts/p14_read_ids.py --bench camroll --ids results/p16/ids/camroll_$k.json --out $O/abl_$k.jsonl --reader lite --workers 8 > $LG/camroll_abl_$k.log 2>&1
done
echo "DONE $(date +%H:%M)" > $LG/STATUS_abl
