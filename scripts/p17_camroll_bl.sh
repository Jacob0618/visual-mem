#!/bin/bash
# Phase 17: the five memory systems and the LLM-rerank baseline on CamRoll (after the photo descriptions exist).
# Uses scripts/p16_run_baselines.sh (outputs results/p16/baselines/camroll/, ledgers results/p16/budget/camroll/).
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python; LG=results/p17/logs; CA="--compact noassist,shortid,telegraph"
while [ ! -f results/p16/logs/STATUS_camroll_refetch ]; do sleep 20; done
while pgrep -f p17_camroll_captions.py > /dev/null; do sleep 20; done
sleep 120                                   # the core chain prepares the last camera roll and its descriptions first
while pgrep -f p17_camroll_captions.py > /dev/null; do sleep 20; done
(scripts/p16_run_baselines.sh vmem camroll 3; scripts/p16_run_baselines.sh mma camroll 4) > /dev/null 2>&1 &
scripts/p16_run_baselines.sh telemem camroll 4 > /dev/null 2>&1 &
scripts/p16_run_baselines.sh omni camroll 5 > /dev/null 2>&1 &
scripts/p16_run_baselines.sh simplemem camroll 6 > /dev/null 2>&1 &
# LLM rerank over the 100 most similar photos (ranking control of the ablation), read by both readers
$PY scripts/select_eval.py --bench camroll $CA --sel "rankr38:100" > $LG/camroll_rank.log 2>&1
$PY scripts/p17_camroll_ids.py > $LG/camroll_ids.log 2>&1
for rd in lite strong; do
  $PY scripts/p14_read_ids.py --bench camroll --ids results/p17/ids/camroll_rerank40.json --out results/p17/camroll/ret_rerank40_$rd.jsonl --reader $rd --workers 8 > $LG/camroll_rerank_$rd.log 2>&1
done
wait
echo "DONE $(date +%H:%M)" > $LG/STATUS_camroll_bl
