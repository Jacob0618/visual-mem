#!/bin/bash
# Phase 22: fill the empty cells of the selector-model table and the retrieval-baseline table. Resumable.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python; LG=results/p22/logs; S="--reader gemini-3.8-flash"; CA="--compact noassist,shortid,telegraph"; R=results/p22/ret
mkdir -p $LG $R results/p22/sel
( $PY scripts/run_mm2.py --bench memeye --cond "toc@2.5f:40|read:medium" --cond "toc@2.5fl:40|read:medium" --cond "toc@gpt41m:40|read:medium" $CA --workers 6 --outdir results/p22/sel > $LG/sel_three.log 2>&1 ) &
( $PY scripts/run_mm2.py --bench memeye --cond "toc@qwent:40|read:medium" $CA --workers 8 --outdir results/p22/sel > $LG/sel_qwent.log 2>&1 ) &
( $PY scripts/run_mm2.py --bench camroll --cond "rag:10|read:medium" --cond "rag:20|read:medium" --cond "rag:30|read:medium" --workers 10 --outdir $R/camroll > $LG/camroll_lite.log 2>&1
  $PY scripts/run_mm2.py --bench camroll $S --cond "rag:10|read:low" --cond "rag:20|read:low" --cond "rag:30|read:low" --workers 8 --outdir $R/camroll > $LG/camroll_s.log 2>&1
  $PY scripts/p16_retrieval_ids.py camroll > $LG/ids_camroll.log 2>&1
  for rd in lite strong; do
    $PY scripts/p14_read_ids.py --bench camroll --ids results/p16/ids/camroll_hybrid40.json --out results/p17/camroll/ret_hybrid40_$rd.jsonl --reader $rd --workers 8 > $LG/hyb_camroll_$rd.log 2>&1
  done ) &
( $PY scripts/run_mm2.py --bench memeye $S --cond "rag:20|read:low" --workers 8 --outdir $R/memeye > $LG/memeye_s.log 2>&1
  $PY scripts/run_mm2.py --bench memlens --split 32k --qids_file results/memlens_test.json $S --cond "rag:10|read:low" --cond "rag:20|read:low" --workers 8 --outdir $R/memlens > $LG/memlens_s.log 2>&1
  $PY scripts/run_mm2.py --bench visualmem $S --cond "rag:10|read:low" --cond "rag:20|read:low" --workers 8 --outdir $R/visualmem > $LG/visualmem_s.log 2>&1 ) &
( for b in memeye memlens visualmem; do
    $PY scripts/p14_read_ids.py --bench $b --ids results/p16/ids/${b}_hybrid40.json --out results/p16/final/$b/ret_hybrid40_strong.jsonl --reader strong --workers 8 > $LG/hyb_${b}_s.log 2>&1
  done ) &
wait
echo "DONE $(date +%H:%M)" > $LG/STATUS
