#!/bin/bash
# Phase 17 runs (user request 2026-11-21): long histories, visibility manipulation, selector models. Resumable.
#   scripts/p17_runs.sh long | vis | sel
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python; LG=results/p17/logs; CA="--compact noassist,shortid,telegraph"; CS="--compact noassist,telegraph"
case $1 in
  long)   # MemEye histories grown with distractor sessions (Mem-Gallery chats): RAG@40 and the standard path of VGS, both readers
    for g in 250 1000 500; do
      O=results/p17/long/dx$g
      $PY scripts/run_mm2.py --bench memeye --grow $g --cond "rag:40|read:medium" --workers 8 --outdir $O > $LG/long_${g}_rag_lite.log 2>&1
      $PY scripts/run_mm2.py --bench memeye --grow $g --cond "toc38:40|read:medium" $CA --xcache --workers 6 --outdir $O > $LG/long_${g}_vgs_lite.log 2>&1
      $PY scripts/run_mm2.py --bench memeye --grow $g --reader gemini-3.8-flash --cond "rag:40|read:low" --workers 6 --outdir $O > $LG/long_${g}_rag_strong.log 2>&1
      $PY scripts/run_mm2.py --bench memeye --grow $g --reader gemini-3.8-flash --cond "toc38:40|read+toc:low" $CS --xcache --workers 6 --outdir $O > $LG/long_${g}_vgs_strong.log 2>&1
      echo "DONE $(date +%H:%M)" > $LG/STATUS_long_$g
    done ;;
  vis)    # index visibility manipulated by the line length (lightweight reader, standard path)
    for s in 30 60; do
      $PY scripts/run_mm2.py --bench memeye --cond "toc38:40|read:medium" $CA --snippet $s --xcache --workers 6 --outdir results/p17/vis/memeye > $LG/vis_memeye_$s.log 2>&1
    done
    for s in 600 2000; do
      $PY scripts/run_mm2.py --bench memlens --split 32k --qids_file results/memlens_test.json --cond "toc38:40|read:medium" $CA --snippet $s --workers 6 --outdir results/p17/vis/memlens > $LG/vis_memlens_$s.log 2>&1
    done
    $PY scripts/run_mm2.py --bench memlens --split 32k --qids_file results/memlens_test.json --cond "toc38:40|readn:medium" $CA --workers 6 --outdir results/p17/vis/memlens > $LG/vis_memlens_notes200.log 2>&1
    echo "DONE $(date +%H:%M)" > $LG/STATUS_vis ;;
  sel)    # selector models on the compact index (selection only, MemEye)
    for m in qwent gpt5m ds 2.5f 2.5fl gpt41m; do
      $PY scripts/select_eval.py --bench memeye $CA --sel "toc@$m:40" > $LG/sel_$m.log 2>&1
    done
    echo "DONE $(date +%H:%M)" > $LG/STATUS_sel ;;
esac
