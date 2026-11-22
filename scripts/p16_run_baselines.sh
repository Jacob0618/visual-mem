#!/bin/bash
# Phase 16: rebuild a memory-system baseline on one benchmark (v2 descriptions, leak-free reader prompt, MemLens = 466
# test questions).  Usage: scripts/p16_run_baselines.sh <vmem|mma|telemem|simplemem|omni> <memeye|memlens|visualmem> [n]
#   n = shards (vmem, telemem), worker processes (mma, simplemem) or threads (omni). Resumable: finished records are kept.
cd "$(dirname "$0")/.." || exit 1
SYS=$1; BENCH=$2; N=${3:-1}
source scripts/p16_bl_env.sh
export P12_LEDGER_DIR=results/p16/budget/$BENCH P12_CAP_USD=60
OUT=results/p16/baselines; LOG=results/p16/bl_logs
mkdir -p "$OUT/$BENCH" "$LOG" "$P12_LEDGER_DIR"
case $SYS in
  vmem|telemem)
    for k in $(seq 0 $((N-1))); do
      ( export P16_SHARD=$k/$N P16_BASELINE_DIR=results/p16/bl_shards/$k
        .venv_bl/bin/python scripts/p12_$SYS.py --bench "$BENCH" > "$LOG/${SYS}_${BENCH}_$k.log" 2>&1 ) &
    done
    wait
    cat results/p16/bl_shards/*/"$BENCH"/"$SYS".jsonl > "$OUT/$BENCH/$SYS.jsonl" ;;
  mma)
    P16_BASELINE_DIR=$OUT .venv_bl/bin/python scripts/p12_mma.py --bench "$BENCH" --procs "$N" > "$LOG/mma_$BENCH.log" 2>&1 ;;
  simplemem)
    .venv_bl/bin/python scripts/p11_baselines.py --system simplemem --bench "$BENCH" --threads "$N" \
      --out "$OUT/$BENCH/simplemem.jsonl" > "$LOG/simplemem_$BENCH.log" 2>&1 ;;
  omni)
    .venv_bl/bin/python scripts/p11_baselines.py --system omni --bench "$BENCH" --threads "$N" \
      --out "$OUT/$BENCH/omni.jsonl" > "$LOG/omni_$BENCH.log" 2>&1 ;;
esac
echo "DONE $(date +%H:%M)" > "$LOG/STATUS_${SYS}_${BENCH}"
