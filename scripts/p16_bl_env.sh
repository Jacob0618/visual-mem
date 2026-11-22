# Phase 16: environment of the rebuilt memory-system baselines (source this file; paths are relative to the project root).
#   v2 image descriptions (served from cache/gen.sqlite), leak-free reader prompt, MemLens on all 466 test questions,
#   separate outputs and ledgers, a $100 cap per system.
export P12_LEDGER_DIR="${P12_LEDGER_DIR:-results/p16/budget}"
export P16_BASELINE_DIR="${P16_BASELINE_DIR:-results/p16/baselines}"
export P16_MEMLENS_QIDS="${P16_MEMLENS_QIDS:-results/memlens_test.json}"
export P16_VMEM_TOPK=20      # V-Mem: 20 rounds before its +/-1 neighbour windows (about 40 rounds read)
export P16_MMA_TOPK=80       # MMA: 80 memory entries (turns), the number of turns in 40 rounds
export P12_CAP_USD=100
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SSL_CERT_FILE="$(.venv_bl/bin/python -c 'import certifi; print(certifi.where())')"
