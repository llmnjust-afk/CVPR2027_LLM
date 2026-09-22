#!/bin/bash
# Finish a rebias stage whose main sweep is done: merge split CSVs, run VCD
# (with the attention_mask fix), write all_results.csv, set flag, sync.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="$1"
GPU="$2"
export HF_HOME="${HF_HOME:-/data/hf_home}"
SLUG=$(echo "$MODEL" | tr '/.' '__')
LOG="logs/finish_rebias_${SLUG}.log"

python3 scripts/run_rebias.py --model "$MODEL" \
    --heads-csv "outputs/taxonomy/$SLUG/heads.csv" \
    --vcd-only --device "cuda:$GPU" 2>&1 | tee -a "$LOG"
touch "outputs/flags/${SLUG}_done.done"
bash scripts/sync_results.sh "S3 rebias done (vcd finish)"
