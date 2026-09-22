#!/bin/bash
# Re-run ONLY the sink probe (grabber first_k slice bug), then resume the pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="$1"
GPU="$2"
export HF_HOME="${HF_HOME:-/data/hf_home}"
SLUG=$(echo "$MODEL" | tr '/.' '__')
DEV="cuda:$GPU"
LOG="logs/rerun_sink_${SLUG}.log"

rm -f "outputs/raw/${SLUG}/sink.npz"
echo "[$(date '+%F %T')] re-running sink for $MODEL on $DEV" | tee -a "$LOG"
python3 scripts/run_probes.py --model "$MODEL" --data data --probes sink \
    --max-samples 400 --device "$DEV" 2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] sink done; resuming pipeline" | tee -a "$LOG"

exec bash scripts/run_pipeline.sh --model "$MODEL" --gpu "$GPU"
