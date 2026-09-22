#!/bin/bash
# Follow-up experiments: fine alpha grid + per-sample paired bootstrap.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="$1"
GPU="$2"
export HF_HOME="${HF_HOME:-/data/hf_home}"
SLUG=$(echo "$MODEL" | tr '/.' '__')
LOG="logs/followup_${SLUG}.log"

if [ ! -f "outputs/rebias/${SLUG}_fine/adversarial.csv" ]; then
    echo "[$(date '+%F %T')] followup fine-grid rebias for $MODEL" | tee -a "$LOG"
    python3 scripts/run_rebias.py --model "$MODEL" \
        --heads-csv "outputs/taxonomy/$SLUG/heads.csv" \
        --alphas 0.25,0.5,0.75,1.0 --max-per-split 200 \
        --out "outputs/rebias/${SLUG}_fine" --device "cuda:$GPU" 2>&1 | tee -a "$LOG"
fi

echo "[$(date '+%F %T')] bootstrap for $MODEL" | tee -a "$LOG"
python3 scripts/run_bootstrap.py --model "$MODEL" \
    --preds-dir "outputs/rebias/${SLUG}_fine" \
    --out "outputs/bootstrap/${SLUG}.csv" 2>&1 | tee -a "$LOG"

bash scripts/sync_results.sh "followup fine-grid + bootstrap done ($SLUG)"
