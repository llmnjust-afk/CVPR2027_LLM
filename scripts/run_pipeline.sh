#!/usr/bin/env bash
# Per-model experiment branch: probes -> taxonomy -> ablation -> patching -> rebias.
# Each stage appends to logs/pipeline_<slug>.log, drops a done-flag, and syncs
# new artifacts to GitHub. Waits for data/.prepped before starting.
# Usage: scripts/run_pipeline.sh --model M --gpu N [--probes ...] [--max-samples 400]
#        [--rebias-per-split 200] [--n-eval 60]
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="" ; GPU=0
PROBES="synth_grounding,ocr_synth,sink,grounding,cond,occlusion,pope"
MAXS=400 ; RBS=200 ; NEVAL=60
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2" ; shift 2 ;;
        --gpu) GPU="$2" ; shift 2 ;;
        --probes) PROBES="$2" ; shift 2 ;;
        --max-samples) MAXS="$2" ; shift 2 ;;
        --rebias-per-split) RBS="$2" ; shift 2 ;;
        --n-eval) NEVAL="$2" ; shift 2 ;;
        *) echo "unknown arg $1" ; exit 1 ;;
    esac
done
[ -n "$MODEL" ] || { echo "--model required" ; exit 1 ; }

SLUG="$(python3 -c "import sys;print(sys.argv[1].replace('/','_').replace('.','_'))" "$MODEL")"
LOG="logs/pipeline_${SLUG}.log"
mkdir -p logs outputs/flags
export PYTHONPATH=src
export HF_HOME="${HF_HOME:-/data/hf_home}"
DEV="cuda:$GPU"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
sync() { bash scripts/sync_results.sh "pipeline[$SLUG]: $1" || true; }

log "waiting for data/.prepped"
while [ ! -f data/.prepped ]; do sleep 60; done
log "data ready; starting branch for $MODEL on $DEV"

log "STAGE probes ($PROBES, max_samples=$MAXS)"
python3 scripts/run_probes.py --model "$MODEL" --data data --probes "$PROBES" \
    --max-samples "$MAXS" --device "$DEV" 2>&1 | tee -a "$LOG"
touch "outputs/flags/${SLUG}_probes.done"
sync "S1 probes done"

log "STAGE taxonomy (per-model)"
python3 scripts/run_taxonomy.py --models "$MODEL" 2>&1 | tee -a "$LOG"
touch "outputs/flags/${SLUG}_taxonomy.done"
sync "taxonomy done"

HC="outputs/taxonomy/$SLUG/heads.csv"
log "STAGE ablation"
python3 scripts/run_ablation.py --model "$MODEL" --heads-csv "$HC" \
    --probe synth_grounding --n-eval "$NEVAL" --device "$DEV" 2>&1 | tee -a "$LOG"
touch "outputs/flags/${SLUG}_ablation.done"
sync "S2 ablation done"

log "STAGE patching"
python3 scripts/run_patching.py --model "$MODEL" --heads-csv "$HC" \
    --probe grounding --role grounding --max-samples 40 --device "$DEV" 2>&1 | tee -a "$LOG"
touch "outputs/flags/${SLUG}_patching.done"
sync "S2 patching done"

log "STAGE rebias (splits x alphas x {vanilla,pai,ours} + vcd)"
python3 scripts/run_rebias.py --model "$MODEL" --heads-csv "$HC" \
    --alphas 0.5,1,2 --max-per-split "$RBS" --vcd --device "$DEV" 2>&1 | tee -a "$LOG"
touch "outputs/flags/${SLUG}_done.done"
sync "S3 rebias done"

log "PIPELINE_DONE $SLUG"
