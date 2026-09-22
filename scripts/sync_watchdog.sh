#!/usr/bin/env bash
# Watchdog loop (run under nohup):
#   1. when BOTH models have taxonomy done flags -> run cross-model alignment
#   2. push any new results/ artifacts to GitHub
#   3. exit when BOTH pipelines are fully done (final sync included)
# Usage: nohup bash scripts/sync_watchdog.sh > logs/watchdog.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

models_line() { cat outputs/flags/models.txt 2>/dev/null || echo ""; }

while true; do
    NTAX=$(ls outputs/flags/*_taxonomy.done 2>/dev/null | wc -l)
    NDONE=$(ls outputs/flags/*_done.done 2>/dev/null | wc -l)

    if [ "$NTAX" -ge 2 ] && [ ! -f outputs/taxonomy/cross_model_alignment.json ]; then
        MODELS=$(models_line)
        if [ -n "$MODELS" ]; then
            echo "[$(date '+%F %T')] running cross-model alignment" >> logs/watchdog.log
            PYTHONPATH=src python scripts/run_taxonomy.py --models $MODELS \
                >> logs/alignment.log 2>&1 || true
            bash scripts/sync_results.sh "watchdog: cross-model alignment" || true
        fi
    fi

    bash scripts/sync_results.sh "watchdog sync $(date -u +%F_%T)" \
        >> logs/watchdog.log 2>&1 || true

    if [ "$NDONE" -ge 2 ]; then
        echo "[$(date '+%F %T')] both pipelines done; final sync + exit" >> logs/watchdog.log
        bash scripts/sync_results.sh "watchdog: all pipelines complete" || true
        exit 0
    fi
    sleep 1800
done
