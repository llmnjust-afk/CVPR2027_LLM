#!/usr/bin/env bash
# Aggregate experiment artifacts (CSVs / JSONs / figures / logs) into results/
# and push to GitHub. Raw .npz attention dumps stay on the lab machine only
# (too large for GitHub's 100MB file limit).
# Usage: scripts/sync_results.sh [commit message]
set -uo pipefail
cd "$(dirname "$0")/.."
MSG="${1:-sync results $(date -u +%F_%T)}"

mkdir -p results logs
copy_artifacts() {
    local src_root="$1"
    [ -d "$src_root" ] || return 0
    find "$src_root" \( -name '*.csv' -o -name '*.json' -o -name '*.png' \) \
        -type f 2>/dev/null | while read -r f; do
        rel="${f#outputs/}"
        mkdir -p "results/$(dirname "$rel")"
        cp -f "$f" "results/$rel"
    done
}
copy_artifacts outputs/taxonomy
copy_artifacts outputs/ablation
copy_artifacts outputs/patching
copy_artifacts outputs/rebias
for lg in logs/*.log; do
    [ -e "$lg" ] || break
    cp -f "$lg" "results/logs_$(basename "$lg")"
done

git add results/ 2>/dev/null || true
if git diff --cached --quiet 2>/dev/null; then
    echo "sync_results: nothing new to push"
    exit 0
fi
git commit -q -m "$MSG" || exit 0
git pull --rebase -q origin main 2>/dev/null || true
if ! git push -q origin main 2>&1 | grep -v '^$'; then
    echo "sync_results: push FAILED (check credentials)"
    exit 1
fi
echo "sync_results: pushed $(git log --oneline -1)"
