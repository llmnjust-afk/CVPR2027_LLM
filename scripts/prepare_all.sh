#!/usr/bin/env bash
# One-shot data preparation for the HeadAtlas probe battery:
#   COCO val2014 instances json (for POPE GT boxes) -> POPE 3 splits ->
#   RefCOCO refs pkl (best-effort) -> lazy COCO image download.
# Safe to re-run (existing files are skipped).
set -uo pipefail
cd "$(dirname "$0")/.."
DATA="${1:-data}"
mkdir -p "$DATA" logs
export PYTHONPATH=src

log() { echo "[$(date '+%F %T')] $*" | tee -a logs/prepare_data.log; }

INST="$DATA/instances_val2014.json"
if [ ! -f "$INST" ]; then
    log "downloading COCO val2014 annotations (~240MB)"
    for u in "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"; do
        curl -fsSL --retry 3 -o "$DATA/ann.zip" "$u" && break
    done
    if [ -f "$DATA/ann.zip" ]; then
        python3 - "$DATA/ann.zip" "$DATA" <<'EOF'
import sys, zipfile, os
zpath, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zpath) as z:
    z.extract("annotations/instances_val2014.json", out)
os.replace(os.path.join(out, "annotations", "instances_val2014.json"),
           os.path.join(out, "instances_val2014.json"))
EOF
        rm -f "$DATA/ann.zip"
        log "instances json ready"
    else
        log "FAILED to download COCO annotations; POPE will have no GT boxes"
    fi
fi

log "preparing POPE splits (raw github primary, HF fallback, outer retry)"
pope_missing() {
    for s in random popular adversarial; do
        [ -f "$DATA/pope_$s.jsonl" ] || return 0
    done
    return 1
}
if pope_missing; then
    if [ -f "$INST" ]; then
        python3 scripts/prepare_data.py --data "$DATA" --pope --coco-instances "$INST" 2>&1 | tee -a logs/prepare_data.log
    else
        python3 scripts/prepare_data.py --data "$DATA" --pope 2>&1 | tee -a logs/prepare_data.log
    fi
fi
if pope_missing; then
    log "POPE via HF mirror (lmms-lab-encoder/POPE)"
    python3 scripts/prepare_pope_hf.py --data "$DATA" --instances "$INST" 2>&1 | tee -a logs/prepare_data.log
fi
if pope_missing; then
    log "network flap? retrying raw POPE once more after 45s"
    sleep 45
    python3 scripts/prepare_data.py --data "$DATA" --pope --coco-instances "$INST" 2>&1 | tee -a logs/prepare_data.log
fi

REFS="$DATA/refs(unc).pkl"
if [ ! -f "$REFS" ]; then
    log "trying RefCOCO refs(unc).pkl download (UNC, may be blocked)"
    curl -fsSL --retry 1 --max-time 60 -o "$REFS" \
        "https://bvisionweb1.cs.unc.edu/public/downloads/refclef/refs(unc).pkl" \
        2>/dev/null || rm -f "$REFS" || true
fi
if [ -f "$REFS" ] && [ -f "$INST" ] && [ ! -f "$DATA/refcoco.jsonl" ]; then
    log "converting RefCOCO from pkl"
    python3 scripts/prepare_data.py --data "$DATA" --refcoco "$REFS" "$INST" 2>&1 | tee -a logs/prepare_data.log
fi
if [ ! -f "$DATA/refcoco.jsonl" ]; then
    log "building RefCOCO from HF mirror (jxu124/refcoco)"
    python3 scripts/prepare_refcoco_hf.py --data "$DATA" --max-samples 800 2>&1 | tee -a logs/prepare_data.log
fi
if [ ! -f "$DATA/refcoco.jsonl" ]; then
    log "WARNING: refcoco.jsonl missing; grounding/cond/occlusion probes will be skipped"
fi

log "lazy-downloading COCO images val2014+train2014 (capped)"
python3 scripts/prepare_data.py --data "$DATA" --ensure-images --max-missing 12000 2>&1 | tee -a logs/prepare_data.log

NIMG=$(find "$DATA/images" -name '*.jpg' 2>/dev/null | wc -l)
log "images on disk: $NIMG"
touch "$DATA/.prepped"
log "PREP_DONE"
