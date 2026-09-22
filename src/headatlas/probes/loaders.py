from __future__ import annotations

import json
import os

from .base import ProbeSample, load_jsonl, subsample

PROBE_FILES = {
    "grounding": "refcoco.jsonl",
    "spatial": "gqa.jsonl",
    "pope": "pope.jsonl",
    "clevr": "clevr.jsonl",
    "ocr_real": "textocr.jsonl",
}

SYNTHETIC_PROBES = {"sink", "ocr_synth", "synth_grounding"}


def load_probe(name, data_dir="data", max_samples=400, seed=0):
    if name in SYNTHETIC_PROBES:
        from . import synthetic
        if name == "sink":
            return synthetic.build_sink_probe(n=max_samples)
        if name == "ocr_synth":
            return synthetic.build_ocr_probe(n=max_samples)
        return synthetic.build_synthetic_grounding(n=max_samples)
    fname = PROBE_FILES.get(name, f"{name}.jsonl")
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; run scripts/prepare_data.py or use a synthetic probe")
    return subsample(load_jsonl(path), max_samples, seed=seed)


def pope_splits(data_dir="data"):
    out = {}
    for split in ("random", "popular", "adversarial"):
        path = os.path.join(data_dir, f"pope_{split}.jsonl")
        if os.path.exists(path):
            out[split] = load_jsonl(path)
    if not out and os.path.exists(os.path.join(data_dir, "pope.jsonl")):
        out["random"] = load_jsonl(os.path.join(data_dir, "pope.jsonl"))
    return out


def dump_canonical(path, samples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_json()) + "\n")
