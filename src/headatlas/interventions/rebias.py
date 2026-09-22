from __future__ import annotations

import csv
import os

import numpy as np
import torch

from ..eval.pope_runner import pope_metrics, run_pope
from ..models.biased_attention import (BiasMode, build_kv_bias,
                                       clear_attention_bias, set_attention_bias)

LAMBDA_ALPHA = 1.0
LAMBDA_ALPHA_SINK = 1.0


def build_lambda_from_heads_csv(heads_csv, n_layers, n_heads,
                                alpha=LAMBDA_ALPHA, alpha_sink=LAMBDA_ALPHA_SINK,
                                alpha_unassigned=0.0):
    lam = np.zeros((n_layers, n_heads), dtype=np.float64)
    with open(heads_csv) as f:
        for r in csv.DictReader(f):
            l, h = int(r["layer"]), int(r["head"])
            if l >= n_layers or h >= n_heads:
                continue
            role = r["role"]
            if role == "grounding":
                lam[l, h] = alpha
            elif role in ("sink", "unassigned"):
                lam[l, h] = -alpha_sink if role == "sink" else alpha_unassigned
            else:
                lam[l, h] = alpha * 0.25
    return lam


def run_rebias(wrapper, samples, lam, alphas, out_csv, max_new_tokens=16,
               include_pai=True):
    """lam: [L,H] base lambda (per-head direction from taxonomy). Runs:
    vanilla, pai-uniform(+a all heads), ours(role-weighted lambda*a).
    Also dumps per-sample predictions to <out_csv>_preds.csv."""
    n_layers, n_heads = lam.shape
    rows = []
    pred_rows = []
    ids = [s.id for s in samples]
    base_lam = torch.as_tensor(lam)

    def _dump(method, alpha, preds):
        for sid, gt, p in zip(ids, [s.answer for s in samples], preds):
            pred_rows.append(dict(method=method, alpha=alpha, sample_id=sid,
                                  answer=gt, pred=p))

    vanilla_preds, _ = run_pope(wrapper, samples, max_new_tokens=max_new_tokens,
                                capture_boxes=False)
    gts = [s.answer for s in samples]
    rows.append(dict(method="vanilla", alpha=0.0, **pope_metrics(vanilla_preds, gts)))
    _dump("vanilla", 0.0, vanilla_preds)
    if include_pai:
        for a in alphas:
            lam_uniform = torch.full((n_layers, n_heads), float(a))
            preds, _ = run_pope(wrapper, samples, max_new_tokens=max_new_tokens,
                                capture_boxes=False, lam_per_head=lam_uniform)
            rows.append(dict(method="pai_uniform", alpha=a,
                             **pope_metrics(preds, gts)))
            _dump("pai_uniform", a, preds)
    for a in alphas:
        lam_ours = base_lam * float(a)
        preds, _ = run_pope(wrapper, samples, max_new_tokens=max_new_tokens,
                            capture_boxes=False, lam_per_head=lam_ours)
        rows.append(dict(method="ours_headselective", alpha=a,
                         **pope_metrics(preds, gts)))
        _dump("ours_headselective", a, preds)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "alpha", "acc", "precision",
                                          "recall", "f1", "yes_rate"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    preds_path = out_csv.replace(".csv", "_preds.csv")
    with open(preds_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "alpha", "sample_id",
                                          "answer", "pred"])
        w.writeheader()
        for r in pred_rows:
            w.writerow(r)
    return rows
