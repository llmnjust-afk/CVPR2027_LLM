from __future__ import annotations

import csv
import os

import numpy as np
import torch

from ..eval.pope_runner import pope_metrics, run_pope
from ..models.attention_grabber import AttentionGrabber
from ..metrics.spatial import mass_in_box, tokens_to_grid, topk_precision
from .head_ops import HeadEditor


def heads_of_role(heads_csv, role, k=3, order_key=None):
    rows = []
    with open(heads_csv) as f:
        for r in csv.DictReader(f):
            if r["role"] == role:
                rows.append(r)
    if order_key:

        def key_fn(r):
            try:
                return -float(r.get(order_key, 0.0) or 0.0)
            except ValueError:
                return 0.0
        rows = sorted(rows, key=key_fn)
    return [(int(r["layer"]), int(r["head"])) for r in rows[:k]]


def grounding_metric(wrapper, samples, n=60):
    grabber = AttentionGrabber(wrapper.model)
    scores = []
    for s in samples[:n]:
        inputs = wrapper.build_inputs(s)
        img_pos = wrapper.image_token_positions(inputs["input_ids"])
        plen = wrapper.prompt_len(inputs)
        with grabber.capture(mode="rows", rows={"ans": plen - 1}, cols=img_pos):
            wrapper.forward(inputs, attentions=True)
        row = grabber.results["ans"][..., 0, :].numpy()
        grid = wrapper.grid_hw(inputs)
        rw, rh = wrapper.resized_hw(inputs)
        if s.image is not None:
            ow, oh = s.image.size
        elif s.image_path:
            from PIL import Image as PILImage
            ow, oh = PILImage.open(s.image_path).size
        else:
            ow, oh = 448, 448
        mask = None
        from ..metrics.spatial import box_to_mask
        if s.boxes:
            mask = box_to_mask(s.boxes[0], (ow, oh), (rw, rh), grid)
        gm = tokens_to_grid(mask, grid) if s.boxes else None
        if gm is None:
            continue
        for l in range(row.shape[0]):
            for h in range(row.shape[1]):
                g = tokens_to_grid(row[l, h], grid)
                scores.append(mass_in_box(g, gm))
    return float(np.mean(scores)) if scores else 0.0


def run_ablation(wrapper, probe_samples, heads_csv, roles, out_csv,
                 n_eval=60, mode="zero", n_calib=12, control_role="sink"):
    """For each role: ablate its top-k heads and measure delta on the grounding
    probe; includes random-head and sink-head controls with matched head count."""
    editor = HeadEditor(wrapper.model)
    base = grounding_metric(wrapper, probe_samples)
    results = [("baseline", "", base)]
    controls = {
        "random": [(np.random.randint(0, editor.n_layers - 1),
                    np.random.randint(0, editor.n_heads))
                   for _ in range(3)],
        control_role: heads_of_role(heads_csv, control_role, k=3),
    }
    plans = {}
    for role in roles:
        plans[role] = heads_of_role(heads_csv, role, k=3)
    plans.update(controls)
    means = None
    if mode == "mean":
        means = editor.compute_means(wrapper.model, iter(probe_samples), n_calib=n_calib)
    for name, heads in plans.items():
        if not heads:
            continue
        plan = {}
        for (l, h) in heads:
            plan.setdefault(l, {})[h] = None
        with editor.ablate(plan, mode=mode, means=means):
            score = grounding_metric(wrapper, probe_samples, n=n_eval)
        results.append((name, str(heads), score))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plan", "heads", "grounding_mass"])
        for name, heads, score in results:
            w.writerow([name, heads, f"{score:.6f}"])
    return results
