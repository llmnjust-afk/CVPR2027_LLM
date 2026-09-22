from __future__ import annotations

import json
import os

import numpy as np

from .spatial import (attention_entropy, box_iou_topk, mass_in_box, pointing_hit,
                      sym_kld, tokens_to_grid, topk_precision)


def _padded_stack(arrays, pad=float("nan")) -> np.ndarray:
    cmax = max(a.shape[-1] for a in arrays)
    out = []
    for a in arrays:
        if a.shape[-1] < cmax:
            p = np.full(a.shape[:-1] + (cmax - a.shape[-1],), pad, dtype=a.dtype)
            a = np.concatenate([a, p], axis=-1)
        out.append(a)
    return np.stack(out, axis=0)


def compute_probe_features(probe, npz, data_dir=None):
    """probe -> dict feature_name -> [L, H] float arrays."""
    L, H = npz["rows"].shape[0], npz["rows"].shape[1]
    feats = {}
    if probe in ("grounding", "ocr", "ocr_synth", "synth_grounding", "spatial"):
        rows = npz["rows"]
        masks = npz["box_masks"]
        grids = npz["grids"]
        n = rows.shape[2]
        acc = {k: np.zeros((L, H)) for k in
               ("mass_in_box", "topk_prec", "pointing", "box_iou", "entropy")}
        for i in range(n):
            mask = masks[i]
            if not np.isfinite(mask).all():
                continue
            if mask.sum() == 0:
                continue
            for l in range(L):
                for h in range(H):
                    row = rows[l, h, i]
                    valid = np.isfinite(row)
                    if not valid.any():
                        continue
                    r = row[valid]
                    m = mask[valid]
                    g = tokens_to_grid(r, grids[i])
                    gm = tokens_to_grid(m, grids[i])
                    acc["mass_in_box"][l, h] += mass_in_box(g, gm)
                    acc["topk_prec"][l, h] += topk_precision(g, gm)
                    acc["pointing"][l, h] += pointing_hit(g, gm)
                    acc["box_iou"][l, h] += box_iou_topk(g, gm)
                    acc["entropy"][l, h] += attention_entropy(r)
        cnt = max(1, sum(1 for i in range(n) if np.isfinite(masks[i]).all()
                         and masks[i].sum() > 0))
        feats = {f"p1_{k}": v / cnt for k, v in acc.items()}
    elif probe == "cond":
        a = npz["rows_attr"]
        b = npz["rows_plain"]
        n = min(a.shape[2], b.shape[2])
        kl = np.zeros((L, H))
        for i in range(n):
            for l in range(L):
                for h in range(H):
                    kl[l, h] += sym_kld(a[l, h, i], b[l, h, i])
        feats = {"p3_cond_kld": kl / max(1, n)}
    elif probe == "sink":
        rows = npz["rows16"]
        mass = rows.sum(axis=-1)
        feats = {
            "p5_sink16": mass.mean(axis=2),
            "p5_sink_stability": 1.0 - (mass.std(axis=2) / np.maximum(mass.mean(axis=2), 1e-8)),
            "p5_head_entropy": np.zeros((L, H)),
        }
        n = rows.shape[2]
        ent = np.zeros((L, H))
        for i in range(n):
            for l in range(L):
                for h in range(H):
                    ent[l, h] += attention_entropy(rows[l, h, i])
        feats["p5_head_entropy"] = ent / max(1, n)
    elif probe == "occlusion":
        base = npz["rows_base"]
        occl = npz["rows_occl"]
        masks = npz["box_masks"]
        grids = npz["grids"]
        n = occl.shape[2]
        omass = np.zeros((L, H))
        shift = np.zeros((L, H))
        cnt = 0
        for i in range(n):
            mask = masks[i]
            if not np.isfinite(mask).all() or mask.sum() == 0:
                continue
            cnt += 1
            for l in range(L):
                for h in range(H):
                    ro = occl[l, h, i]
                    valid = np.isfinite(ro)
                    g = tokens_to_grid(ro[valid], grids[i])
                    gm = tokens_to_grid(mask[valid], grids[i])
                    omass[l, h] += mass_in_box(g, gm)
                    rb = base[l, h, i][valid]
                    shift[l, h] += sym_kld(rb, ro)
        cnt = max(1, cnt)
        feats = {"p6_occluded_mass": omass / cnt, "p6_reroute_kld": shift / cnt}
    elif probe == "pope":
        rows = npz["rows"]
        masks = npz["box_masks"]
        correct = npz["correct"]
        grids = npz["grids"]
        L0, H0, n = rows.shape[:3]
        corr = np.zeros((L, H))
        cnt = 0
        for i in range(n):
            mask = masks[i]
            if not np.isfinite(mask).all() or mask.sum() == 0:
                continue
            cnt += 1
            for l in range(L):
                for h in range(H):
                    r = rows[l, h, i]
                    valid = np.isfinite(r)
                    g = tokens_to_grid(r[valid], grids[i])
                    gm = tokens_to_grid(mask[valid], grids[i])
                    m = mass_in_box(g, gm)
                    c = 1.0 if correct[i] else 0.0
                    corr[l, h] += m * c
        feats = {"p8_mass_when_correct": corr / max(1, cnt)}
    else:
        raise ValueError(f"unknown probe {probe}")
    return feats


def collect_model_fingerprints(raw_dir, probes, out_path):
    all_feats = {}
    for probe in probes:
        path = os.path.join(raw_dir, f"{probe}.npz")
        if not os.path.exists(path):
            continue
        npz = dict(np.load(path, allow_pickle=True))
        feats = compute_probe_features(probe, npz)
        for k, v in feats.items():
            all_feats[k] = v
    keys = sorted(all_feats.keys())
    X = np.stack([all_feats[k] for k in keys], axis=-1)
    np.savez(out_path, features=X, names=np.array(keys), **all_feats)
    csv_path = out_path.replace(".npz", ".csv")
    with open(csv_path, "w") as f:
        f.write("layer,head," + ",".join(keys) + "\n")
        L, H = X.shape[0], X.shape[1]
        for l in range(L):
            for h in range(H):
                f.write(f"{l},{h}," + ",".join(f"{X[l, h, j]:.6f}"
                                               for j in range(len(keys))) + "\n")
    return all_feats, keys
