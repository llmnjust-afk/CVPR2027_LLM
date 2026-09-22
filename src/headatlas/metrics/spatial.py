from __future__ import annotations

import numpy as np


def tokens_to_grid(vals, grid_hw):
    h, w = grid_hw
    v = np.asarray(vals, dtype=np.float32)
    if v.size != h * w:
        raise ValueError(f"token count {v.size} != grid {h}x{w}")
    return v.reshape(h, w)


def box_to_mask(box, orig_hw, resized_hw, grid_hw):
    """Binary grid mask of GT box: cells whose center falls inside the box
    (coordinates mapped original -> resized -> grid)."""
    ox, oy = orig_hw
    rx, ry = resized_hw
    gh, gw = grid_hw
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = x1 * rx / ox
    x2 = x2 * rx / ox
    y1 = y1 * ry / oy
    y2 = y2 * ry / oy
    mask = np.zeros((gh, gw), dtype=np.float32)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return mask
    for i in range(gh):
        cy = (i + 0.5) * ry / gh
        for j in range(gw):
            cx = (j + 0.5) * rx / gw
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                mask[i, j] = 1.0
    return mask


def mass_in_box(grid_vals, mask):
    total = float(grid_vals.sum())
    if total <= 0:
        return 0.0
    return float((grid_vals * mask).sum() / total)


def topk_precision(grid_vals, mask, frac=0.1):
    k = max(1, int(round(frac * mask.size)))
    flat = grid_vals.flatten()
    idx = np.argsort(flat)[::-1][:k]
    selected = np.zeros_like(flat, dtype=bool)
    selected[idx] = True
    sel = selected.reshape(mask.shape)
    return float((sel * (mask > 0)).sum() / max(1, sel.sum()))


def pointing_hit(grid_vals, mask):
    idx = int(np.argmax(grid_vals))
    i, j = np.unravel_index(idx, grid_vals.shape)
    return float(mask[i, j] > 0)


def box_iou_topk(grid_vals, mask, frac=0.1):
    k = max(1, int(round(frac * mask.size)))
    flat = grid_vals.flatten()
    idx = np.argsort(flat)[::-1][:k]
    sel = np.zeros_like(flat, dtype=bool)
    sel[idx] = True
    sel = sel.reshape(mask.shape).astype(np.float32)
    inter = float((sel * (mask > 0)).sum())
    union = float((sel.sum() + (mask > 0).sum() - inter))
    return inter / max(1.0, union)


def attention_entropy(vals):
    v = np.asarray(vals, dtype=np.float64)
    s = v.sum()
    if s <= 0:
        return 0.0
    p = v / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def sym_kld(p, q, eps=1e-8):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    kl_pq = (p * np.log((p + eps) / (q + eps))).sum()
    kl_qp = (q * np.log((q + eps) / (p + eps))).sum()
    return float(0.5 * (kl_pq + kl_qp))
