from __future__ import annotations

import csv
import os

import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROLE_FEATURES = {
    "grounding": ["p1_mass_in_box", "p1_topk_prec", "p1_box_iou", "synth_mass_in_box"],
    "ocr": ["p4_mass_in_box", "ocr_mass_in_box"],
    "sink": ["p5_sink16"],
    "cond": ["p3_cond_kld"],
    "reroute": ["p6_reroute_kld", "p6_occluded_mass"],
    "halluc": ["p8_mass_when_correct"],
}


def layerwise_z(X, layer_ids):
    Z = np.zeros(X.shape, dtype=np.float64)
    for l in np.unique(layer_ids):
        m = layer_ids == l
        mu = np.nanmean(X[m], axis=0)
        sd = np.nanstd(X[m], axis=0) + 1e-8
        Z[m] = (X[m] - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def cluster_heads(X, layer_ids, k_min=4, k_max=10, seed=0):
    Z = layerwise_z(X, layer_ids)
    best = None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        sil = float(silhouette_score(Z, km.labels_)) if k > 1 else 0.0
        if best is None or sil > best[1]:
            best = (k, sil, km.labels_, km.cluster_centers_)
    k, sil, labels, centers = best
    return dict(k=k, silhouette=sil, labels=labels, centers=centers, Z=Z)


def guess_roles(feature_names, X, layer_ids, threshold=1.0):
    Z = layerwise_z(X, layer_ids)
    roles = []
    for i in range(X.shape[0]):
        best_role, best_score = "unassigned", threshold
        for role, feats in ROLE_FEATURES.items():
            idx = [j for j, n in enumerate(feature_names) if n in feats]
            if not idx:
                continue
            score = float(np.mean(Z[i, idx]))
            if score > best_score:
                best_role, best_score = role, score
        roles.append(best_role)
    return np.array(roles)


def plot_embedding(Z, labels, layer_ids, out_png, method="pca"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    coords = None
    if method == "umap":
        try:
            import umap
            coords = umap.UMAP(n_neighbors=30, min_dist=0.2, random_state=0
                               ).fit_transform(Z)
        except Exception:
            method = "pca"
    if coords is None:
        coords = PCA(n_components=2, random_state=0).fit_transform(Z)
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=8)
    plt.title(f"Head Atlas ({method}), colored by cluster")
    plt.xlabel("dim 1"); plt.ylabel("dim 2")
    plt.colorbar(sc)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_layer_head_heatmap(scores, out_png, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.imshow(scores, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.xlabel("head"); plt.ylabel("layer")
    plt.title(title or "layer x head heatmap")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def export_heads_csv(feature_names, X, labels, roles, layer_ids, out_csv,
                     n_heads=1):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", "cluster", "role"] + list(feature_names))
        for i in range(X.shape[0]):
            w.writerow([int(layer_ids[i]), i % n_heads, int(labels[i]), roles[i]]
                       + [f"{v:.6f}" for v in X[i]])


def align_clusters(Z_a, labels_a, Z_b, labels_b, k=None):
    """Hungarian matching of cluster centroids across two models by cosine.
    Handles different k between models (rectangular assignment)."""
    from scipy.optimize import linear_sum_assignment
    k_a = int(labels_a.max()) + 1
    k_b = int(labels_b.max()) + 1
    ca = np.stack([Z_a[labels_a == c].mean(axis=0) for c in range(k_a)])
    cb = np.stack([Z_b[labels_b == c].mean(axis=0) for c in range(k_b)])
    sim = np.zeros((k_a, k_b))
    for i in range(k_a):
        for j in range(k_b):
            na, nb = np.linalg.norm(ca[i]), np.linalg.norm(cb[j])
            sim[i, j] = float(ca[i] @ cb[j] / max(na * nb, 1e-8))
    rows, cols = linear_sum_assignment(-sim)
    return dict(zip(rows.tolist(), cols.tolist())), sim
