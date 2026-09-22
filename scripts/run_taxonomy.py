import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from headatlas.metrics.fingerprints import collect_model_fingerprints
from headatlas.metrics.taxonomy import (align_clusters, cluster_heads,
                                        export_heads_csv, guess_roles,
                                        plot_embedding, plot_layer_head_heatmap)


def slug(s):
    return s.replace("/", "_").replace(".", "_")


def process_model(raw_dir, out_dir, probes):
    os.makedirs(out_dir, exist_ok=True)
    feats, keys = collect_model_fingerprints(raw_dir, probes,
                                             os.path.join(out_dir, "fingerprints.npz"))
    X = feats[keys[0]].reshape(-1, 1)
    L, H = feats[keys[0]].shape
    X = np.stack([feats[k] for k in keys], axis=-1).reshape(L * H, len(keys))
    layer_ids = np.repeat(np.arange(L), H)
    cl = cluster_heads(X, layer_ids, k_min=3, k_max=8)
    roles = guess_roles(keys, X, layer_ids)
    export_heads_csv(keys, X, cl["labels"], roles, layer_ids,
                     os.path.join(out_dir, "heads.csv"))
    plot_embedding(cl["Z"], cl["labels"], layer_ids,
                   os.path.join(out_dir, "fig_atlas.png"))
    for fname in keys:
        scores = feats[fname].reshape(L, H)
        plot_layer_head_heatmap(scores,
                                os.path.join(out_dir, f"fig_heat_{fname}.png"),
                                title=fname)
    summary = dict(k=cl["k"], silhouette=cl["silhouette"], n_heads=L * H,
                   features=keys,
                   role_counts={r: int((roles == r).sum())
                                for r in set(roles.tolist())})
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return cl, roles, keys, X, layer_ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--raw-root", default="outputs/raw")
    ap.add_argument("--out-root", default="outputs/taxonomy")
    ap.add_argument("--probes", default="synth_grounding,ocr_synth,sink,cond,occlusion,pope")
    args = ap.parse_args()
    probes = [p for p in args.probes.split(",") if p]
    results = {}
    for mid in args.models:
        s = slug(mid)
        raw_dir = os.path.join(args.raw_root, s)
        if not os.path.isdir(raw_dir):
            print("missing raw dir", raw_dir)
            continue
        results[mid] = process_model(raw_dir, os.path.join(args.out_root, s), probes)
    if len(results) == 2:
        (m1, r1), (m2, r2) = results.items()
        mapping, sim = align_clusters(r1[0]["Z"], r1[0]["labels"],
                                      r2[0]["Z"], r2[0]["labels"], r1[0]["k"])
        out = os.path.join(args.out_root, "cross_model_alignment.json")
        with open(out, "w") as f:
            json.dump(dict(models=[m1, m2], mapping=mapping,
                           sim=sim.tolist()), f, indent=2)
        print("wrote", out)
