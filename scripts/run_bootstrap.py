import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def f1_of(pred_yes, gt_yes):
    tp = int(np.sum(pred_yes & gt_yes))
    fp = int(np.sum(pred_yes & ~gt_yes))
    fn = int(np.sum(~pred_yes & gt_yes))
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / max(1e-9, p + r)


def acc_of(pred_yes, gt_yes):
    return float(np.mean(pred_yes == gt_yes))


def boot_delta(pa, pb, gtv, rng, reps):
    """Paired bootstrap of delta F1 / delta accuracy over resampled samples."""
    n = len(pa)
    idx = rng.integers(0, n, size=(reps, n))
    df1 = np.empty(reps)
    dacc = np.empty(reps)
    for i in range(reps):
        m = idx[i]
        g = gtv[m]
        df1[i] = f1_of(pa[m], g) - f1_of(pb[m], g)
        dacc[i] = (pa[m] == g).mean() - (pb[m] == g).mean()
    return df1, dacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--preds-dir", required=True,
                    help="dir with {split}_preds.csv sidecars")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gts = {}
    preds = defaultdict(dict)
    split_names = set()
    for fn in sorted(os.listdir(args.preds_dir)):
        if not fn.endswith("_preds.csv"):
            continue
        split = fn[:-len("_preds.csv")]
        split_names.add(split)
        with open(os.path.join(args.preds_dir, fn)) as f:
            for r in csv.DictReader(f):
                preds[(split, r["method"], float(r["alpha"]))][r["sample_id"]] \
                    = (r["pred"] == "yes")
                gts[(split, r["sample_id"])] = (r["answer"] == "yes")
    split_names = sorted(split_names)
    assert split_names, f"no *_preds.csv under {args.preds_dir}"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ["model", "split", "alpha", "compare", "f1_a", "f1_b",
              "delta_f1", "ci_low", "ci_high", "p_f1", "acc_a", "acc_b",
              "delta_acc", "acc_ci_low", "acc_ci_high", "p_acc", "n"]
    out_rows = []

    def vec(split, method, alpha):
        tab = preds.get((split, method, alpha))
        return tab

    for tag in split_names + ["pooled"]:
        pooled = tag == "pooled"
        if pooled:
            ids = [(s, i) for s in split_names
                   for i in sorted(preds[(s, "vanilla", 0.0)].keys())]
            gtv = np.array([gts[(s, i)] for s, i in ids])

            def get(m, a, i):
                s, sid = i
                return preds.get((s, m, a), {}).get(sid, False)
        else:
            ids = sorted(preds[(tag, "vanilla", 0.0)].keys())
            gtv = np.array([gts[(tag, i)] for i in ids])

            def get(m, a, i, _tag=tag):
                return preds.get((_tag, m, a), {}).get(i, False)
        vanilla = np.array([get("vanilla", 0.0, i) for i in ids])
        alphas = sorted({a for (s, m, a) in preds
                         if m != "vanilla" and (pooled or s == tag)})
        for alpha in alphas:
            ours = np.array([get("ours_headselective", alpha, i) for i in ids])
            pai = np.array([get("pai_uniform", alpha, i) for i in ids])
            for cmp_name, other_vec in (("ours_vs_vanilla", vanilla),
                                        ("ours_vs_pai", pai)):
                rng = np.random.RandomState(args.seed)
                df1, dacc = boot_delta(ours, other_vec, gtv, rng, args.reps)
                point = f1_of(ours, gtv) - f1_of(other_vec, gtv)
                lo, hi = np.percentile(df1, [2.5, 97.5])
                p_left = float(np.mean(df1 <= 0))
                p_f1 = 2 * min(p_left, 1 - p_left)
                acc_a, acc_b = acc_of(ours, gtv), acc_of(other_vec, gtv)
                alo, ahi = np.percentile(dacc, [2.5, 97.5])
                p_al = float(np.mean(dacc <= 0))
                p_acc = 2 * min(p_al, 1 - p_al)
                out_rows.append(dict(
                    model=args.model, split=tag, alpha=alpha,
                    compare=cmp_name, f1_a=f1_of(ours, gtv),
                    f1_b=f1_of(other_vec, gtv), delta_f1=point,
                    ci_low=lo, ci_high=hi, p_f1=p_f1, acc_a=acc_a,
                    acc_b=acc_b, delta_acc=acc_a - acc_b,
                    acc_ci_low=alo, acc_ci_high=ahi, p_acc=p_acc, n=len(ids)))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print("bootstrap rows:", len(out_rows), "->", args.out)


if __name__ == "__main__":
    main()
