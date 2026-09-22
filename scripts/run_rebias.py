import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from headatlas.interventions.rebias import (build_lambda_from_heads_csv,
                                            run_rebias)
from headatlas.models.wrapper import VLMWrapper, n_layers_heads
from headatlas.probes import loaders
from headatlas.probes.base import subsample


def slug(s):
    return s.replace("/", "_").replace(".", "_")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--heads-csv", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--splits", default="random,popular,adversarial")
    ap.add_argument("--alphas", default="0.5,1,2")
    ap.add_argument("--max-per-split", type=int, default=200)
    ap.add_argument("--vcd", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join("outputs", "rebias", slug(args.model))
    os.makedirs(out, exist_ok=True)
    wrapper = VLMWrapper(args.model, device=args.device)
    L, H = n_layers_heads(wrapper.model)
    lam = build_lambda_from_heads_csv(args.heads_csv, L, H)
    alphas = [float(a) for a in args.alphas.split(",")]
    splits = loaders.pope_splits(args.data)
    all_rows = []
    for split in args.splits.split(","):
        if split not in splits:
            continue
        samples = subsample(splits[split], args.max_per_split, seed=0)
        rows = run_rebias(wrapper, samples, lam, alphas,
                          os.path.join(out, f"{split}.csv"))
        for r in rows:
            r["split"] = split
            all_rows.append(r)
        print(split, "done")
    if args.vcd:
        from headatlas.eval.contrastive import vcd_generate
        from headatlas.eval.pope_runner import parse_yes_no, pope_metrics
        for split in args.splits.split(","):
            if split not in splits:
                continue
            samples = subsample(splits[split], min(50, args.max_per_split), seed=0)
            preds = []
            for s in samples:
                preds.append(parse_yes_no(vcd_generate(wrapper, s, alpha=1.0)))
            m = pope_metrics(preds, [s.answer for s in samples])
            m.update(method="vcd", alpha=1.0, split=split)
            all_rows.append(m)
            print("vcd", split, m["f1"])
    with open(os.path.join(out, "all_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "method", "alpha", "acc",
                                          "precision", "recall", "f1", "yes_rate"])
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print("DONE", out)
