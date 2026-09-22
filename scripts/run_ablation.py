import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from headatlas.interventions.ablation import run_ablation
from headatlas.models.wrapper import VLMWrapper
from headatlas.probes import loaders


def slug(s):
    return s.replace("/", "_").replace(".", "_")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--heads-csv", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--probe", default="synth_grounding")
    ap.add_argument("--roles", default="grounding,cond,reroute,ocr,unassigned")
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--mode", default="zero", choices=["zero", "mean"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join("outputs", "ablation",
                                   slug(args.model), f"{args.probe}_{args.mode}.csv")
    wrapper = VLMWrapper(args.model, device=args.device)
    samples = loaders.load_probe(args.probe, args.data, max_samples=args.n_eval)
    roles = args.roles.split(",")
    res = run_ablation(wrapper, samples, args.heads_csv, roles, out,
                       n_eval=args.n_eval, mode=args.mode)
    for name, heads, score in res:
        print(f"{name:14s} {heads:40s} mass={score:.4f}")
    print("DONE", out)
