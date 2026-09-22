import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from headatlas.interventions.ablation import heads_of_role
from headatlas.interventions.patching import run_patching
from headatlas.models.wrapper import VLMWrapper
from headatlas.probes import loaders


def slug(s):
    return s.replace("/", "_").replace(".", "_")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--heads-csv", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--probe", default="grounding")
    ap.add_argument("--role", default="grounding")
    ap.add_argument("--max-samples", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join("outputs", "patching", slug(args.model),
                                   f"{args.role}.csv")
    wrapper = VLMWrapper(args.model, device=args.device)
    samples = loaders.load_probe(args.probe, args.data, max_samples=args.max_samples)
    heads = heads_of_role(args.heads_csv, args.role, k=3)
    res = run_patching(wrapper, samples, heads, out)
    print(f"n={res['n']} flip_rate={res['flip_rate']:.3f}")
    print("DONE", out)
