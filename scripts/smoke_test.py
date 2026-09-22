import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from headatlas.eval.pope_runner import parse_yes_no, pope_metrics
from headatlas.interventions.ablation import run_ablation
from headatlas.interventions.rebias import run_rebias
from headatlas.models.wrapper import VLMWrapper, n_layers_heads
from headatlas.probes import loaders
from headatlas.probes.synthetic import (build_sink_probe,
                                        make_bright_square_sample, make_ocr_sample)
from headatlas.probes.base import ProbeSample


def slug(s):
    return s.replace("/", "_").replace(".", "_")


def mini_pope_samples(n=4):
    samples = []
    for i in range(n):
        img = make_bright_square_sample(i)
        samples.append(ProbeSample(id=f"mp_{i}", prompt="Is there a bright object in this image?",
                                   image=img.image, boxes=img.boxes, answer="yes"))
        arr = np.asarray(img.image).copy()
        arr = np.clip((arr.astype(np.float32) * 0.3), 0, 255).astype(np.uint8)
        from PIL import Image
        dark = ProbeSample(id=f"mn_{i}", prompt="Is there a bright object in this image?",
                           image=Image.fromarray(arr), answer="no")
        samples.append(dark)
    return samples


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    wrapper = VLMWrapper(args.model, device=args.device)
    L, H = n_layers_heads(wrapper.model)
    print(f"loaded {args.model} L={L} H={H} ({time.time()-t0:.0f}s)")
    out = os.path.join("outputs", "raw", slug(args.model))
    os.makedirs(out, exist_ok=True)
    probes = []
    for i in range(args.n):
        probes.append(make_bright_square_sample(i, size=(448, 448)))
    from headatlas.metrics.spatial import box_to_mask, mass_in_box, tokens_to_grid
    from headatlas.models.attention_grabber import AttentionGrabber
    grabber = AttentionGrabber(wrapper.model)
    all_mass, n_tokens = [], None
    for s in probes:
        inputs = wrapper.build_inputs(s)
        img_pos = wrapper.image_token_positions(inputs["input_ids"])
        plen = wrapper.prompt_len(inputs)
        with grabber.capture(mode="rows", rows={"ans": plen - 1}, cols=img_pos):
            wrapper.forward(inputs, attentions=True)
        row = grabber.results["ans"][..., 0, :].numpy()
        grid = wrapper.grid_hw(inputs)
        mask = box_to_mask(s.boxes[0], (448, 448), wrapper.resized_hw(inputs), grid)
        mask_grid = tokens_to_grid(mask, grid)
        n_tokens = row.shape[-1]
        for l in range(row.shape[0]):
            for h in range(row.shape[1]):
                all_mass.append(mass_in_box(tokens_to_grid(row[l, h], grid), mask_grid))
    all_mass = np.array(all_mass)
    chance = 2.0 / n_tokens
    print(f"synthetic grounding: best-head mass={all_mass.max():.3f}, "
          f"mean={all_mass.mean():.3f}, chance={chance:.4f}")
    assert all_mass.max() > max(0.05, 6 * chance), \
        "no head focuses on the GT box; mapping likely broken"
    ocr = [make_ocr_sample(i, size=(448, 448)) for i in range(args.n)]
    texts = []
    for s in ocr:
        inputs = wrapper.build_inputs(s)
        texts.append(wrapper.generate(inputs, max_new_tokens=16))
    print("ocr answers:", texts)
    sink_samples = build_sink_probe(n=args.n)
    s = sink_samples[0]
    inputs = wrapper.build_inputs(s)
    plen = wrapper.prompt_len(inputs)
    img_pos = wrapper.image_token_positions(inputs["input_ids"])
    with grabber.capture(mode="query_block", row_mask=wrapper.image_token_mask(inputs["input_ids"]), first_k=16):
        wrapper.forward(inputs, attentions=True)
    block = grabber.results["block"].numpy()
    print(f"sink first-16 mass per head mean={block.mean():.4f}")
    pope = mini_pope_samples(n=args.n)
    preds, rows = None, None
    from headatlas.eval.pope_runner import run_pope
    preds, rows = run_pope(wrapper, pope, capture_boxes=True, max_new_tokens=8)
    m = pope_metrics(preds, [s.answer for s in pope])
    print(f"mini-POPE: acc={m['acc']:.2f} f1={m['f1']:.2f} preds={preds}")
    heads_csv = os.path.join(out, "heads_smoke.csv")
    with open(heads_csv, "w") as f:
        f.write("layer,head,cluster,role,p1_mass_in_box,p5_sink16\n")
        for l in range(L):
            for h in range(H):
                role = "grounding" if (l, h) == (0, 0) else "unassigned"
                f.write(f"{l},{h},0,{role},0.0,0.0\n")
    ab = run_ablation(wrapper, probes, heads_csv, ["grounding"], 
                      os.path.join(out, "ablation_smoke.csv"), n_eval=args.n)
    for name, heads, score in ab:
        print(f"ablation {name}: {score:.4f}")
    lam = __import__("numpy").ones((L, H))
    rb = run_rebias(wrapper, pope[: args.n], lam, [1.0],
                    os.path.join(out, "rebias_smoke.csv"), max_new_tokens=8,
                    include_pai=False)
    for r in rb:
        print("rebias", r["method"], r["f1"])
    print(f"SMOKE_OK total={time.time()-t0:.0f}s")
