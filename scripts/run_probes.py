import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from headatlas.models.attention_grabber import AttentionGrabber
from headatlas.models.wrapper import VLMWrapper, n_layers_heads
from headatlas.metrics.fingerprints import _padded_stack
from headatlas.metrics.spatial import box_to_mask
from headatlas.probes import loaders
from headatlas.probes.synthetic import build_conditioning_pairs, occlude_sample


def slug(s):
    return s.replace("/", "_").replace(".", "_")


def meta_json(grid_list, resized_list, ids, extra=None):
    return dict(grids=grid_list, resized=resized_list, ids=ids, extra=extra or {})


def compute_masks(wrapper, samples, inputs_cache):
    masks = []
    for s in samples:
        inputs = inputs_cache[s.id]
        grid = wrapper.grid_hw(inputs)
        rw, rh = wrapper.resized_hw(inputs)
        if s.image is not None:
            ow, oh = s.image.size
        else:
            from PIL import Image
            ow, oh = Image.open(s.image_path).size
        if s.boxes:
            m = box_to_mask(s.boxes[0], (ow, oh), (rw, rh), grid).flatten()
        else:
            m = np.zeros(grid[0] * grid[1], dtype=np.float32)
        masks.append(m)
    return masks


def run_box_probe(wrapper, samples, tag, out_dir):
    grabber = AttentionGrabber(wrapper.model)
    rows, cache, grids, resized, ids = [], {}, [], [], []
    for s in samples:
        inputs = wrapper.build_inputs(s)
        cache[s.id] = inputs
        img_pos = wrapper.image_token_positions(inputs["input_ids"])
        plen = wrapper.prompt_len(inputs)
        t0 = time.time()
        with grabber.capture(mode="rows", rows={"ans": plen - 1}, cols=img_pos):
            wrapper.forward(inputs, attentions=True)
        rows.append(grabber.results["ans"][..., 0, :].numpy())
        grids.append(wrapper.grid_hw(inputs))
        resized.append(wrapper.resized_hw(inputs))
        ids.append(s.id)
        print(f"  {tag} {s.id} {time.time()-t0:.1f}s", flush=True)
    masks = compute_masks(wrapper, samples, cache)
    cmax = max(r.shape[-1] for r in rows)
    rows_pad = np.stack([np.pad(r, ((0, 0), (0, cmax - r.shape[-1])),
                                constant_values=np.nan) for r in rows])
    masks_pad = np.stack([np.pad(m, (0, cmax - m.shape[0]),
                                 constant_values=np.nan) for m in masks])
    np.savez(os.path.join(out_dir, f"{tag}.npz"),
             rows=rows_pad.astype(np.float16), box_masks=masks_pad.astype(np.float16),
             grids=np.array(grids), ids=np.array(ids),
             meta=json.dumps(meta_json(grids, resized, ids)))
    return cache


def run_sink(wrapper, samples, out_dir, first_k=16):
    grabber = AttentionGrabber(wrapper.model)
    rows, ids = [], []
    for s in samples:
        inputs = wrapper.build_inputs(s)
        plen = wrapper.prompt_len(inputs)
        with grabber.capture(mode="rows", rows={"ans": plen - 1}, first_k=first_k):
            wrapper.forward(inputs, attentions=True)
        rows.append(grabber.results["ans"][..., 0, :].numpy())
        ids.append(s.id)
    np.savez(os.path.join(out_dir, "sink.npz"),
             rows16=np.stack(rows).astype(np.float16), ids=np.array(ids),
             first_k=first_k, meta=json.dumps(meta_json([], [], ids)))


def run_cond(wrapper, pairs, out_dir):
    grabber = AttentionGrabber(wrapper.model)
    ra, rb, grids, ids = [], {}, [], []
    cache = {}
    for s in pairs:
        inputs = wrapper.build_inputs(s)
        cache[s.id] = inputs
        img_pos = wrapper.image_token_positions(inputs["input_ids"])
        plen = wrapper.prompt_len(inputs)
        with grabber.capture(mode="rows", rows={"ans": plen - 1}, cols=img_pos):
            wrapper.forward(inputs, attentions=True)
        rows = grabber.results["ans"][..., 0, :].numpy()
        if s.variant == "attr":
            ra.append(rows)
        else:
            rb.append(rows)
        ids.append(s.id)
        grids.append(wrapper.grid_hw(inputs))
    masks = compute_masks(wrapper, [p for p in pairs], cache)
    cmax = max(r.shape[-1] for r in ra + rb)
    def pad_all(arrs):
        return np.stack([np.pad(r, ((0, 0), (0, cmax - r.shape[-1])),
                                constant_values=np.nan) for r in arrs])
    np.savez(os.path.join(out_dir, "cond.npz"),
             rows_attr=pad_all(ra).astype(np.float16),
             rows_plain=pad_all(rb).astype(np.float16),
             ids=np.array(ids), grids=np.array(grids),
             box_masks=np.stack([np.pad(m, (0, cmax - m.shape[0]),
                                        constant_values=np.nan) for m in masks]).astype(np.float16),
             meta=json.dumps(meta_json(grids, [], ids)))


def run_occlusion(wrapper, samples, out_dir):
    grabber = AttentionGrabber(wrapper.model)
    rbase, roccl, grids, masks, ids = [], [], [], [], []
    for s in samples:
        occ = occlude_sample(s)
        if occ is None:
            continue
        inputs_b = wrapper.build_inputs(s)
        img_pos = wrapper.image_token_positions(inputs_b["input_ids"])
        plen = wrapper.prompt_len(inputs_b)
        with grabber.capture(mode="rows", rows={"ans": plen - 1}, cols=img_pos):
            wrapper.forward(inputs_b, attentions=True)
        rbase.append(grabber.results["ans"][..., 0, :].numpy())
        inputs_o = wrapper.build_inputs(occ)
        with grabber.capture(mode="rows", rows={"ans": wrapper.prompt_len(inputs_o) - 1},
                             cols=wrapper.image_token_positions(inputs_o["input_ids"])):
            wrapper.forward(inputs_o, attentions=True)
        roccl.append(grabber.results["ans"][..., 0, :].numpy())
        grid = wrapper.grid_hw(inputs_b)
        grids.append(grid)
        rw, rh = wrapper.resized_hw(inputs_b)
        if s.image is not None:
            ow, oh = s.image.size
        else:
            from PIL import Image
            ow, oh = Image.open(s.image_path).size
        masks.append(box_to_mask(s.boxes[0], (ow, oh), (rw, rh), grid).flatten())
        ids.append(s.id)
    cmax = max(r.shape[-1] for r in rbase + roccl)
    def pad_all(arrs):
        return np.stack([np.pad(r, ((0, 0), (0, cmax - r.shape[-1])),
                                constant_values=np.nan) for r in arrs])
    np.savez(os.path.join(out_dir, "occlusion.npz"),
             rows_base=pad_all(rbase).astype(np.float16),
             rows_occl=pad_all(roccl).astype(np.float16),
             box_masks=np.stack([np.pad(m, (0, cmax - m.shape[0]),
                                        constant_values=np.nan) for m in masks]).astype(np.float16),
             grids=np.array(grids), ids=np.array(ids),
             meta=json.dumps(meta_json(grids, [], ids)))


def run_pope_capture(wrapper, samples, out_dir):
    from headatlas.eval.pope_runner import run_pope
    preds, rows = run_pope(wrapper, samples, capture_boxes=True)
    gts = [s.answer for s in samples]
    correct = [p == g for p, g in zip(preds, gts)]
    cache = {}
    grids, masks = [], []
    for s in samples:
        inputs = wrapper.build_inputs(s)
        cache[s.id] = inputs
        grid = wrapper.grid_hw(inputs)
        grids.append(grid)
        rw, rh = wrapper.resized_hw(inputs)
        if s.image is not None:
            ow, oh = s.image.size
        else:
            from PIL import Image
            ow, oh = Image.open(s.image_path).size
        m = box_to_mask(s.boxes[0], (ow, oh), (rw, rh), grid).flatten() if s.boxes \
            else np.zeros(grid[0] * grid[1], dtype=np.float32)
        masks.append(m)
    cmax = max(r.shape[-1] for r in rows)
    rows_pad = np.stack([np.pad(r, ((0, 0), (0, cmax - r.shape[-1])),
                                constant_values=np.nan) for r in rows])
    np.savez(os.path.join(out_dir, "pope.npz"),
             rows=rows_pad.astype(np.float16),
             box_masks=np.stack([np.pad(m, (0, cmax - m.shape[0]),
                                        constant_values=np.nan)
                                 for m in masks]).astype(np.float16),
             grids=np.array(grids), correct=np.array(correct), preds=np.array(preds),
             gts=np.array(gts), ids=np.array([s.id for s in samples]),
             meta=json.dumps(meta_json(grids, [], [s.id for s in samples],
                                       dict(preds=preds, gts=gts))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--probes", default="synth_grounding,ocr_synth,sink")
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-visual-tokens", type=int, default=576)
    args = ap.parse_args()
    out = args.out or os.path.join("outputs", "raw", slug(args.model))
    os.makedirs(out, exist_ok=True)
    wrapper = VLMWrapper(args.model, device=args.device,
                         max_visual_tokens=args.max_visual_tokens)
    L, H = n_layers_heads(wrapper.model)
    print(f"model={args.model} layers={L} heads={H} family={wrapper.family}")
    probes = args.probes.split(",")
    has_refcoco = os.path.exists(os.path.join(args.data, "refcoco.jsonl"))
    if "synth_grounding" in probes:
        samples = loaders.load_probe("synth_grounding", args.data, args.max_samples)
        run_box_probe(wrapper, samples, "synth_grounding", out)
    if "grounding" in probes:
        if not has_refcoco:
            print("grounding skipped (refcoco.jsonl missing)")
        else:
            samples = loaders.load_probe("grounding", args.data, args.max_samples)
            run_box_probe(wrapper, samples, "grounding", out)
    if "ocr_synth" in probes:
        samples = loaders.load_probe("ocr_synth", args.data, args.max_samples)
        run_box_probe(wrapper, samples, "ocr_synth", out)
    if "ocr" in probes:
        samples = loaders.load_probe("ocr_real", args.data, args.max_samples)
        run_box_probe(wrapper, samples, "ocr", out)
    if "spatial" in probes:
        samples = loaders.load_probe("spatial", args.data, args.max_samples)
        run_box_probe(wrapper, samples, "spatial", out)
    if "sink" in probes:
        samples = loaders.load_probe("sink", args.data, args.max_samples)
        run_sink(wrapper, samples, out)
    if "cond" in probes:
        base = loaders.load_probe("grounding", args.data, args.max_samples)
        pairs = build_conditioning_pairs(base)
        if pairs:
            run_cond(wrapper, pairs, out)
        else:
            print("no conditioning pairs available (RefCOCO jsonl missing?)")
    if "occlusion" in probes:
        if not has_refcoco:
            print("occlusion skipped (refcoco.jsonl missing)")
        else:
            samples = loaders.load_probe("grounding", args.data, args.max_samples)
            run_occlusion(wrapper, samples, out)
    if "pope" in probes:
        splits = loaders.pope_splits(args.data)
        if not splits:
            print("POPE data not found; run prepare_data.py --pope")
        else:
            samples = []
            for split, ss in splits.items():
                samples.extend(ss[:args.max_samples // len(splits)])
            run_pope_capture(wrapper, samples, out)
    print("DONE", out)


if __name__ == "__main__":
    main()
