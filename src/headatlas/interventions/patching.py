from __future__ import annotations

import csv
import os

import torch

from ..eval.pope_runner import parse_yes_no
from .head_ops import HeadEditor


def find_corrupted_sample(sample, pool):
    """Another sample over the same image with a different referred object."""
    for other in pool:
        if other.id != sample.id and other.image_path == sample.image_path:
            if other.boxes and sample.boxes:
                return other
    return None


def run_patching(wrapper, samples, heads, out_csv, max_new_tokens=12):
    """clean run (sample) vs corrupted run (different object, same image);
    patch target heads' o_proj input from corrupted into clean at the answer
    positions; record answer flips."""
    editor = HeadEditor(wrapper.model)
    results = []
    n_done = 0
    for s in samples:
        corr = find_corrupted_sample(s, samples)
        if corr is None or not heads:
            continue
        clean_inputs = wrapper.build_inputs(s)
        plen = wrapper.prompt_len(clean_inputs)
        positions = [plen - 1, plen]
        corr_inputs = wrapper.build_inputs(corr)
        with editor.capture(positions) as ed:
            wrapper.generate(corr_inputs, max_new_tokens=1, attentions=False)
        store = ed.store
        if not store[0]:
            continue
        ans_clean = parse_yes_no(wrapper.generate(clean_inputs,
                                                  max_new_tokens=max_new_tokens))
        plan = {}
        for (l, h) in heads:
            heads_at = {}
            for p in positions:
                vec = store[l].get(p)
                if vec is None:
                    continue
                v = vec.reshape(editor.n_heads, editor.head_dim)[h]
                heads_at.setdefault(p, {})[h] = v
            if heads_at:
                plan[l] = {"__heads__": heads_at}
        if not plan:
            continue
        with editor.patch(plan):
            ans_patch = parse_yes_no(wrapper.generate(clean_inputs,
                                                      max_new_tokens=max_new_tokens))
        results.append(dict(id=s.id, clean=ans_clean, patched=ans_patch,
                            flip=int(ans_patch != ans_clean)))
        n_done += 1
    flip_rate = (sum(r["flip"] for r in results) / max(1, len(results)))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "clean", "patched", "flip"])
        for r in results:
            w.writerow([r["id"], r["clean"], r["patched"], r["flip"]])
    return dict(n=n_done, flip_rate=flip_rate, rows=results)
