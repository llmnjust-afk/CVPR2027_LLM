from __future__ import annotations

import json
import os

import numpy as np
import torch

from ..models.attention_grabber import AttentionGrabber
from ..models.biased_attention import (BiasMode, build_kv_bias,
                                       clear_attention_bias, set_attention_bias)


def parse_yes_no(text: str):
    t = text.strip().lower()
    for w in t.replace(".", " ").replace(",", " ").split()[:3]:
        if w in ("yes", "no"):
            return w
    if "yes" in t:
        return "yes"
    if "no" in t:
        return "no"
    return "unknown"


def pope_metrics(preds, gts):
    tp = sum(1 for p, g in zip(preds, gts) if p == "yes" and g == "yes")
    fp = sum(1 for p, g in zip(preds, gts) if p == "yes" and g != "yes")
    fn = sum(1 for p, g in zip(preds, gts) if p != "yes" and g == "yes")
    tn = len(gts) - tp - fp - fn
    acc = (tp + tn) / max(1, len(gts))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return dict(acc=acc, precision=precision, recall=recall, f1=f1,
                yes_rate=sum(1 for p in preds if p == "yes") / max(1, len(preds)))


def run_pope(wrapper, samples, max_new_tokens=16, capture_boxes=True,
             lam_per_head=None, max_len_pad=64):
    """Runs POPE-style yes/no evaluation. If lam_per_head is given, applies
    head-selective visual KV bias during generation. Returns (preds, rows)."""
    grabber = AttentionGrabber(wrapper.model) if capture_boxes else None
    preds = []
    rows_all = []
    for s in samples:
        inputs = wrapper.build_inputs(s)
        img_pos = wrapper.image_token_positions(inputs["input_ids"])
        plen = wrapper.prompt_len(inputs)
        bias_ctx = None
        if lam_per_head is not None:
            bias = build_kv_bias(wrapper.model, img_pos, lam_per_head,
                                 max_len=plen + max_len_pad,
                                 device=wrapper.device)
            set_attention_bias(wrapper.model, bias)
            bias_ctx = BiasMode(wrapper.model)
            bias_ctx.__enter__()
        try:
            if grabber is not None:
                with grabber.capture(mode="rows", rows={"ans": plen - 1},
                                     cols=img_pos):
                    text = wrapper.generate(inputs, max_new_tokens=max_new_tokens,
                                            attentions=True)
                rows_all.append(grabber.results["ans"][..., 0, :].numpy())
            else:
                text = wrapper.generate(inputs, max_new_tokens=max_new_tokens)
            preds.append(parse_yes_no(text))
        finally:
            if bias_ctx is not None:
                bias_ctx.__exit__()
                clear_attention_bias(wrapper.model)
    return preds, rows_all
