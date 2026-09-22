import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from headatlas.interventions.head_ops import HeadEditor
from headatlas.models.attention_grabber import AttentionGrabber
from headatlas.models.biased_attention import (BiasMode, build_kv_bias,
                                               clear_attention_bias,
                                               set_attention_bias)
from headatlas.models.wrapper import find_lm_attention_modules, n_layers_heads

from transformers import LlamaConfig, LlamaForCausalLM


def build_tiny():
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=8,
                      num_key_value_heads=4, max_position_embeddings=512,
                      attn_implementation="eager")
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    return model


def reference_attentions(model, input_ids):
    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True)
    return [a[0].float() for a in out.attentions], out.logits


def test_rows_vs_reference():
    model = build_tiny()
    L, H = n_layers_heads(model)
    assert (L, H) == (2, 8), (L, H)
    ids = torch.randint(0, 100, (1, 24))
    ref, logits = reference_attentions(model, ids)
    cols = torch.tensor([0, 3, 7, 23])
    rows = {"ans": 23, "mid": 12}
    g = AttentionGrabber(model)
    with g.capture(mode="rows", rows=rows, cols=cols):
        model(input_ids=ids)
    for label, p in rows.items():
        got = g.results[label]
        assert got.shape == (L, H, 1, 4), got.shape
        for l in range(L):
            exp = ref[l][:, p, :][:, cols]
            assert torch.allclose(got[l, :, 0, :], exp, atol=1e-5), \
                f"{label} layer {l} mismatch"
    print("PASS rows-vs-reference")


def test_decode_step_vs_reference():
    model = build_tiny()
    ids = torch.randint(0, 100, (1, 20))
    g = AttentionGrabber(model)
    # transformers 5.x generate skips the final forward (last token's logits
    # are never needed), so decoding 2 input positions requires 3 new tokens.
    out = model.generate(input_ids=ids, max_new_tokens=3, do_sample=False,
                         eos_token_id=None)
    full = out
    ref, _ = reference_attentions(model, full)
    with g.capture(mode="rows", rows={"gen0": 20, "gen1": 21}, cols=torch.tensor([5, 19])):
        model.generate(input_ids=ids, max_new_tokens=3, do_sample=False,
                       eos_token_id=None)
    for l in range(g.n_layers):
        exp0 = ref[l][:, 20, :][:, [5, 19]]
        assert torch.allclose(g.results["gen0"][l, :, 0, :], exp0, atol=1e-5)
        exp1 = ref[l][:, 21, :][:, [5, 19]]
        assert torch.allclose(g.results["gen1"][l, :, 0, :], exp1, atol=1e-5)
    print("PASS decode-step-vs-reference")


def test_query_block():
    model = build_tiny()
    ids = torch.randint(0, 100, (1, 24))
    ref, _ = reference_attentions(model, ids)
    mask = torch.zeros(24, dtype=torch.bool)
    mask[[4, 10, 17]] = True
    g = AttentionGrabber(model)
    with g.capture(mode="query_block", row_mask=mask, first_k=6):
        model(input_ids=ids)
    got = g.results["block"]
    assert got.shape == (g.n_layers, g.n_heads, 1, 6), got.shape
    for l in range(g.n_layers):
        exp = ref[l][:, mask, :6].sum(dim=1)
        assert torch.allclose(got[l, :, 0, :], exp, atol=1e-5)
    print("PASS query-block")


def test_bias():
    model = build_tiny()
    ids = torch.randint(0, 100, (1, 24))
    with torch.no_grad():
        base = model(input_ids=ids).logits
    bias = build_kv_bias(model, torch.tensor([2, 5, 9]), 3.0, max_len=24)
    with BiasMode(model):
        set_attention_bias(model, bias)
        with torch.no_grad():
            out = model(input_ids=ids).logits
        zero = torch.zeros_like(bias)
        set_attention_bias(model, zero)
        with torch.no_grad():
            out_zero = model(input_ids=ids).logits
    clear_attention_bias(model)
    assert not torch.allclose(base, out), "nonzero bias had no effect"
    assert torch.allclose(base, out_zero, atol=1e-4), \
        "zero bias changed outputs vs default eager"
    print("PASS bias (zero==default, nonzero differs)")


def test_editor_ablate_capture_patch():
    model = build_tiny()
    editor = HeadEditor(model)
    ids = torch.randint(0, 100, (1, 24))
    with torch.no_grad():
        base = model(input_ids=ids).logits
    plan = {0: {1: None}, 1: {3: None}}
    with editor.ablate(plan, mode="zero"):
        with torch.no_grad():
            out = model(input_ids=ids).logits
    assert not torch.allclose(base, out)
    with torch.no_grad():
        again = model(input_ids=ids).logits
    assert torch.allclose(base, again, atol=1e-6), "state leaked after ablation"
    pos = [23]
    with editor.capture(pos) as ed:
        with torch.no_grad():
            model(input_ids=ids)
    store = ed.store
    assert all(23 in store[li] for li in range(editor.n_layers))
    vec = store[0][23].reshape(editor.n_heads, editor.head_dim)[1]
    patch_plan = {0: {"__heads__": {23: {1: vec * 2.0}}}}
    with editor.patch(patch_plan):
        with torch.no_grad():
            out_p = model(input_ids=ids).logits
    assert not torch.allclose(base, out_p)
    print("PASS editor ablate/capture/patch")


def test_mean_ablate():
    model = build_tiny()
    editor = HeadEditor(model)
    ids = torch.randint(0, 100, (1, 24))
    means = {li: {h: torch.full((editor.head_dim,), 0.01)
                  for h in [0, 2]} for li in range(editor.n_layers)}
    plan = {li: {h: "mean" for h in [0, 2]} for li in range(editor.n_layers)}
    with torch.no_grad():
        base = model(input_ids=ids).logits
    with editor.ablate(plan, mode="mean", means=means):
        with torch.no_grad():
            out = model(input_ids=ids).logits
    assert not torch.allclose(base, out)
    print("PASS mean-ablate")


if __name__ == "__main__":
    test_rows_vs_reference()
    test_decode_step_vs_reference()
    test_query_block()
    test_bias()
    test_editor_ablate_capture_patch()
    test_mean_ablate()
    print("ALL TESTS PASSED")
