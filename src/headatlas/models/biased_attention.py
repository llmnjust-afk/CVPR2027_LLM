from __future__ import annotations

import importlib

import torch

KEY = "headatlas_eager"


def _attention_functions():
    for path in ("transformers.modeling_utils", "transformers.masking_utils",
                 "transformers.integrations.sdpa_attention", "transformers"):
        try:
            mod = importlib.import_module(path)
            if hasattr(mod, "ALL_ATTENTION_FUNCTIONS"):
                return getattr(mod, "ALL_ATTENTION_FUNCTIONS")
        except Exception:
            continue
    raise RuntimeError("ALL_ATTENTION_FUNCTIONS not found; check transformers version")


def register_headatlas_mask_interface():
    """Alias the stock `eager` mask factory for our custom impl name so that
    masking behaves exactly like eager (materialized 4D float mask)."""
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    except Exception:
        return
    if KEY in ALL_MASK_ATTENTION_FUNCTIONS:
        return
    if "eager" in ALL_MASK_ATTENTION_FUNCTIONS:
        ALL_MASK_ATTENTION_FUNCTIONS[KEY] = ALL_MASK_ATTENTION_FUNCTIONS["eager"]


def headatlas_eager_attention(module, query, key, value, attention_mask,
                              scaling, dropout=0.0, **kwargs):
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key[:, :, None].expand(key.shape[0], key.shape[1], n_rep,
                                     key.shape[2], key.shape[3]).reshape(
            key.shape[0], key.shape[1] * n_rep, key.shape[2], key.shape[3])
        value = value[:, :, None].expand(value.shape[0], value.shape[1], n_rep,
                                         value.shape[2], value.shape[3]).reshape(
            value.shape[0], value.shape[1] * n_rep, value.shape[2], value.shape[3])
    w = torch.matmul(query, key.transpose(2, 3)) * scaling
    bias = getattr(module, "_headatlas_kv_bias", None)
    if bias is not None:
        klen = key.shape[-2]
        w = w + bias[None, :, None, :klen].to(w.dtype)
    if attention_mask is not None:
        am = attention_mask
        if am.dtype == torch.bool:
            am = torch.where(am, torch.zeros_like(am, dtype=w.dtype),
                             torch.full_like(am, torch.finfo(w.dtype).min,
                                              dtype=w.dtype))
        if am.dim() == 2:
            am = am[None, None]
        elif am.dim() == 3:
            am = am[:, None]
        w = w + am.to(w.dtype)
    else:
        q_len, k_len = query.shape[-2], key.shape[-2]
        causal = torch.triu(torch.full((q_len, k_len), torch.finfo(w.dtype).min,
                                       device=w.device, dtype=w.dtype),
                            diagonal=1 + k_len - q_len)
        w = w + causal
    w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0 and module.training:
        w = torch.nn.functional.dropout(w, p=dropout)
    out = torch.matmul(w, value)
    out = out.transpose(1, 2).contiguous().reshape(
        query.shape[0], query.shape[2], -1)
    return out, w


def register_headatlas_attention():
    _attention_functions()[KEY] = headatlas_eager_attention
    register_headatlas_mask_interface()


def set_attention_bias(model, bias):
    """bias: [n_layers, n_heads, max_len] additive logit bias on KV columns."""
    from .wrapper import find_lm_attention_modules
    layers = find_lm_attention_modules(model)
    assert bias.shape[0] == len(layers) and bias.shape[1] == layers[0]["n_heads"]
    for li, m in enumerate(layers):
        m["attn"]._headatlas_kv_bias = bias[li]


def clear_attention_bias(model):
    from .wrapper import find_lm_attention_modules
    for m in find_lm_attention_modules(model):
        if hasattr(m["attn"], "_headatlas_kv_bias"):
            delattr(m["attn"], "_headatlas_kv_bias")


class BiasMode:
    """Context manager: route LM decoder attention through headatlas_eager and
    apply the per-head KV-column bias currently set via set_attention_bias."""

    def __init__(self, model):
        self.model = model
        self._backup = {}

    def __enter__(self):
        register_headatlas_attention()
        from .wrapper import find_lm_attention_modules
        for m in find_lm_attention_modules(self.model):
            cfg = getattr(m["attn"], "config", None)
            if cfg is None:
                continue
            if id(cfg) not in self._backup:
                self._backup[id(cfg)] = (cfg, getattr(cfg, "_attn_implementation", None))
            cfg._attn_implementation = KEY
        return self

    def __exit__(self, *exc):
        for cfg, old in self._backup.values():
            if old is not None:
                cfg._attn_implementation = old
        self._backup = {}
        return False


def build_kv_bias(model, image_positions, lam_per_head, max_len, device=None,
                  dtype=torch.float32):
    """Returns [n_layers, n_heads, max_len] dense zeros with lam added on the
    given visual KV columns. lam_per_head: [n_heads] (broadcast across layers)
    or [n_layers, n_heads] full head-selective matrix (PAI-uniform is the
    special case lam == const)."""
    from .wrapper import n_layers_heads
    if device is None:
        device = next(model.parameters()).device
    n_layers, n_heads = n_layers_heads(model)
    lam = torch.as_tensor(lam_per_head, dtype=dtype)
    if lam.dim() == 0:
        lam = lam.reshape(1).repeat(n_heads)
    bias = torch.zeros(n_layers, n_heads, max_len, dtype=dtype)
    pos = torch.as_tensor(image_positions, dtype=torch.long).reshape(-1)
    if lam.dim() == 1:
        bias[:, :, pos] = lam.view(1, -1, 1)
    else:
        bias[:, :, pos] = lam.unsqueeze(-1).to(bias.dtype)
    return bias.to(device)
