from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager

import torch

from .wrapper import find_lm_attention_modules


class AttentionGrabber:
    """Captures per-head attention slices via forward hooks on decoder self_attn
    modules, replacing the weights tensor with None so the top-level output never
    accumulates the full [L, H, q, k] stack.

    modes:
      rows: rows = {label: abs_position}, cols = absolute KV positions (or first_k)
            -> results[label] : [L, H, n_chunks, C]
      query_block: row_mask = bool tensor over prefill query positions,
            first_k = int  -> results["block"] : [L, H, n_chunks, first_k]
    """

    def __init__(self, model):
        self.layers = find_lm_attention_modules(model)
        self.n_layers = len(self.layers)
        self.n_heads = self.layers[0]["n_heads"]
        self.results = {}

    @contextmanager
    def capture(self, mode="rows", rows=None, cols=None, first_k=None,
                row_mask=None):
        if mode not in ("rows", "query_block"):
            raise ValueError(mode)
        state = dict(mode=mode, rows=rows or {}, cols=cols, first_k=first_k,
                     row_mask=row_mask,
                     buf={li: defaultdict(list) for li in range(self.n_layers)})
        handles = [m["attn"].register_forward_hook(self._hook_for(li, state))
                   for li, m in enumerate(self.layers)]
        try:
            yield self
        finally:
            for h in handles:
                h.remove()
            self._finalize(state)
        labels = set()
        for li in range(self.n_layers):
            labels |= set(state["buf"][li].keys())
        self.results = {}
        for label in sorted(labels):
            self.results[label] = self._stack(state, label)

    def _hook_for(self, li, state):
        n_heads = self.n_heads

        def hook(module, args, output):
            if output is None:
                return None
            w = None
            widx = None
            for i, el in enumerate(output):
                if torch.is_tensor(el) and el.dim() == 4 and el.shape[1] == n_heads:
                    w = el
                    widx = i
                    break
            if w is None:
                return None
            new_output = list(output)
            new_output[widx] = None
            out = tuple(new_output) if isinstance(output, tuple) else new_output
            w = w.detach()
            if state["mode"] == "rows":
                cols = state["cols"]
                fk = state["first_k"]
                if w.shape[2] > 1:
                    for label, pos in state["rows"].items():
                        p = int(pos)
                        if p < w.shape[2]:
                            row = w[0, :, p, :]
                            row = row[:, cols] if cols is not None else row[:fk]
                            state["buf"][li][label].append(row.float().cpu())
                else:
                    p = int(w.shape[3]) - 1
                    for label, pos in state["rows"].items():
                        if isinstance(pos, int) and pos == p:
                            row = w[0, :, 0, :]
                            row = row[:, cols] if cols is not None else row[:fk]
                            state["buf"][li][label].append(row.float().cpu())
            else:
                if w.shape[2] > 1 and state["row_mask"] is not None:
                    mask = state["row_mask"].to(w.device)
                    agg = w[0][:, mask, :state["first_k"]].sum(dim=1)
                    state["buf"][li]["block"].append(agg.float().cpu())
            return out

        return hook

    def _finalize(self, state):
        pass

    def _stack(self, state, label):
        per_layer = []
        ref = None
        for li in range(self.n_layers):
            chunks = state["buf"][li].get(label, [])
            if chunks:
                per_layer.append(torch.stack(chunks, dim=1))
                if ref is None:
                    ref = per_layer[-1]
            else:
                per_layer.append(None)
        if ref is None:
            return None
        filled = []
        for t in per_layer:
            if t is None:
                filled.append(torch.full(ref.shape, float("nan"),
                                         dtype=torch.float32))
            else:
                filled.append(t)
        return torch.stack(filled, dim=0)
