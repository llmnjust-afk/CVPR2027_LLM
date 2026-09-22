from __future__ import annotations

from contextlib import contextmanager

import torch

from ..models.wrapper import find_lm_attention_modules


class HeadEditor:
    """Installs wrappers on each decoder layer's o_proj so that per-head slices
    of the attention output (the o_proj input) can be zeroed / mean-replaced /
    replaced with values captured from another run (activation patching).

    Slice layout: o_proj input is [q, H*head_dim] (or [1, q, H*head_dim]);
    head h occupies columns [h*head_dim, (h+1)*head_dim).

    Position tracking: prefill covers absolute positions 0..q-1; every cached
    decode call has q==1 and advances an internal counter starting at
    prefill_len (= last prefill pos + 1).
    """

    def __init__(self, model):
        self.layers = find_lm_attention_modules(model)
        self.n_layers = len(self.layers)
        self.n_heads = self.layers[0]["n_heads"]
        self.head_dim = self.layers[0]["head_dim"]
        self._state = None
        self._last_store = None
        for li, m in enumerate(self.layers):
            op = m["o_proj"]
            orig = op.forward
            op.forward = self._make_wrapped(orig, li)

    def _make_wrapped(self, orig, li):
        def wrapped(x, *a, **k):
            st = self._state
            if st is not None:
                x = self._process(x, li, st)
            return orig(x, *a, **k)
        return wrapped

    def _as2d(self, x):
        if x.dim() == 3 and x.shape[0] == 1:
            return x[0], True
        return x, False

    def _restore(self, x, had_batch):
        return x[None] if had_batch else x

    def _process(self, x, li, st):
        kind = st["kind"]
        if kind == "capture":
            x2, had_batch = self._as2d(x)
            q = x2.shape[0]
            if q > 1:
                st["last_pos"] = q - 1
                for p in st["positions"]:
                    if p < q:
                        st["store"][li][int(p)] = x2[p].detach().float().cpu().clone()
            else:
                nxt = st["next_pos"]
                st["next_pos"] = nxt + 1
                if nxt in st["positions"]:
                    st["store"][li][int(nxt)] = x2[0].detach().float().cpu().clone()
            return x
        plan = st["plan"].get(li)
        if not plan:
            return x
        x2, had_batch = self._as2d(x)
        if kind == "ablate":
            x2 = x2.clone()
            for h, mode in plan.items():
                sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                if mode == "zero":
                    x2[:, sl] = 0.0
                else:
                    x2[:, sl] = st["means"][li][h].to(x2.dtype)
            return self._restore(x2, had_batch)
        if kind == "patch":
            heads_by_pos = plan.get("__heads__", {})
            q = x2.shape[0]
            if q > 1:
                st["last_pos"] = q - 1
                if heads_by_pos:
                    x2 = x2.clone()
                    for p, heads in heads_by_pos.items():
                        if p < q:
                            for h, vec in heads.items():
                                sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                                x2[p, sl] = vec.to(x2.dtype)
            else:
                nxt = st["next_pos"]
                st["next_pos"] = nxt + 1
                heads = heads_by_pos.get(nxt)
                if heads:
                    x2 = x2.clone()
                    for h, vec in heads.items():
                        sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                        x2[0, sl] = vec.to(x2.dtype)
            return self._restore(x2, had_batch)
        return x

    @contextmanager
    def ablate(self, plan, mode="zero", means=None):
        """plan: {layer_idx: {head: mode_override_or_None}}; mode 'zero'|'mean'."""
        clean = {int(li): {int(h): (mode if mv is None else mv) for h, mv in heads.items()}
                 for li, heads in plan.items()}
        state = dict(kind="ablate", plan=clean, means=means,
                     last_pos=None, next_pos=0)
        prev = self._state
        self._state = state
        try:
            yield self
        finally:
            self._state = prev

    @contextmanager
    def patch(self, plan):
        """plan: {layer_idx: {"__heads__": {abs_pos: {head: vec}}}}"""
        state = dict(kind="patch", plan=plan, last_pos=None, next_pos=0)
        prev = self._state
        self._state = state
        try:
            yield self
        finally:
            self._state = prev

    @contextmanager
    def capture(self, abs_positions):
        """Captures full o_proj-input vectors at the given absolute positions
        for every layer. After exit, `.store` -> {layer_idx: {pos: [hidden]}}."""
        positions = set(int(p) for p in abs_positions)
        state = dict(kind="capture", positions=positions,
                     store={li: {} for li in range(self.n_layers)},
                     last_pos=None, next_pos=0)
        prev = self._state
        self._state = state
        try:
            yield self
        finally:
            self._last_store = state["store"]
            self._state = prev

    @property
    def store(self):
        return self._last_store

    def compute_means(self, model, sample_iter, n_calib=16, max_new_tokens=1):
        """Mean per-head o_proj input over calibration samples (for mean-ablation)."""
        import collections
        import numpy as np
        sums = {li: torch.zeros(self.n_heads, self.head_dim) for li in range(self.n_layers)}
        cnt = 0
        for sample in sample_iter:
            if cnt >= n_calib:
                break
            inputs = model.build_inputs(sample)
            positions = [model.prompt_len(inputs) - 1]
            with self.capture(positions) as ed:
                model.forward(inputs)
            store = ed.store
            for li in range(self.n_layers):
                vec = store[li].get(positions[0])
                if vec is None:
                    continue
                v = vec.reshape(self.n_heads, self.head_dim)
                sums[li] += v
            cnt += 1
        if cnt == 0:
            raise RuntimeError("no calibration samples for mean ablation")
        return {li: (sums[li] / cnt) for li in range(self.n_layers)}
