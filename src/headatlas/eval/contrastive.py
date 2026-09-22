from __future__ import annotations

import torch
import torch.nn.functional as F


def vcd_generate(wrapper, sample, alpha=1.0, beta=0.1, noise_std=0.3,
                 max_new_tokens=16):
    """Contrastive decoding baseline (VCD, CVPR 2024): greedy decode with
    logits = (1+alpha) * log p(clean) - alpha * log p(visually-degraded),
    restricted to plausible tokens (p_clean > beta * max)."""
    import numpy as np
    clean_img = sample.load_image()
    rng = np.random.RandomState(0)
    arr = np.asarray(clean_img).astype(np.float32)
    noisy = np.clip(arr + rng.randn(*arr.shape) * (255.0 * noise_std), 0, 255
                    ).astype(np.uint8)
    from PIL import Image
    from ..probes.base import ProbeSample
    degraded = ProbeSample(id=sample.id + "_vcd", prompt=sample.prompt,
                           image=Image.fromarray(noisy), meta={"kind": "vcd"})
    c_inputs = wrapper.build_inputs(sample)
    d_inputs = wrapper.build_inputs(degraded)
    device = wrapper.device
    eos_ids = wrapper.model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = wrapper.processor.tokenizer.eos_token_id
    if not isinstance(eos_ids, (list, tuple)):
        eos_ids = [eos_ids]
    seq_c = c_inputs["input_ids"]
    seq_d = d_inputs["input_ids"]
    assert seq_c.shape == seq_d.shape, "clean/degraded token layouts differ"
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out_c = wrapper.model(**c_inputs)
            out_d = wrapper.model(**d_inputs)
            lc = F.log_softmax(out_c.logits[:, -1, :], dim=-1)
            ld = F.log_softmax(out_d.logits[:, -1, :], dim=-1)
            scores = (1 + alpha) * lc - alpha * ld
            plaus = lc >= beta * lc.max(dim=-1, keepdim=True).values
            scores = scores.masked_fill(~plaus, float("-inf"))
            nxt = scores.argmax(dim=-1, keepdim=True)
            if int(nxt.item()) in eos_ids:
                break
            c_inputs = dict(c_inputs)
            d_inputs = dict(d_inputs)
            c_inputs["input_ids"] = torch.cat([seq_c, nxt.to(seq_c.device)], dim=-1)
            d_inputs["input_ids"] = torch.cat([seq_d, nxt.to(seq_d.device)], dim=-1)
            seq_c = c_inputs["input_ids"]
            seq_d = d_inputs["input_ids"]
            for d0 in (c_inputs, d_inputs):
                t_len = d0["input_ids"].shape[1]
                for k, fill in (("attention_mask", 1),
                                ("mm_token_type_ids", 0)):
                    v = d0.get(k)
                    if v is None:
                        continue
                    if v.shape[1] < t_len:
                        pad = torch.full((v.shape[0], t_len - v.shape[1]),
                                         fill, dtype=v.dtype,
                                         device=v.device)
                        d0[k] = torch.cat([v, pad], dim=-1)
                    elif v.shape[1] > t_len:
                        d0[k] = v[:, :t_len]
    return wrapper.processor.decode(seq_c[0, wrapper.prompt_len(c_inputs):],
                                    skip_special_tokens=True).strip()
