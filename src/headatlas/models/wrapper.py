from __future__ import annotations

import torch
from PIL import Image


def detect_family(model_id: str) -> str:
    try:
        from transformers import AutoConfig
        mt = AutoConfig.from_pretrained(model_id).model_type
    except Exception:
        mt = None
    if mt in ("llava", "llava_next", "llava-onevision"):
        return "llava"
    if mt == "qwen2_5_vl":
        return "qwen2.5vl"
    if mt == "qwen2_vl":
        return "qwen2vl"
    if mt == "internvl" or "internvl" in (model_id or "").lower():
        return "internvl"
    mid = (model_id or "").lower()
    if "llava" in mid:
        return "llava"
    if "qwen" in mid and ("2.5" in mid or "2_5" in mid) and "vl" in mid:
        return "qwen2.5vl"
    if "qwen" in mid and "vl" in mid:
        return "qwen2vl"
    raise ValueError(f"unsupported model id: {model_id}")


class VLMWrapper:
    def __init__(self, model_id: str, device: str = "cuda", dtype=None,
                 max_visual_tokens: int = 576, min_visual_tokens: int = 256):
        self.model_id = model_id
        self.family = detect_family(model_id)
        self.device = device
        self.dtype = dtype or torch.bfloat16
        self.max_visual_tokens = max_visual_tokens
        self.min_visual_tokens = min_visual_tokens
        load_kw = dict(torch_dtype=self.dtype, attn_implementation="eager",
                       low_cpu_mem_usage=True)
        if self.family == "llava":
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = self._load(
                LlavaForConditionalGeneration, model_id, load_kw).to(device).eval()
        elif self.family in ("qwen2.5vl", "qwen2vl"):
            cls = {"qwen2.5vl": "Qwen2_5_VLForConditionalGeneration",
                   "qwen2vl": "Qwen2VLForConditionalGeneration"}[self.family]
            import transformers as tf
            from transformers import AutoProcessor
            model_cls = getattr(tf, cls)
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                min_pixels=self.min_visual_tokens * 28 * 28,
                max_pixels=self.max_visual_tokens * 28 * 28)
            self.model = self._load(
                model_cls, model_id, load_kw).to(device).eval()
        elif self.family == "internvl":
            import transformers as tf
            if not hasattr(tf, "InternVLForConditionalGeneration"):
                raise NotImplementedError(
                    "InternVL needs transformers>=4.52 native support; "
                    "use LLaVA/Qwen2.5-VL for the atlas or upgrade transformers")
            from transformers import InternVLForConditionalGeneration, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = InternVLForConditionalGeneration.from_pretrained(
                model_id, **load_kw).to(device).eval()
        self._lm_layers_cache = None

    def _load(self, cls, model_id, load_kw):
        try:
            return cls.from_pretrained(model_id, **load_kw)
        except TypeError:
            kw = dict(load_kw)
            if "torch_dtype" in kw:
                kw["dtype"] = kw.pop("torch_dtype")
            return cls.from_pretrained(model_id, **kw)

    def build_inputs(self, sample) -> dict:
        image = sample.load_image()
        if self.family == "llava":
            text = f"USER: <image>\n{sample.prompt} ASSISTANT:"
            inputs = self.processor(images=[image], text=[text], return_tensors="pt")
        else:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": sample.prompt}]}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        return {k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in inputs.items()}

    @property
    def image_token_id(self) -> int:
        if self.family == "llava":
            return self.model.config.image_token_index
        if self.family in ("qwen2.5vl", "qwen2vl"):
            return self.model.config.image_token_id
        tok = getattr(self.processor, "tokenizer", None) or self.processor
        return tok.convert_tokens_to_ids("<image>")

    def image_token_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids[0] == self.image_token_id
        return torch.nonzero(mask, as_tuple=False).squeeze(-1)

    def image_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids[0] == self.image_token_id

    def grid_hw(self, inputs: dict):
        if self.family == "llava":
            n = int(self.image_token_mask(inputs["input_ids"]).sum().item())
            side = int(round(n ** 0.5))
            assert side * side == n, f"non-square visual grid: {n}"
            return side, side
        if self.family in ("qwen2.5vl", "qwen2vl"):
            thw = inputs["image_grid_thw"][0]
            return int(thw[1].item()) // 2, int(thw[2].item()) // 2
        return 16, 16

    def resized_hw(self, inputs: dict):
        if self.family == "llava":
            return 336, 336
        if self.family in ("qwen2.5vl", "qwen2vl"):
            thw = inputs["image_grid_thw"][0]
            return int(thw[1].item()) * 14, int(thw[2].item()) * 14
        return 448, 448

    def prompt_len(self, inputs: dict) -> int:
        return int(inputs["input_ids"].shape[1])

    def forward(self, inputs: dict, attentions: bool = False):
        with torch.no_grad():
            return self.model(**inputs, output_attentions=attentions)

    def generate(self, inputs: dict, max_new_tokens: int = 32,
                 attentions: bool = False) -> str:
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False, output_attentions=attentions)
        text = self.processor.decode(out[0, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        return text.strip()


def find_lm_attention_modules(model):
    found = {}
    for name, module in model.named_modules():
        low = name.lower()
        if "vision" in low or "visual" in low or "vit" in low:
            continue
        if not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
            continue
        first = module[0]
        attn0 = getattr(first, "self_attn", None)
        if attn0 is None or not hasattr(attn0, "o_proj"):
            continue
        entries = []
        for i, layer in enumerate(module):
            sa = getattr(layer, "self_attn", None)
            if sa is None:
                continue
            cfg = getattr(sa, "config", None)
            n_heads = getattr(cfg, "num_attention_heads", None) if cfg else None
            hidden = getattr(cfg, "hidden_size", None) if cfg else None
            head_dim = getattr(cfg, "head_dim", None) if cfg else None
            if head_dim is None and n_heads:
                head_dim = sa.o_proj.in_features // n_heads
            if not n_heads:
                n_heads = sa.o_proj.in_features // (head_dim or 64)
            entries.append(dict(idx=i, attn=sa, o_proj=sa.o_proj,
                                n_heads=int(n_heads), head_dim=int(head_dim),
                                name=f"{name}[{i}]"))
        if entries:
            found[name] = entries
    if not found:
        raise RuntimeError("no language-model decoder attention modules found")
    return max(found.values(), key=len)


def lm_config(model):
    for cand in ("text_config",):
        cfg = getattr(model.config, cand, None)
        if cfg is not None and getattr(cfg, "num_attention_heads", None):
            return cfg
    layers = find_lm_attention_modules(model)
    return getattr(layers[0]["attn"], "config", None)


def n_layers_heads(model):
    layers = find_lm_attention_modules(model)
    return len(layers), layers[0]["n_heads"]
