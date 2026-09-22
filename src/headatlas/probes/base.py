from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Optional

from PIL import Image


@dataclass
class ProbeSample:
    id: str
    prompt: str
    image_path: Optional[str] = None
    image: Optional[Image.Image] = None
    boxes: list = field(default_factory=list)
    answer: Optional[str] = None
    pair_group: Optional[str] = None
    variant: str = "base"
    meta: dict = field(default_factory=dict)

    def load_image(self) -> Image.Image:
        if self.image is not None:
            return self.image
        return Image.open(self.image_path).convert("RGB")

    def to_json(self):
        d = asdict(self)
        d.pop("image", None)
        return d

    @classmethod
    def from_json(cls, d: dict):
        d = dict(d)
        return cls(**d)


def load_jsonl(path) -> list:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(ProbeSample.from_json(json.loads(line)))
    return samples


def save_jsonl(samples, path):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_json()) + "\n")


def subsample(samples, n, seed=0):
    if n is None or n >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(samples)), n))
    return [samples[i] for i in idx]
