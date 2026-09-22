from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw

from .base import ProbeSample

WORDS = ["STOP", "EXIT", "SALE", "OPEN", "PARK", "TAXI", "MENU", "CAFE", "BOOK",
         "HELP", "ZONE", "GATE", "SHOP", "WALK", "NOTE", "INFO", "DESK", "ROOM"]


def _noise_image(size, seed):
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr)


def make_noise_sample(i, size=(448, 448), prompt="Describe this image."):
    img = _noise_image(size, seed=i)
    return ProbeSample(id=f"noise_{i}", prompt=prompt, image=img,
                       meta={"kind": "noise"})


def build_sink_probe(n=400, size=(448, 448)):
    return [make_noise_sample(i, size=size) for i in range(n)]


def make_ocr_sample(i, size=(448, 448), n_words=2, seed=None):
    rng = random.Random(seed if seed is not None else i)
    img = _noise_image(size, seed=10000 + i)
    draw = ImageDraw.Draw(img)
    words = [rng.choice(WORDS) for _ in range(n_words)]
    text = " ".join(words)
    w = rng.randint(size[0] // 5, size[0] // 3)
    h = max(24, size[1] // 14)
    x1 = rng.randint(0, size[0] - w - 1)
    y1 = rng.randint(0, size[1] - h - 1)
    draw.text((x1, y1), text, fill=(255, 255, 255))
    box = [x1 - 4, y1 - 4, x1 + w + 4, y1 + h + 4]
    box = [max(0, box[0]), max(0, box[1]), min(size[0], box[2]), min(size[1], box[3])]
    prompt = "Read the text shown in the image. Reply with the exact text."
    return ProbeSample(id=f"ocr_{i}", prompt=prompt, image=img, boxes=[box],
                       answer=text, meta={"kind": "ocr_synth", "text": text})


def build_ocr_probe(n=400, size=(448, 448)):
    return [make_ocr_sample(i, size=size) for i in range(n)]


def make_bright_square_sample(i, size=(448, 448), seed=None, square_frac=0.18):
    """Synthetic grounding probe: a bright gray square (the 'object') placed on
    a dim noise background; GT box known exactly."""
    rng = np.random.RandomState(20000 + i)
    arr = (rng.randint(0, 40, size=(size[1], size[0], 3))).astype(np.uint8)
    side = int(size[0] * square_frac)
    x1 = rng.randint(0, size[0] - side - 1)
    y1 = rng.randint(0, size[1] - side - 1)
    arr[y1:y1 + side, x1:x1 + side, :] = 255
    img = Image.fromarray(arr)
    box = [x1, y1, x1 + side, y1 + side]
    prompt = "Where is the bright object in this image?"
    return ProbeSample(id=f"sq_{i}", prompt=prompt, image=img, boxes=[box],
                       meta={"kind": "bright_square"})


def build_synthetic_grounding(n=400, size=(448, 448)):
    return [make_bright_square_sample(i, size=size) for i in range(n)]


def occlude_sample(sample, seed=0, fill=0):
    """Return a copy of the sample whose image has the GT box region blanked."""
    img = sample.load_image().copy()
    if not sample.boxes:
        return None
    x1, y1, x2, y2 = [int(v) for v in sample.boxes[0]]
    arr = np.asarray(img)
    arr[y1:y2, x1:x2, :] = fill
    out = ProbeSample(id=sample.id + "_occl", prompt=sample.prompt,
                      image=Image.fromarray(arr), boxes=sample.boxes,
                      answer=sample.answer, pair_group=sample.id,
                      variant="occluded", meta={**sample.meta, "kind": "occluded"})
    return out


def build_conditioning_pairs(samples, attribute_words=None, seed=0):
    """For grounding samples whose phrase contains an attribute word, emit two
    variants: full phrase (attr) and attribute-stripped phrase (plain)."""
    attribute_words = attribute_words or [
        "red", "blue", "green", "yellow", "black", "white", "left", "right",
        "small", "large", "brown", "orange", "purple", "gray", "pink"]
    rng = random.Random(seed)
    pairs = []
    for s in samples:
        words = s.prompt.split()
        if any(w in attribute_words for w in words):
            plain = " ".join(w for w in words if w not in attribute_words)
        else:
            plain = None
        if plain and plain != s.prompt:
            a = ProbeSample(id=s.id + "_attr", prompt=s.prompt,
                            image_path=s.image_path, image=s.image,
                            boxes=s.boxes, pair_group=s.id, variant="attr")
            b = ProbeSample(id=s.id + "_plain", prompt=plain,
                            image_path=s.image_path, image=s.image,
                            boxes=s.boxes, pair_group=s.id, variant="plain")
            pairs.extend([a, b])
    return pairs
