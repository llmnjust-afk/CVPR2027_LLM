"""Build data/refcoco.jsonl from the HF mirror jxu124/refcoco (val split).

The UNC refs(unc).pkl host is unreachable from the lab; this parquet carries
the same refs (sentences + ann bbox + image file). GT box is taken from
raw_anns['bbox'] (COCO [x, y, w, h]) and converted to [x1, y1, x2, y2].
Images live under COCO train2014 (RefCOCO val uses train2014 files).
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from huggingface_hub import hf_hub_download, list_repo_files


def clean_name(fn):
    """jxu124/refcoco appends per-annotation suffixes (_0, _1, ...) to make
    file names unique; the real COCO file has no suffix."""
    m = re.match(r"(COCO_(?:train|val)2014_\d{12})(?:_\d+)?\.jpg$", fn)
    return f"{m.group(1)}.jpg" if m else fn


def get_xywh(r):
    raw = r.get("raw_anns")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = None
    if isinstance(raw, dict) and "bbox" in raw:
        v = [float(t) for t in raw["bbox"]]
        if len(v) == 4:
            return v
    b = r.get("bbox")
    if isinstance(b, str):
        b = json.loads(b.replace("(", "[").replace(")", "]")) \
            if b.startswith("[") else json.loads(b)
    v = [float(t) for t in (list(b) if not isinstance(b, str) else json.loads(b))]
    x, y, w, h = v[0], v[1], v[2], v[3]
    if w > x and h > y and w + x <= 641 and h + y <= 481:
        pass
    return x, y, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--max-samples", type=int, default=800)
    ap.add_argument("--split", default="validation")
    args = ap.parse_args()

    files = list_repo_files("jxu124/refcoco", repo_type="dataset")
    fname = next(f for f in files if args.split in f and f.endswith(".parquet"))
    path = hf_hub_download("jxu124/refcoco", fname, repo_type="dataset")

    import pandas as pd
    df = pd.read_parquet(path)
    df = df[df["split"] == "val"].reset_index(drop=True)

    out = []
    for _, r in df.iterrows():
        x, y, w, h = get_xywh(r.to_dict())
        if w > x and h > y:
            x1, y1, x2, y2 = x, y, w, h
        else:
            x1, y1, x2, y2 = x, y, x + w, y + h
        sents = r["sentences"]
        s0 = sents[0]["sent"] if isinstance(sents[0], dict) else str(sents[0])
        fn = clean_name(r["file_name"])
        out.append(dict(id=f"refcoco_{len(out)}",
                        image_path=os.path.join(args.data, "images",
                                                "train2014", fn),
                        prompt=s0, boxes=[[x1, y1, x2, y2]],
                        meta={"image_file": fn, "ann_id": str(r["ann_id"])}))
    out = out[:args.max_samples]
    path_out = os.path.join(args.data, "refcoco.jsonl")
    with open(path_out, "w") as f:
        for s in out:
            f.write(json.dumps(s) + "\n")
    print("wrote", path_out, len(out))


if __name__ == "__main__":
    main()
