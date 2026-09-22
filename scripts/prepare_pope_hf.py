"""POPE fallback source: lmms-lab-encoder/POPE parquets (raw.githubusercontent
is intermittently unreachable from the lab). Uses question/answer/image_source
only; images are lazy-downloaded from images.cocodataset.org as usual.
GT boxes attached from instances_val2014.json when available.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def attach_boxes(samples, data_dir, instances_json):
    if not instances_json or not os.path.exists(instances_json):
        return samples
    with open(instances_json) as f:
        d = json.load(f)
    cats = {c["id"]: c["name"] for c in d["categories"]}
    inst = {}
    for a in d["annotations"]:
        inst.setdefault(a["image_id"], []).append((cats[a["category_id"]],
                                                   a["bbox"]))
    for s in samples:
        fname = os.path.basename(s["image_path"])
        iid = int(re.sub(r"\D", "", fname.split(".")[0]) or 0)
        q = s["prompt"]
        m = re.search(r"Is there (?:a|an) (.+?) in the image", q, re.I)
        obj = m.group(1) if m else None
        if obj:
            for name, bbox in inst.get(iid, []):
                if name == obj:
                    x, y, w, h = bbox
                    s["boxes"] = [[x, y, x + w, y + h]]
                    break
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--instances", default=None)
    ap.add_argument("--splits", default="random,popular,adversarial")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    import pandas as pd
    for split in args.splits.split(","):
        out_path = os.path.join(args.data, f"pope_{split}.jsonl")
        if os.path.exists(out_path):
            print("skip existing", out_path)
            continue
        p = hf_hub_download("lmms-lab-encoder/POPE",
                            f"Full/{split}-00000-of-00001.parquet",
                            repo_type="dataset")
        df = pd.read_parquet(p)
        samples = []
        for k, r in df.iterrows():
            src = str(r["image_source"])
            fname = src if src.endswith(".jpg") else f"{src}.jpg"
            samples.append(dict(id=f"{split}_{k}",
                                image_path=os.path.join(args.data, "images",
                                                        "val2014", fname),
                                prompt=str(r["question"]),
                                answer=str(r["answer"]), boxes=[],
                                meta={"image_file": fname}))
        samples = attach_boxes(samples, args.data, args.instances)
        with open(out_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print("wrote", out_path, len(samples))


if __name__ == "__main__":
    main()
