import argparse
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from headatlas.probes.synthetic import build_ocr_probe, build_sink_probe

POPE_URLS = [
    "https://raw.githubusercontent.com/RUCAIBox/POPE/main/data/POPE/{split}_pope.json",
    "https://raw.githubusercontent.com/RUCAIBox/POPE/master/data/POPE/{split}_pope.json",
    "https://raw.githubusercontent.com/RUCAIBox/POPE/main/data/POPE/coco_{split}_pope.json",
    "https://raw.githubusercontent.com/RUCAIBox/POPE/master/data/POPE/coco_{split}_pope.json",
]


def fetch(url, timeout=60):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def image_file(field):
    s = str(field)
    if ".jpg" in s or ".png" in s:
        return os.path.basename(s)
    return f"COCO_val2014_{int(s):012d}.jpg"


def prepare_pope(data_dir, coco_instances=None):
    import random
    rng = random.Random(0)
    inst = None
    cats = None
    if coco_instances and os.path.exists(coco_instances):
        with open(coco_instances) as f:
            d = json.load(f)
        cats = {c["id"]: c["name"] for c in d["categories"]}
        inst = {}
        for a in d["annotations"]:
            inst.setdefault(a["image_id"], []).append((cats[a["category_id"]],
                                                      a["bbox"]))
    for split in ("random", "popular", "adversarial"):
        out_path = os.path.join(data_dir, f"pope_{split}.jsonl")
        if os.path.exists(out_path):
            print("skip existing", out_path)
            continue
        raw = None
        for pat in POPE_URLS:
            raw = fetch(pat.format(split=split))
            if raw is not None:
                break
        if raw is None:
            print(f"FAILED to download POPE {split}; check URLs manually")
            continue
        items = json.loads(raw)
        samples = []
        for k, it in enumerate(items):
            fname = image_file(it.get("image", it.get("image_id")))
            q = it["question"]
            m = re.search(r"Is there (?:a|an) (.+?) in the image", q, re.I)
            obj = m.group(1) if m else None
            boxes = []
            if obj and inst is not None:
                iid = int(re.sub(r"\D", "", fname.split(".")[0]) or 0)
                for name, bbox in inst.get(iid, []):
                    if name == obj:
                        x, y, w, h = bbox
                        boxes = [[x, y, x + w, y + h]]
                        break
            samples.append(dict(id=f"{split}_{k}", image_path=os.path.join(
                data_dir, "images", "val2014", fname), prompt=q,
                answer=it["answer"], boxes=boxes, meta={"image_file": fname}))
        with open(out_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print("wrote", out_path, len(samples))


def ensure_images(data_dir, max_missing=4000):
    img_dir = os.path.join(data_dir, "images", "val2014")
    os.makedirs(img_dir, exist_ok=True)
    files = set()
    for fn in os.listdir(data_dir):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(data_dir, fn)) as f:
            for line in f:
                s = json.loads(line)
                ip = s.get("image_path") or s.get("meta", {}).get("image_file")
                if ip:
                    files.add(os.path.basename(ip))
    missing = [f for f in sorted(files)
               if not os.path.exists(os.path.join(img_dir, f))]
    print(f"{len(missing)} images missing")
    for i, fn in enumerate(missing[:max_missing]):
        raw = fetch(f"http://images.cocodataset.org/val2014/{fn}", timeout=120)
        if raw is None:
            continue
        with open(os.path.join(img_dir, fn), "wb") as f:
            f.write(raw)
        if (i + 1) % 50 == 0:
            print("downloaded", i + 1)


def prepare_refcoco(data_dir, refs_pkl, instances_json, max_samples=800):
    import pickle
    with open(instances_json) as f:
        d = json.load(f)
    images = {im["id"]: im["file_name"] for im in d["images"]}
    anns = {a["id"]: a["bbox"] for a in d["annotations"]}
    with open(refs_pkl, "rb") as f:
        refs = pickle.load(f)
    val = [r for r in refs if r.get("split") == "val"]
    rng = __import__("random").Random(0)
    rng.shuffle(val)
    val = val[:max_samples]
    out = []
    for k, r in enumerate(val):
        sent = r["sentences"][0]["sent"]
        box = anns[r["ann_id"]]
        x, y, w, h = box
        out.append(dict(id=f"refcoco_{k}",
                        image_path=os.path.join(data_dir, "images", "val2014",
                                                images[r["image_id"]]),
                        prompt=sent, boxes=[[x, y, x + w, y + h]],
                        meta={"image_file": images[r["image_id"]],
                              "ann_id": r["ann_id"]}))
    path = os.path.join(data_dir, "refcoco.jsonl")
    with open(path, "w") as f:
        for s in out:
            f.write(json.dumps(s) + "\n")
    print("wrote", path, len(out))


def prepare_synthetic(data_dir, n=400):
    ocr = build_ocr_probe(n=n)
    with open(os.path.join(data_dir, "ocr_synth_synth.jsonl"), "w") as f:
        for s in ocr:
            d = s.to_json()
            d.pop("image", None)
            f.write(json.dumps(d) + "\n")
    print("synthetic ocr/noise probes are generated in-memory at run time;"
          " nothing else to prepare")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--pope", action="store_true")
    ap.add_argument("--refcoco", nargs=2, default=None, metavar=("REFS", "INSTANCES"))
    ap.add_argument("--max-samples", type=int, default=800)
    ap.add_argument("--ensure-images", action="store_true")
    ap.add_argument("--coco-instances", default=None)
    args = ap.parse_args()
    os.makedirs(args.data, exist_ok=True)
    if args.pope:
        prepare_pope(args.data, args.coco_instances)
    if args.refcoco:
        prepare_refcoco(args.data, args.refcoco[0], args.refcoco[1],
                        max_samples=args.max_samples)
    if args.ensure_images:
        ensure_images(args.data)
