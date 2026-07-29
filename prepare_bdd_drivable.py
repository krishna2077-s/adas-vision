"""
prepare_bdd_drivable.py — Phase 12: index BDD100K drivable-area data.

BDD100K has ~70k train / 10k val real driving images WITH per-image weather and
time-of-day attributes — real night/rain/snow/fog, not synthetic. This script
does NOT copy or convert anything heavy; it writes a small manifest that
train_bdd.py consumes:

  * finds the images + drivable masks (tries the standard BDD layouts),
  * auto-detects which mask value is "background" (the modal value over a
    sample — robust to whichever 0/1/2 convention a download uses), so the
    dataset can make a binary drivable mask on the fly (drivable = value != bg),
  * joins the attributes JSON (if present) to tag each frame night / adverse,
    used by train_bdd's weighted sampler to over-sample hard conditions.

Download (needs a free https://bdd-data.berkeley.edu account):
  * "100K Images"            -> bdd100k/images/100k/{train,val}
  * "Drivable Area" labels   -> bdd100k/labels/drivable/masks/{train,val}
  * (optional) "Detection" labels JSON -> per-frame weather/timeofday attributes
Unzip all under one folder and pass it as --bdd-root.

  python prepare_bdd_drivable.py --bdd-root "C:\\Users\\...\\Downloads\\bdd100k"
"""

import argparse
import json
import os
import random

import cv2
import numpy as np


def _first_dir(root, candidates):
    for c in candidates:
        p = os.path.join(root, *c)
        if os.path.isdir(p):
            return p
    return None


def _first_file(root, candidates):
    for c in candidates:
        p = os.path.join(root, *c)
        if os.path.isfile(p):
            return p
    return None


def locate(root, split):
    """Return (images_dir, masks_dir, attr_json) for a split, trying BDD layouts."""
    img = _first_dir(root, [("images", "100k", split),
                            ("bdd100k", "images", "100k", split)])
    msk = _first_dir(root, [("labels", "drivable", "masks", split),
                            ("bdd100k", "labels", "drivable", "masks", split),
                            ("labels", "drivable", "colormaps", split)])
    attr = _first_file(root, [("labels", f"bdd100k_labels_images_{split}.json"),
                              ("bdd100k", "labels", f"bdd100k_labels_images_{split}.json"),
                              ("labels", "det_20", f"det_{split}.json"),
                              ("bdd100k", "labels", "det_20", f"det_{split}.json")])
    return img, msk, attr


def detect_background_value(masks_dir, sample=200):
    """The background is the globally most-common mask value across a sample."""
    files = [f for f in os.listdir(masks_dir) if f.lower().endswith(".png")]
    random.seed(0)
    files = random.sample(files, min(sample, len(files)))
    hist = np.zeros(256, dtype=np.int64)
    for f in files:
        m = cv2.imread(os.path.join(masks_dir, f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        hist += np.bincount(m.ravel(), minlength=256)
    bg = int(hist.argmax())
    nz = {int(v): int(hist[v]) for v in np.nonzero(hist)[0]}
    return bg, nz


def load_attributes(attr_json):
    """name -> (is_night, is_adverse) from the BDD attributes JSON, if available."""
    if not attr_json:
        return {}
    with open(attr_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # both legacy (list of frames) and det_20 (list) share {"name","attributes"}
    frames = data if isinstance(data, list) else data.get("frames", [])
    tags = {}
    for fr in frames:
        a = fr.get("attributes", {}) or {}
        tod = str(a.get("timeofday", "")).lower()
        wx = str(a.get("weather", "")).lower()
        night = ("night" in tod) or ("dawn" in tod) or ("dusk" in tod)
        adverse = any(w in wx for w in ("rain", "snow", "fog"))
        tags[fr.get("name", "")] = (night, adverse)
    return tags


def build_split(images_dir, masks_dir, attr_json):
    tags = load_attributes(attr_json)
    masks = {os.path.splitext(f)[0]: f for f in os.listdir(masks_dir)
             if f.lower().endswith(".png")}
    entries, n_night, n_adv, n_tagged = [], 0, 0, 0
    for f in os.listdir(images_dir):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(f)[0]
        if stem not in masks:
            continue
        night, adverse = tags.get(f, tags.get(stem, (False, False)))
        if f in tags or stem in tags:
            n_tagged += 1
        n_night += night
        n_adv += adverse
        entries.append({
            "img": os.path.join(images_dir, f),
            "mask": os.path.join(masks_dir, masks[stem]),
            "night": bool(night), "adverse": bool(adverse),
        })
    return entries, n_night, n_adv, n_tagged


def main():
    ap = argparse.ArgumentParser(description="Index BDD100K drivable-area data into a manifest.")
    ap.add_argument("--bdd-root", default=r"C:\Users\Rakesh Sharma\Downloads\bdd100k")
    ap.add_argument("--out", default="bdd_manifest.json")
    ap.add_argument("--bg-value", type=int, default=-1,
                    help="override background mask value (default: auto-detect)")
    args = ap.parse_args()

    if not os.path.isdir(args.bdd_root):
        raise SystemExit(f"[bdd] root not found: {args.bdd_root}\n"
                         f"      Download 100K Images + Drivable Area labels and unzip under it.")

    manifest = {"bdd_root": args.bdd_root, "splits": {}}
    bg_value = args.bg_value
    for split in ("train", "val"):
        img_dir, msk_dir, attr = locate(args.bdd_root, split)
        if not img_dir or not msk_dir:
            print(f"[{split}] MISSING — images:{bool(img_dir)} masks:{bool(msk_dir)} "
                  f"(skipping; check the download layout)")
            continue
        if bg_value < 0:
            bg_value, hist = detect_background_value(msk_dir)
            print(f"[{split}] mask value histogram (sample): {hist}")
            print(f"[{split}] -> background value = {bg_value}  (drivable = value != {bg_value})")
        entries, n_night, n_adv, n_tagged = build_split(img_dir, msk_dir, attr)
        manifest["splits"][split] = entries
        print(f"[{split}] {len(entries)} image/mask pairs | attributes: "
              f"{'yes' if attr else 'NONE (uniform sampling)'} "
              f"({n_tagged} tagged) | night {n_night} | adverse {n_adv}")

    manifest["background_value"] = bg_value
    if not manifest["splits"]:
        raise SystemExit("[bdd] no usable splits found — check --bdd-root layout.")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    print(f"[done] wrote {args.out}  ->  now run:  python train_bdd.py")


if __name__ == "__main__":
    main()
