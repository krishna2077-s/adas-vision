"""
train_signs.py — CPU trainer for the Module 11b traffic-sign classifier.

Trains the small CNN in traffic_sign_recognition.py on GTSRB (German Traffic
Sign Recognition Benchmark, 43 classes). GTSRB is tiny by modern standards, so
this reaches ~98% val accuracy in a handful of epochs — a few minutes on a
laptop CPU, no GPU needed.

The architecture and the crop->tensor preprocessing are imported from
traffic_sign_recognition so training and inference can never drift apart.

Dataset layout (either is accepted) — a folder per class id 0..42:
    <GTSRB_DIR>/0/*.png        <GTSRB_DIR>/00000/*.ppm
    <GTSRB_DIR>/1/*.png        <GTSRB_DIR>/00001/*.ppm
    ...                        ...
The Kaggle "GTSRB - German Traffic Sign" Train/ folder is exactly this. Point
GTSRB_DIR at it (or pass --data).

Quick start:
    python train_signs.py --smoke          # no dataset needed: proves the loop runs
    python train_signs.py --epochs 8       # real training -> gtsrb_sign_cnn.pth

Output weights: gtsrb_sign_cnn.pth (config.SIGN_MODEL_PATH). Once present,
main.py picks up sign recognition automatically.
"""

import argparse
import glob
import logging
import os
import random

import cv2
import numpy as np

import config as cfg
from traffic_sign_recognition import (build_sign_cnn, sign_crop_to_tensor,
                                      NUM_CLASSES, GTSRB_CLASSES)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# --- Config (edit these or pass CLI flags) ---------------------------------
GTSRB_DIR   = r"C:\Users\Rakesh Sharma\Downloads\GTSRB\Train"   # folder-per-class root
OUT_WEIGHTS = cfg.SIGN_MODEL_PATH                                # gtsrb_sign_cnn.pth
SIZE        = cfg.SIGN_INPUT_SIZE
IMG_EXTS    = ("*.png", "*.ppm", "*.jpg", "*.jpeg", "*.bmp")


def _class_dir(root, cls):
    for name in (str(cls), f"{cls:05d}", f"{cls:02d}"):
        d = os.path.join(root, name)
        if os.path.isdir(d):
            return d
    return None


def list_samples(root, subset=0):
    """Return (path, class_id) pairs across all present class folders."""
    items = []
    for cls in range(NUM_CLASSES):
        d = _class_dir(root, cls)
        if d is None:
            continue
        paths = []
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(d, ext)))
        for p in paths:
            items.append((p, cls))
    if subset and len(items) > subset:
        rng = random.Random(0)
        items = rng.sample(items, subset)
    return items


class GTSRBData:
    """Minimal indexable dataset — reads, resizes, and tensorises on the fly."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def get(self, i):
        import torch
        path, cls = self.items[i]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        return sign_crop_to_tensor(img, SIZE), cls


def _smoke():
    """Prove the training loop runs with zero external data (random tensors)."""
    import torch
    from torch import nn, optim
    logger.info("SMOKE: 4 random batches through the real architecture ...")
    model = build_sign_cnn()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    model.train()
    first = last = None
    for step in range(4):
        x = torch.rand(16, 3, SIZE, SIZE)
        y = torch.randint(0, NUM_CLASSES, (16,))
        opt.zero_grad()
        out = model(x)
        assert out.shape == (16, NUM_CLASSES), out.shape
        loss = lossf(out, y)
        loss.backward(); opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
        logger.info(f"  step {step}  loss {loss.item():.3f}")
    logger.info(f"SMOKE PASS — arch OK, loop OK (loss {first:.3f} -> {last:.3f}).")


def main():
    ap = argparse.ArgumentParser(description="Train the GTSRB traffic-sign classifier (CPU).")
    ap.add_argument("--data", default=GTSRB_DIR, help="GTSRB folder-per-class root.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--subset", type=int, default=0, help="cap total samples (0 = all).")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default).")
    ap.add_argument("--smoke", action="store_true", help="run a no-data loop check and exit.")
    args = ap.parse_args()

    import torch
    from torch import nn, optim
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.smoke:
        _smoke()
        return

    if not os.path.isdir(args.data):
        logger.error(f"GTSRB dir not found: {args.data}")
        logger.error("Download GTSRB (Kaggle: 'GTSRB - German Traffic Sign') and point "
                     "--data at its Train/ folder, or edit GTSRB_DIR in this file.")
        logger.error("No dataset handy? Run `python train_signs.py --smoke` to verify the loop.")
        return

    items = list_samples(args.data, subset=args.subset)
    if not items:
        logger.error(f"No class folders / images found under {args.data}.")
        return
    random.Random(1).shuffle(items)
    n_val = max(1, int(len(items) * args.val_frac))
    val_items, train_items = items[:n_val], items[n_val:]
    n_classes_present = len({c for _, c in items})
    logger.info(f"{len(items)} images, {n_classes_present}/{NUM_CLASSES} classes present "
                f"-> train {len(train_items)} / val {len(val_items)}")

    train_ds, val_ds = GTSRBData(train_items), GTSRBData(val_items)
    model = build_sign_cnn()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()

    def batches(ds, bs, shuffle):
        idx = list(range(len(ds)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), bs):
            xs, ys = [], []
            for j in idx[i:i + bs]:
                x, y = ds.get(j)
                if x is not None:
                    xs.append(x); ys.append(y)
            if xs:
                yield torch.stack(xs), torch.tensor(ys)

    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        seen = correct = 0
        run_loss = 0.0
        for xb, yb in batches(train_ds, args.batch, True):
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward(); opt.step()
            run_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            seen += len(yb)
        sched.step()
        tr_acc = correct / max(1, seen)

        model.eval()
        vseen = vcorrect = 0
        with torch.no_grad():
            for xb, yb in batches(val_ds, args.batch, False):
                vcorrect += (model(xb).argmax(1) == yb).sum().item()
                vseen += len(yb)
        val_acc = vcorrect / max(1, vseen)
        logger.info(f"epoch {ep:2d}/{args.epochs}  loss {run_loss/max(1,seen):.3f}  "
                    f"train_acc {tr_acc:.3f}  val_acc {val_acc:.3f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), OUT_WEIGHTS)
            logger.info(f"  saved {OUT_WEIGHTS} (val_acc {val_acc:.3f})")

    logger.info(f"Done. Best val_acc {best_acc:.3f}. Weights: {OUT_WEIGHTS}")


if __name__ == "__main__":
    main()
