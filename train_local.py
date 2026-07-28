"""
train_local.py — CPU-only local training for the drivable-area model.
                 (Phase 7 "make the model better", the version that fits your PC.)

Why this file exists
--------------------
Phase 7 on Colab kept dying on free-tier disconnects ("stuck at epoch 2 for
days" = the runtime was killed long ago; nothing was computing). This script
trains ON YOUR PC — no GPU, no cloud, no online collaborator.

The catch: this laptop is CPU-only (i7, 4 threads). Retraining DeepLabV3 from
scratch on ~7,000 images for 28 epochs — what the Colab notebook does — would
take a week+ on CPU. That is not the right job for this machine.

So this does the CPU-honest thing that still delivers Phase 7's real payoff
(robustness on night / haze / blur / shadow frames):

  * it FINE-TUNES the working Phase 6c weights (LRASPP, val IoU 0.92) instead of
    training from scratch, and
  * it does so with the same night / fog / motion-blur / shadow augmentation, on
    a subset of IDD, at the same 768x432 the app already runs.

Fine-tuning from good weights converges in a handful of epochs, so it finishes
in ~3 hours on CPU instead of a week. The output is the same LRASPP architecture,
so it drops straight into the app — just point LEARNED_MODEL_PATH at the new file.

Advisory / research only — never wire any of this to a vehicle's controls.

Run
---
    # 1) quick self-test (no data, ~5 s) — proves the model + weights load:
    python train_local.py --smoke

    # 2) the real run (extracts IDD once, then trains — leave it overnight):
    python train_local.py

    # options:
    python train_local.py --epochs 8 --subset 2500 --batch 4
"""

import argparse
import glob
import json
import os
import random
import shutil
import tarfile
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

# ── Paths (kept OUT of OneDrive so 20 GB isn't uploaded to the cloud) ──────────
IDD_TAR      = r"C:\Users\Rakesh Sharma\Downloads\idd-segmentation.tar.gz"
DATA_DIR     = r"C:\Users\Rakesh Sharma\Downloads\idd_extracted"   # ~20 GB extracted here
MASK_DIR     = r"C:\Users\Rakesh Sharma\Downloads\idd_masks"       # rasterized masks cache
INIT_WEIGHTS = "drivable_idd_full_best.pth"          # Phase 6c LRASPP — fine-tune FROM this
OUT_WEIGHTS  = "drivable_idd_lraspp_aug_best.pth"    # best weights (what the app loads)
CKPT         = "train_local_ckpt.pth"                # resumable state (survives reboots)

# ── Training config (CPU-sized; override on the command line) ─────────────────
IN_W, IN_H   = 768, 432        # keep same as the deployed model
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)

DRIVABLE_LABELS = {"road", "parking", "drivable fallback"}
MW, MH = 960, 540              # mask rasterization resolution


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────
def _progress(i, n, every=500, what="items"):
    if i % every == 0 or i == n - 1:
        print(f"    {i + 1}/{n} {what}", flush=True)


def build_model():
    """LRASPP MobileNetV3-Large, 2-class head — matches the Phase 6c weights."""
    model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)
    model.classifier.low_classifier  = nn.Conv2d(40, 2, 1)
    model.classifier.high_classifier = nn.Conv2d(128, 2, 1)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Data prep: extract archive, rasterize polygon labels to binary masks
# ─────────────────────────────────────────────────────────────────────────────
def ensure_extracted():
    hit = glob.glob(os.path.join(DATA_DIR, "**", "leftImg8bit"), recursive=True)
    if hit:
        root = os.path.dirname(hit[0])
        print(f"[data] already extracted: {root}")
        return root

    if not os.path.exists(IDD_TAR):
        raise SystemExit(f"[data] archive not found: {IDD_TAR}")

    free_gb = shutil.disk_usage(os.path.dirname(DATA_DIR) or ".").free / 1e9
    print(f"[data] extracting {os.path.basename(IDD_TAR)} -> {DATA_DIR}")
    print(f"[data] one-time, ~20-40 min; needs ~25 GB free (you have {free_gb:.0f} GB)")
    if free_gb < 25:
        print("[data] WARNING: low free disk space — extraction may fail partway.")
    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()
    with tarfile.open(IDD_TAR) as t:
        t.extractall(DATA_DIR)
    print(f"[data] extracted in {time.time() - t0:.0f}s")

    hit = glob.glob(os.path.join(DATA_DIR, "**", "leftImg8bit"), recursive=True)
    if not hit:
        raise SystemExit("[data] extraction finished but no leftImg8bit/ found — unexpected layout.")
    return os.path.dirname(hit[0])


def rasterize_masks(root):
    """Polygon JSON -> binary drivable mask PNG. Cached; only new files are drawn."""
    jobs = []
    for split in ("train", "val"):
        for jp in glob.glob(os.path.join(root, "gtFine", split, "*", "*_polygons.json")):
            rel = os.path.relpath(jp, os.path.join(root, "gtFine"))
            outp = os.path.join(
                MASK_DIR,
                rel.replace("_gtFine_polygons.json", "_drivable.png").replace("_polygons.json", "_drivable.png"),
            )
            jobs.append((jp, outp))

    todo = [(jp, outp) for jp, outp in jobs if not os.path.exists(outp)]
    print(f"[masks] {len(jobs)} labels total, {len(todo)} to rasterize")
    for i, (jp, outp) in enumerate(todo):
        try:
            with open(jp) as f:
                d = json.load(f)
            w, h = d.get("imgWidth", 1920), d.get("imgHeight", 1080)
            m = Image.new("L", (MW, MH), 0)
            draw = ImageDraw.Draw(m)
            sx, sy = MW / w, MH / h
            for obj in d.get("objects", []):
                if obj.get("deleted"):
                    continue
                if str(obj.get("label", "")).lower() in DRIVABLE_LABELS:
                    poly = [(x * sx, y * sy) for x, y in obj.get("polygon", [])]
                    if len(poly) >= 3:
                        draw.polygon(poly, fill=255)
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            m.save(outp)
        except Exception as exc:
            print(f"[masks] skip {jp}: {exc}")
        _progress(i, len(todo), every=500, what="masks")
    print("[masks] ready")


def pairs(root, split):
    out = []
    for ip in sorted(glob.glob(os.path.join(root, "leftImg8bit", split, "*", "*.*"))):
        base = os.path.splitext(os.path.basename(ip))[0].replace("_leftImg8bit", "").replace("_image", "")
        seq = os.path.basename(os.path.dirname(ip))
        cands = glob.glob(os.path.join(MASK_DIR, split, seq, base + "*_drivable.png"))
        if cands:
            out.append((ip, cands[0]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Augmentations (float32 RGB 0-255) — the point of this run
# ─────────────────────────────────────────────────────────────────────────────
def aug_night(img):
    img = np.clip(img * random.uniform(0.35, 0.60), 0, 255)
    mu = img.mean()
    img = np.clip((img - mu) * random.uniform(0.7, 0.9) + mu, 0, 255)
    img[..., 2] = np.clip(img[..., 2] * 1.05, 0, 255)
    return img


def aug_fog(img):
    h = img.shape[0]
    grad = np.linspace(1.0, 0.3, h).reshape(h, 1, 1)
    a = random.uniform(0.2, 0.5) * grad
    haze = np.full_like(img, random.uniform(180, 220))
    return np.clip(img * (1 - a) + haze * a, 0, 255)


def aug_motion_blur(img):
    k = random.choice([5, 7, 9])
    kern = np.zeros((k, k), np.float32)
    if random.random() < 0.5:
        kern[k // 2, :] = 1.0 / k
    else:
        kern[:, k // 2] = 1.0 / k
    return cv2.filter2D(img, -1, kern)


def aug_shadow(img):
    h, w = img.shape[:2]
    x1, x2 = sorted(random.sample(range(w), 2))
    poly = np.array([[x1, 0], [x2, 0],
                     [min(w, x2 + random.randint(-w // 4, w // 4)), h],
                     [max(0, x1 + random.randint(-w // 4, w // 4)), h]], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [poly], 1)
    img[m == 1] = np.clip(img[m == 1] * random.uniform(0.4, 0.7), 0, 255)
    return img


class IDDAug(Dataset):
    """train=True  → random augmentation (the training regime).
       hard=True   → FIXED night+fog per image (a repeatable robustness metric).
       both False  → clean val."""

    def __init__(self, pair_list, train=True, hard=False):
        self.pairs, self.train, self.hard = pair_list, train, hard

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IN_W, IN_H)).astype(np.float32)
        lbl = cv2.resize(lbl, (IN_W, IN_H), interpolation=cv2.INTER_NEAREST)
        if self.train:
            if random.random() < 0.5:
                img, lbl = img[:, ::-1].copy(), lbl[:, ::-1].copy()
            if random.random() < 0.25: img = aug_night(img)
            if random.random() < 0.20: img = aug_fog(img)
            if random.random() < 0.20: img = aug_motion_blur(img)
            if random.random() < 0.20: img = aug_shadow(img)
            if random.random() < 0.30:
                img = np.clip(img * random.uniform(0.7, 1.3), 0, 255)
        elif self.hard:
            # deterministic hard conditions (seeded per image) → comparable across epochs
            st = random.getstate()
            random.seed(10_000 + i)
            img = aug_fog(aug_night(img))
            random.setstate(st)
        x = (img / 255.0 - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()
        y = torch.from_numpy(np.ascontiguousarray((lbl > 127).astype(np.int64)))
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Loss + eval
# ─────────────────────────────────────────────────────────────────────────────
def dice_loss(logits, target, eps=1.0):
    prob = torch.softmax(logits, 1)[:, 1]
    t = (target == 1).float()
    inter = (prob * t).sum((1, 2))
    union = prob.sum((1, 2)) + t.sum((1, 2))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def seg_loss(out, y, ce):
    loss = ce(out["out"], y) + dice_loss(out["out"], y)
    if "aux" in out:                      # LRASPP has no aux; kept for safety
        loss = loss + 0.4 * ce(out["aux"], y)
    return loss


@torch.no_grad()
def val_iou(model, val_dl):
    model.eval()
    inter = union = 0
    for x, y in val_dl:
        p = model(x)["out"].argmax(1)
        inter += ((p == 1) & (y == 1)).sum().item()
        union += ((p == 1) | (y == 1)).sum().item()
    return inter / max(union, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test — no dataset needed; proves the ML path works in seconds
# ─────────────────────────────────────────────────────────────────────────────
def smoke_test():
    print("[smoke] building LRASPP + loading Phase 6c weights ...")
    model = build_model()
    if os.path.exists(INIT_WEIGHTS):
        state = torch.load(INIT_WEIGHTS, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[smoke] weights loaded — missing {len(missing)}, unexpected {len(unexpected)}")
    else:
        print(f"[smoke] WARNING: {INIT_WEIGHTS} not found — running with random weights")
    model.eval()
    ce = nn.CrossEntropyLoss()
    x = torch.randn(1, 3, IN_H, IN_W)
    y = torch.randint(0, 2, (1, IN_H, IN_W))
    t0 = time.time()
    out = model(x)
    fwd = time.time() - t0
    loss = seg_loss(out, y, ce)
    print(f"[smoke] forward ok: out {tuple(out['out'].shape)} | 1-frame {fwd*1000:.0f} ms "
          f"(~{1/fwd:.1f} fps) | loss {loss.item():.3f}")
    print("[smoke] PASS — the model, weights, and loss all work on your CPU.")


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="CPU-only local fine-tuning of the drivable-area model")
    ap.add_argument("--smoke", action="store_true", help="quick self-test, no dataset needed")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--subset", type=int, default=2000, help="train images per epoch (0 = all)")
    ap.add_argument("--val-cap", type=int, default=400, help="val images used for the IoU estimate")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4, help="head LR; backbone uses 0.1x this")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (lower to keep PC usable)")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))

    if args.smoke:
        smoke_test()
        return

    if not os.path.exists(INIT_WEIGHTS):
        raise SystemExit(f"[init] {INIT_WEIGHTS} not found in repo root — needed to fine-tune from.")

    # ── data ──────────────────────────────────────────────────────────────
    root = ensure_extracted()
    print(f"[data] dataset root: {root}")
    rasterize_masks(root)
    train_pairs = pairs(root, "train")
    val_pairs = pairs(root, "val")
    print(f"[data] pairs — train {len(train_pairs)} | val {len(val_pairs)}")
    if len(train_pairs) < 500:
        raise SystemExit("[data] too few pairs — mask/image pairing looks wrong.")

    if args.val_cap and len(val_pairs) > args.val_cap:
        random.seed(0)
        val_pairs = random.sample(val_pairs, args.val_cap)
    val_dl = DataLoader(IDDAug(val_pairs, train=False), batch_size=args.batch, shuffle=False, num_workers=0)
    hard_dl = DataLoader(IDDAug(val_pairs, train=False, hard=True), batch_size=args.batch, shuffle=False, num_workers=0)

    # ── model + optimizer (differential LR, gentle on the backbone) ───────
    model = build_model()
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(),   "lr": args.lr * 0.1},   # pretrained — nudge it
        {"params": model.classifier.parameters(), "lr": args.lr},          # head — adapt faster
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    # ── resume if a checkpoint exists, else start from Phase 6c ───────────
    start_ep, best_iou = 0, None
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep, best_iou = ck["epoch"] + 1, ck["best_iou"]
        print(f"[resume] {CKPT} found — continuing at epoch {start_ep+1} "
              f"(best IoU so far {best_iou:.3f}). Delete {CKPT} to start fresh.")
    else:
        model.load_state_dict(torch.load(INIT_WEIGHTS, map_location="cpu"))
        print(f"[init] fine-tuning from {INIT_WEIGHTS}")

    if best_iou is None:
        print("[eval] baseline IoU (before fine-tuning) ...")
        base_clean = val_iou(model, val_dl)
        best_iou = val_iou(model, hard_dl)      # we select on HARD (night+fog) IoU
        print(f"[eval] baseline — clean {base_clean:.3f} | hard(night+fog) {best_iou:.3f}")
        torch.save(model.state_dict(), OUT_WEIGHTS)   # never end up worse than we started

    if start_ep >= args.epochs:
        print(f"[done] checkpoint already reached {start_ep} epochs — nothing to do.")
        print(f"[done] best weights: {OUT_WEIGHTS}  (delete {CKPT} to train again)")
        return

    for ep in range(start_ep, args.epochs):
        # fresh random subset each epoch → more coverage over the whole run
        if args.subset and len(train_pairs) > args.subset:
            epoch_pairs = random.sample(train_pairs, args.subset)
        else:
            epoch_pairs = train_pairs
        train_dl = DataLoader(IDDAug(epoch_pairs, train=True), batch_size=args.batch,
                              shuffle=True, num_workers=0)

        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        for bi, (x, y) in enumerate(train_dl):
            opt.zero_grad(set_to_none=True)
            loss = seg_loss(model(x), y, ce)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
            if bi % 25 == 0:
                rate = seen / max(1e-6, time.time() - t0)
                eta_h = (len(epoch_pairs) - seen) / max(1e-6, rate) / 3600
                print(f"  epoch {ep+1}/{args.epochs}  {seen}/{len(epoch_pairs)}  "
                      f"loss {running/max(1,seen):.4f}  {rate:.1f} img/s  "
                      f"(~{eta_h:.1f}h left this epoch)", flush=True)
        sched.step()

        clean_iou = val_iou(model, val_dl)
        hard_iou = val_iou(model, hard_dl)      # the metric we optimize for
        dt = (time.time() - t0) / 60.0
        flag = ""
        if hard_iou > best_iou:
            best_iou = hard_iou
            torch.save(model.state_dict(), OUT_WEIGHTS)
            flag = "  <- new best, saved"
        # resume checkpoint AFTER best_iou is updated, so it records the truth
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best_iou": best_iou}, CKPT)
        print(f"[epoch {ep+1}/{args.epochs}] loss {running/max(1,seen):.4f} | "
              f"clean {clean_iou:.3f} | hard {hard_iou:.3f} | best-hard {best_iou:.3f} "
              f"| {dt:.1f} min{flag}", flush=True)

    print(f"\n[done] best hard(night+fog) IoU {best_iou:.3f}")
    print(f"[done] best weights saved to: {OUT_WEIGHTS}")
    print("[done] to use it in the app, set in config.py:")
    print('         LEARNED_ARCH       = "lraspp"')
    print(f'         LEARNED_MODEL_PATH = "{OUT_WEIGHTS}"')
    print("       then:  python main.py --video dashcam.mp4")


if __name__ == "__main__":
    main()
