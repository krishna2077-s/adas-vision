"""
train_bdd.py — Phase 12: fine-tune the drivable-area model on BDD100K.

Builds on the IDD-trained model with REAL adverse driving data. Two ideas make
this worth the download over synthetic augmentation:

  * Weighted sampling of hard conditions. BDD tags every frame night/adverse;
    a WeightedRandomSampler over-samples night (x3) and rain/snow/fog (x2) so a
    CPU-sized subset still sees plenty of the cases that matter.
  * A REAL robustness metric. Model selection is on IoU over BDD val's actual
    night/adverse frames — not a synthetic filter. (If the attributes JSON
    wasn't downloaded, it falls back to a synthetic night+fog+rain hard val.)

Reuses the model, loss, and IoU from train_local, and adverse_aug for train-time
augmentation on top of the real conditions. Resumable (per-epoch checkpoint).

  python prepare_bdd_drivable.py --bdd-root "...\\bdd100k"   # once
  python train_bdd.py --epochs 10 --subset 4000              # then this
  python train_bdd.py --smoke                                # no data needed
"""

import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import adverse_aug
from train_local import IN_W, IN_H, MEAN, STD, build_model, seg_loss, val_iou

MANIFEST = "bdd_manifest.json"
INIT_CANDIDATES = ["drivable_idd_lraspp_adv_best.pth",   # Phase 12 adverse (best if ready)
                   "drivable_idd_lraspp_aug_best.pth",   # adopted augmented model
                   "drivable_idd_full_best.pth"]         # Phase 6c fallback
OUT_WEIGHTS = "drivable_bdd_lraspp_best.pth"
CKPT = "train_bdd_ckpt.pth"


class BDDDataset(Dataset):
    """Binary drivable masks made on the fly (drivable = mask != background)."""

    def __init__(self, entries, bg, train=False, synth_hard=False):
        self.entries, self.bg, self.train, self.synth_hard = entries, bg, train, synth_hard

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        bgr = cv2.imread(e["img"])
        m = cv2.imread(e["mask"], cv2.IMREAD_GRAYSCALE)
        if bgr is None or m is None:                       # unreadable -> empty sample
            x = torch.zeros(3, IN_H, IN_W); return x, torch.zeros(IN_H, IN_W, dtype=torch.long)
        img = cv2.cvtColor(cv2.resize(bgr, (IN_W, IN_H)), cv2.COLOR_BGR2RGB).astype(np.float32)
        m = cv2.resize(m, (IN_W, IN_H), interpolation=cv2.INTER_NEAREST)
        lbl = (m != self.bg).astype(np.int64)              # drivable = 1
        if self.train:
            if random.random() < 0.5:
                img, lbl = img[:, ::-1].copy(), lbl[:, ::-1].copy()
            img = adverse_aug.random_adverse(img, random)  # synthetic on top of real
        elif self.synth_hard:
            img = adverse_aug.hard_adverse(img, 20_000 + i)
        x = (img / 255.0 - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()
        y = torch.from_numpy(np.ascontiguousarray(lbl))
        return x, y


def sample_weights(entries):
    """Over-sample the hard conditions: night x3, adverse weather x2 (multiplicative)."""
    w = []
    for e in entries:
        v = 1.0
        if e.get("night"):   v *= 3.0
        if e.get("adverse"): v *= 2.0
        w.append(v)
    return w


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as fh:
        man = json.load(fh)
    return man, man["background_value"], man["splits"]


# ── smoke test: synthesize a tiny BDD tree, exercise prep + train end-to-end ──
def smoke_test():
    import shutil, tempfile
    import prepare_bdd_drivable as prep
    tmp = tempfile.mkdtemp(prefix="bdd_smoke_")
    print(f"[smoke] synthesizing a tiny fake BDD tree at {tmp}")
    try:
        rng = np.random.default_rng(0)
        attrs = {"train": [], "val": []}
        for split in ("train", "val"):
            idir = os.path.join(tmp, "images", "100k", split)
            mdir = os.path.join(tmp, "labels", "drivable", "masks", split)
            os.makedirs(idir); os.makedirs(mdir)
            for k in range(6):
                name = f"f{split}{k}.jpg"
                cv2.imwrite(os.path.join(idir, name),
                            rng.integers(0, 255, (64, 96, 3), dtype=np.uint8))
                mask = np.full((64, 96), 2, np.uint8)     # bg=2 (majority)
                mask[40:, :] = 0                          # bottom = direct drivable
                cv2.imwrite(os.path.join(mdir, f"f{split}{k}.png"), mask)
                attrs[split].append({"name": name, "attributes": {
                    "timeofday": "night" if k % 2 else "daytime",
                    "weather": "rainy" if k % 3 == 0 else "clear"}})
            with open(os.path.join(tmp, "labels", f"bdd100k_labels_images_{split}.json"),
                      "w") as fh:
                json.dump(attrs[split], fh)

        # run the real prep functions
        idir, mdir, attr = prep.locate(tmp, "train")
        bg, hist = prep.detect_background_value(mdir, sample=12)
        print(f"[smoke] detected background value = {bg} (expect 2) | hist {hist}")
        assert bg == 2, "background auto-detect failed"
        entries, n_night, n_adv, n_tag = prep.build_split(idir, mdir, attr)
        print(f"[smoke] built {len(entries)} pairs | night {n_night} adverse {n_adv} tagged {n_tag}")
        assert len(entries) == 6 and n_tag == 6

        # exercise dataset + one train step + val IoU
        model = build_model()
        ds = BDDDataset(entries, bg, train=True)
        sw = sample_weights(entries)
        dl = DataLoader(ds, batch_size=2,
                        sampler=WeightedRandomSampler(sw, num_samples=4, replacement=True))
        ce = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()
        x, y = next(iter(dl))
        loss = seg_loss(model(x), y, ce); loss.backward(); opt.step()
        iou = val_iou(model, DataLoader(BDDDataset(entries, bg), batch_size=2))
        print(f"[smoke] train step loss {loss.item():.3f} | val IoU {iou:.3f}")
        print("[smoke] PASS — BDD prep + weighted sampling + train + eval all work.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune drivable-area model on BDD100K.")
    ap.add_argument("--smoke", action="store_true", help="synthetic self-test, no data needed")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--subset", type=int, default=4000, help="weighted train draws per epoch")
    ap.add_argument("--val-cap", type=int, default=500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4, help="head LR; backbone uses 0.1x")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    if args.smoke:
        smoke_test()
        return

    if not os.path.exists(args.manifest):
        raise SystemExit(f"[bdd] {args.manifest} not found — run prepare_bdd_drivable.py first.")
    man, bg, splits = load_manifest(args.manifest)
    train_e = splits.get("train", [])
    val_e = splits.get("val", [])
    if len(train_e) < 100:
        raise SystemExit("[bdd] too few train pairs — check the download / manifest.")
    print(f"[data] train {len(train_e)} | val {len(val_e)} | background={bg}")

    # validation: cap for speed; hard = real night/adverse frames (or synthetic fallback)
    random.seed(0)
    val_sample = random.sample(val_e, min(args.val_cap, len(val_e))) if val_e else []
    real_hard = [e for e in val_sample if e.get("night") or e.get("adverse")]
    val_dl = DataLoader(BDDDataset(val_sample, bg), batch_size=args.batch, num_workers=0)
    if len(real_hard) >= 20:
        hard_dl = DataLoader(BDDDataset(real_hard, bg), batch_size=args.batch, num_workers=0)
        hard_name = f"real night/adverse (n={len(real_hard)})"
    else:
        hard_dl = DataLoader(BDDDataset(val_sample, bg, synth_hard=True),
                             batch_size=args.batch, num_workers=0)
        hard_name = "synthetic night+fog+rain (no attributes)"
    print(f"[data] hard-val metric: {hard_name}")

    model = build_model()
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(),   "lr": args.lr * 0.1},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    start_ep, best_iou = 0, None
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start_ep, best_iou = ck["epoch"] + 1, ck["best_iou"]
        print(f"[resume] {CKPT} — continuing at epoch {start_ep+1} (best {best_iou:.3f})")
    else:
        init = next((c for c in INIT_CANDIDATES if os.path.exists(c)), None)
        if not init:
            raise SystemExit(f"[init] none of {INIT_CANDIDATES} found to fine-tune from.")
        model.load_state_dict(torch.load(init, map_location="cpu"))
        print(f"[init] fine-tuning from {init}")

    if best_iou is None:
        base_clean = val_iou(model, val_dl)
        best_iou = val_iou(model, hard_dl)
        print(f"[eval] baseline — clean {base_clean:.3f} | hard {best_iou:.3f}")
        torch.save(model.state_dict(), OUT_WEIGHTS)

    sw = sample_weights(train_e)
    for ep in range(start_ep, args.epochs):
        dl = DataLoader(BDDDataset(train_e, bg, train=True), batch_size=args.batch, num_workers=0,
                        sampler=WeightedRandomSampler(sw, num_samples=min(args.subset, len(train_e)),
                                                      replacement=True))
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        for bi, (x, y) in enumerate(dl):
            opt.zero_grad(set_to_none=True)
            loss = seg_loss(model(x), y, ce); loss.backward(); opt.step()
            running += loss.item() * x.size(0); seen += x.size(0)
            if bi % 25 == 0:
                rate = seen / max(1e-6, time.time() - t0)
                print(f"  epoch {ep+1}/{args.epochs}  {seen}/{min(args.subset, len(train_e))}  "
                      f"loss {running/max(1,seen):.3f}  {rate:.2f} img/s")
        sched.step()

        clean = val_iou(model, val_dl)
        hard = val_iou(model, hard_dl)
        print(f"[epoch {ep+1}] clean {clean:.3f} | hard {hard:.3f} | best {best_iou:.3f}")
        if hard > best_iou:
            best_iou = hard
            torch.save(model.state_dict(), OUT_WEIGHTS)
            print(f"  ** new best hard IoU {best_iou:.3f} -> saved {OUT_WEIGHTS}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best_iou": best_iou}, CKPT)

    print(f"[done] best hard IoU {best_iou:.3f} | weights: {OUT_WEIGHTS}")


if __name__ == "__main__":
    main()
