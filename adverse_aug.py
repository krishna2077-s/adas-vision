"""
adverse_aug.py — Phase 12: real-world adverse-condition augmentation.

A small, self-contained library of photometric weather/lighting effects for
drivable-area training. Each effect takes an RGB image as float32 in [0, 255]
and returns a float32 image in [0, 255] — masks/labels are never touched (these
are appearance changes, not geometry), so the caller keeps the same label.

Every function takes an `rng` that is either the stdlib `random` module or a
`random.Random` instance (both expose random/uniform/randint/choice/sample), so
callers control determinism. `random_adverse()` composes a realistic subset for
training; `hard_adverse()` is a FIXED worst-case (night+fog+rain) for a
repeatable robustness metric across epochs — the same idea as the old night+fog
"hard" val, just tougher and matching the new effects.

Used by both train_local.py (--adverse, on IDD) and train_bdd.py (on BDD100K).
"""

import random as _random

import cv2
import numpy as np


def _nrng(rng) -> np.random.Generator:
    """Derive a numpy Generator seeded from the caller's rng (reproducible)."""
    return np.random.default_rng(rng.randint(0, 2**32 - 1))


# ── individual effects ───────────────────────────────────────────────────────

def night(img, rng):
    """Darken + lower contrast + slight blue lift (dusk/night)."""
    img = np.clip(img * rng.uniform(0.35, 0.60), 0, 255)
    mu = img.mean()
    img = np.clip((img - mu) * rng.uniform(0.7, 0.9) + mu, 0, 255)
    img[..., 2] = np.clip(img[..., 2] * 1.05, 0, 255)
    return img


def lowlight_noise(img, rng):
    """Very dark + Gaussian sensor noise + blue cast (a cheap CMOS at night)."""
    img = np.clip(img * rng.uniform(0.25, 0.45), 0, 255)
    noise = _nrng(rng).normal(0.0, rng.uniform(6.0, 16.0), img.shape)
    img = np.clip(img + noise, 0, 255)
    img[..., 2] = np.clip(img[..., 2] * 1.08, 0, 255)
    return img


def fog(img, rng):
    """Distance-graded haze — thicker toward the top (horizon)."""
    h = img.shape[0]
    grad = np.linspace(1.0, 0.3, h).reshape(h, 1, 1)
    a = rng.uniform(0.2, 0.5) * grad
    haze = np.full_like(img, rng.uniform(180, 220))
    return np.clip(img * (1 - a) + haze * a, 0, 255)


def rain(img, rng):
    """Overcast dim + diagonal streaks + a touch of wet-lens blur."""
    h, w = img.shape[:2]
    img = np.clip(img * rng.uniform(0.6, 0.85), 0, 255)
    nrng = _nrng(rng)
    density = rng.uniform(0.002, 0.006)
    drops = (nrng.random((h, w)) < density).astype(np.float32) * rng.uniform(160, 220)
    # diagonal motion kernel to stretch each drop into a streak
    k = rng.choice([9, 13, 17])
    kern = np.zeros((k, k), np.float32)
    dx = rng.uniform(-0.4, 0.4)
    for t in range(k):
        c = int((k - 1) / 2 + dx * (t - (k - 1) / 2))
        if 0 <= c < k:
            kern[t, c] = 1.0
    kern /= max(1.0, kern.sum())
    streaks = cv2.filter2D(drops, -1, kern)
    img = np.clip(img + streaks[..., None], 0, 255)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def snow(img, rng):
    """Bright, low-contrast, hazy, with soft white flakes."""
    img = np.clip(img * rng.uniform(0.85, 1.0) + rng.uniform(10, 40), 0, 255)
    nrng = _nrng(rng)
    density = rng.uniform(0.01, 0.03)
    flakes = (nrng.random(img.shape[:2]) < density).astype(np.float32)
    flakes = cv2.dilate(flakes, np.ones((2, 2), np.uint8))
    flakes = cv2.GaussianBlur(flakes, (3, 3), 0) * 255.0
    img = np.clip(img + flakes[..., None] * rng.uniform(0.6, 1.0), 0, 255)
    return np.clip(img * 0.9 + 230 * 0.1, 0, 255)      # atmospheric whitening


def glare(img, rng):
    """Bright radial bloom near the top — low sun or oncoming headlights."""
    h, w = img.shape[:2]
    cx = rng.randint(0, w - 1)
    cy = rng.randint(0, max(1, h // 3))
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rad = rng.uniform(0.15, 0.35) * w
    bloom = np.clip(1.0 - d / rad, 0, 1) ** 2
    return np.clip(img + bloom[..., None] * rng.uniform(120, 220), 0, 255)


def motion_blur(img, rng):
    """Directional blur (camera shake / speed)."""
    k = rng.choice([5, 7, 9])
    kern = np.zeros((k, k), np.float32)
    if rng.random() < 0.5:
        kern[k // 2, :] = 1.0 / k
    else:
        kern[:, k // 2] = 1.0 / k
    return cv2.filter2D(img, -1, kern)


def shadow(img, rng):
    """A dark quadrilateral (building/overpass shadow across the road)."""
    h, w = img.shape[:2]
    x1, x2 = sorted(rng.sample(range(w), 2))
    poly = np.array([[x1, 0], [x2, 0],
                     [min(w, x2 + rng.randint(-w // 4, w // 4)), h],
                     [max(0, x1 + rng.randint(-w // 4, w // 4)), h]], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [poly], 1)
    img[m == 1] = np.clip(img[m == 1] * rng.uniform(0.4, 0.7), 0, 255)
    return img


# ── composers ────────────────────────────────────────────────────────────────

def random_adverse(img, rng):
    """
    Compose a realistic adverse scene for training. One primary weather/lighting
    theme (~72% of the time) plus independent extras. ~28% stay near-clean so
    the model keeps its daytime skill (no catastrophic forgetting).
    """
    img = img.astype(np.float32, copy=True)
    theme = rng.random()
    if theme < 0.22:
        img = night(img, rng)
    elif theme < 0.34:
        img = lowlight_noise(img, rng)
    elif theme < 0.50:
        img = fog(img, rng)
    elif theme < 0.62:
        img = rain(img, rng)
    elif theme < 0.72:
        img = snow(img, rng)
    # independent extras (can stack on any theme)
    if rng.random() < 0.22:
        img = shadow(img, rng)
    if rng.random() < 0.18:
        img = glare(img, rng)
    if rng.random() < 0.18:
        img = motion_blur(img, rng)
    if rng.random() < 0.30:
        img = np.clip(img * rng.uniform(0.7, 1.3), 0, 255)   # exposure jitter
    return img


def hard_adverse(img, seed):
    """
    FIXED worst-case (night + fog + rain), seeded per image → a repeatable
    robustness metric that's comparable across epochs and between models.
    """
    rng = _random.Random(seed)
    img = night(img.astype(np.float32, copy=True), rng)
    img = fog(img, rng)
    img = rain(img, rng)
    return np.clip(img, 0, 255)


# effect table for the smoke/preview grid
EFFECTS = {
    "night": night, "lowlight_noise": lowlight_noise, "fog": fog,
    "rain": rain, "snow": snow, "glare": glare,
    "motion_blur": motion_blur, "shadow": shadow,
}
