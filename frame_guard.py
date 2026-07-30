"""
frame_guard.py — Phase 13: reject malformed frames before they reach perception.

A real ADAS reads from a camera or a file that can hiccup: a dropped grab returns
None, a half-decoded frame comes back the wrong shape, a driver glitch yields a
1-pixel or non-uint8 buffer. None of that should crash the pipeline or feed
garbage into the models. This is the single check the main loop runs on every
frame, and that each perception module runs defensively on its own input, so a
bad frame degrades to "no perception this frame" (the decision engine then holds
/ goes conservative) instead of an exception mid-drive.
"""

import numpy as np


def is_valid_frame(frame) -> bool:
    """
    True only for a usable BGR image: a real H×W×3 uint8 ndarray with area.

    Deliberately strict and cheap — it rejects the failure modes a capture can
    actually produce (None, wrong ndim/channels, degenerate size, wrong dtype)
    without scanning pixel values (a uint8 buffer can't carry NaN/inf, and an
    all-black frame is *valid* input that the modules already handle by finding
    nothing).
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return False
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        return False
    if frame.dtype != np.uint8:
        return False
    return True
