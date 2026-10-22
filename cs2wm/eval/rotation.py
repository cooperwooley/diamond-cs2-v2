"""Ego-rotation estimation from generated frames, for H1 (#32).

Estimates the per-frame image shift between consecutive frames and converts it to degrees with the horizontal FOV
(``eval.h1.hfov_deg``). Validated on ground-truth test episodes: estimated d-yaw must correlate with the label-implied
d-yaw at r >= 0.8.
"""

from __future__ import annotations

import numpy as np


def estimate_shift_phasecorr(prev: np.ndarray, curr: np.ndarray) -> tuple[float, float]:
    """Estimate the shift with ``cv2.phaseCorrelate`` on Hann-windowed grayscale images.

    Args:
        prev: ``uint8[H, W, 3]`` earlier frame.
        curr: ``uint8[H, W, 3]`` later frame.

    Returns:
        ``(dx, dy)`` in pixels.
    """
    # TODO(#32): grayscale, cv2.createHanningWindow, cv2.phaseCorrelate.
    raise NotImplementedError("TODO(#32): estimate_shift_phasecorr")


def estimate_shift_flow(prev: np.ndarray, curr: np.ndarray) -> tuple[float, float]:
    """Estimate the shift as the median of Farneback optical flow.

    Args:
        prev: ``uint8[H, W, 3]`` earlier frame.
        curr: ``uint8[H, W, 3]`` later frame.

    Returns:
        ``(dx, dy)`` in pixels.
    """
    # TODO(#32): cv2.calcOpticalFlowFarneback, median over pixels.
    raise NotImplementedError("TODO(#32): estimate_shift_flow")


def shift_to_degrees(dx: float, dy: float, width: int, hfov_deg: float) -> tuple[float, float]:
    """Convert a pixel shift to ``(d_yaw, d_pitch)`` degrees.

    Args:
        dx: Horizontal shift in pixels.
        dy: Vertical shift in pixels.
        width: Image width in pixels.
        hfov_deg: Horizontal field of view in degrees.

    Returns:
        Rotation in degrees, with the sign convention of ``cs2wm.actions`` (+x right, +y down).
    """
    # TODO(#32): deg_per_px = hfov_deg / width (refit the slope on ground truth).
    raise NotImplementedError("TODO(#32): shift_to_degrees")
