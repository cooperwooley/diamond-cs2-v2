"""Ego-rotation estimation from generated frames, for H1 (#32).

Estimates the per-frame image shift between consecutive frames and converts it to degrees. Validated on ground-truth
test episodes (``h1_controllability.validate_estimator``): estimated d-yaw must correlate with the label-implied d-yaw
at r >= 0.8, and the slope fitted there replaces the theoretical pixel-to-degree conversion for model rollouts.

Sign conventions
----------------
* ``dx, dy`` are the displacement of the scene **content** from ``prev`` to ``curr`` in pixels: ``dx > 0`` means the
  content moved right, ``dy > 0`` means it moved down (image coordinates, y grows downward). Both estimators follow
  this, and tests pin it on synthetic shifted images.
* Turning right moves the content left (``dx < 0``) and looking down moves it up (``dy < 0``). The repo's action
  convention is +x = turn right and +y = look down, so ``d_yaw = -k * dx`` and ``d_pitch = -k * dy`` with ``k > 0``:
  a turn to the right gives a positive estimated d-yaw, looking down a positive estimated d-pitch.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import cv2
import numpy as np

# The HLAE capture size (``resolution.capture``): the stored frames are a horizontally squashed version of it.
DEFAULT_CAPTURE_HW = (720, 1280)


class Shift(NamedTuple):
    """A pixel shift estimate.

    Attributes:
        dx: Horizontal content displacement in pixels (+ = right).
        dy: Vertical content displacement in pixels (+ = down).
        response: Peak response of the phase correlation (higher is a sharper peak, not bounded by 1); NaN for the flow
            estimator, which has no equivalent.
    """

    dx: float
    dy: float
    response: float


def _gray32(img: np.ndarray) -> np.ndarray:
    """``uint8[H, W, 3]`` BGR (or ``[H, W]``) to ``float32[H, W]`` grayscale."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return gray.astype(np.float32)


def estimate_shift_phasecorr(prev: np.ndarray, curr: np.ndarray) -> Shift:
    """Estimate the shift with ``cv2.phaseCorrelate`` on Hann-windowed grayscale images.

    Phase correlation assumes one global translation; with a rotating camera the real shift is non-uniform (see
    ``shift_to_degrees``), and the peak lands at a weighted average of it.

    Args:
        prev: ``uint8[H, W, 3]`` BGR earlier frame.
        curr: ``uint8[H, W, 3]`` BGR later frame.

    Returns:
        The content displacement from ``prev`` to ``curr`` and the correlation response.
    """
    a, b = _gray32(prev), _gray32(curr)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    return Shift(float(dx), float(dy), float(response))


def _farneback(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Farneback flow ``float32[H, W, 2]`` (x, y) of the content from ``prev`` to ``curr``."""
    a = _gray32(prev).clip(0, 255).astype(np.uint8)
    b = _gray32(curr).clip(0, 255).astype(np.uint8)
    # A smaller window for the tiny 30x56 frames, where 15 px would span a quarter of the image.
    winsize = 15 if min(a.shape) >= 100 else 7
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, winsize, 3, 5, 1.1, 0)


def estimate_shift_flow(prev: np.ndarray, curr: np.ndarray, border: int = 0) -> Shift:
    """Estimate the shift as the median of Farneback optical flow.

    Args:
        prev: ``uint8[H, W, 3]`` BGR earlier frame.
        curr: ``uint8[H, W, 3]`` BGR later frame.
        border: Pixels to exclude on every side (flow is unreliable at the edges).

    Returns:
        The median content displacement from ``prev`` to ``curr``; ``response`` is NaN.

    Raises:
        ValueError: If ``border`` leaves no pixels.
    """
    flow = _farneback(prev, curr)
    h, w = flow.shape[:2]
    if border < 0 or 2 * border >= min(h, w):
        raise ValueError(f"border {border} leaves no pixels in a {h}x{w} image")
    inner = flow[border : h - border, border : w - border]
    return Shift(float(np.median(inner[..., 0])), float(np.median(inner[..., 1])), float("nan"))


def flow_divergence(prev: np.ndarray, curr: np.ndarray, border: int = 0) -> float:
    """Mean divergence of the Farneback flow, in 1/frame: positive when the scene expands (moving forward).

    Used for the optional W/S check: moving forward expands the image around the focus of expansion, backward
    contracts it. Rotation has (nearly) zero mean divergence.

    Args:
        prev: ``uint8[H, W, 3]`` BGR earlier frame.
        curr: ``uint8[H, W, 3]`` BGR later frame.
        border: Pixels to exclude on every side.

    Returns:
        ``mean(du/dx + dv/dy)`` over the interior.
    """
    flow = _farneback(prev, curr)
    div = np.gradient(flow[..., 0], axis=1) + np.gradient(flow[..., 1], axis=0)
    h, w = div.shape
    return float(div[border : h - border, border : w - border].mean())


def focal_lengths_px(
    width: int, height: int, hfov_deg: float, capture_hw: tuple[int, int] = DEFAULT_CAPTURE_HW
) -> tuple[float, float]:
    """Pinhole focal lengths ``(f_x, f_y)`` in pixels of a stored frame.

    ``f_x = (W / 2) / tan(hfov / 2)`` and ``f_y = (H / 2) / tan(vfov / 2)``, with the vertical FOV taken from the
    capture aspect ratio, ``tan(vfov / 2) = tan(hfov / 2) * H_cap / W_cap``. The stored frames (150x280) have a
    slightly different aspect ratio from the capture (720x1280, 16:9) because they are squashed horizontally, as in
    v1; ``f_x`` and ``f_y`` are therefore computed separately from the stored ``W`` and ``H`` and differ.

    Args:
        width: Stored frame width.
        height: Stored frame height.
        hfov_deg: True horizontal FOV of the capture in degrees (``eval.h1.hfov_deg``).
        capture_hw: ``(H, W)`` of the capture (``resolution.capture``).

    Returns:
        ``(f_x, f_y)`` in pixels.
    """
    tan_h = math.tan(math.radians(hfov_deg) / 2)
    tan_v = tan_h * capture_hw[0] / capture_hw[1]
    return (width / 2) / tan_h, (height / 2) / tan_v


def shift_to_degrees(
    dx: float | np.ndarray,
    dy: float | np.ndarray,
    width: int,
    height: int,
    hfov_deg: float,
    capture_hw: tuple[int, int] = DEFAULT_CAPTURE_HW,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert a content shift in pixels to ``(d_yaw, d_pitch)`` degrees with the pinhole model.

    ``d_yaw = -atan(dx / f_x)`` and ``d_pitch = -atan(dy / f_y)`` (see ``focal_lengths_px``); the minus signs make a
    right turn (content moves left) a positive d-yaw and looking down (content moves up) a positive d-pitch, the
    convention of ``cs2wm.actions``.

    A pinhole camera shifts the image non-uniformly: a rotation by ``a`` moves a point at angle ``t`` off-axis by
    ``f (tan(t + a) - tan(t))``, more at the edges than in the centre. Phase correlation over the whole frame returns
    one number that averages this, so the conversion is only approximately right. That is exactly why H1 refits the
    px-to-degree slope on ground-truth frames and uses this function as the theoretical reference.

    Args:
        dx: Horizontal content displacement in pixels (+ = right).
        dy: Vertical content displacement in pixels (+ = down).
        width: Stored frame width.
        height: Stored frame height.
        hfov_deg: True horizontal FOV in degrees.
        capture_hw: ``(H, W)`` of the capture, which fixes the vertical FOV.

    Returns:
        ``(d_yaw, d_pitch)`` in degrees; arrays in, arrays out.
    """
    f_x, f_y = focal_lengths_px(width, height, hfov_deg, capture_hw)
    return -np.degrees(np.arctan(np.asarray(dx) / f_x))[()], -np.degrees(np.arctan(np.asarray(dy) / f_y))[()]


def theoretical_slopes(
    width: int, height: int, hfov_deg: float, capture_hw: tuple[int, int] = DEFAULT_CAPTURE_HW
) -> tuple[float, float]:
    """Degrees of rotation per pixel of content shift near the image centre, per axis.

    The small-angle limit of ``shift_to_degrees``: ``deg(1 / f_x)`` and ``deg(1 / f_y)``. Positive; a yaw of
    ``slope_x * (-dx)`` degrees.

    Returns:
        ``(deg_per_px_x, deg_per_px_y)``.
    """
    f_x, f_y = focal_lengths_px(width, height, hfov_deg, capture_hw)
    return math.degrees(1 / f_x), math.degrees(1 / f_y)
