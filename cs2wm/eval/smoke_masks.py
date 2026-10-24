"""Smoke masks for H2 (#33), from smoke events and the spectated player's eye pose.

Each smoke volume is projected as a sphere (``eval.h2.smoke_radius_units``) to a screen ellipse and dilated by
``eval.h2.mask_dilate_px``. Wall occlusion is ignored; this is a documented limitation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def project_sphere(
    center: np.ndarray,
    radius: float,
    eye: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    hfov_deg: float,
    size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Project a world-space sphere to a screen ellipse.

    Args:
        center: ``float[3]`` sphere centre (Source units).
        radius: Sphere radius in Source units.
        eye: ``float[3]`` eye position (X/Y/Z plus view offset).
        yaw_deg: Source yaw of the camera.
        pitch_deg: Source pitch of the camera.
        hfov_deg: Horizontal field of view.
        size: Image ``(H, W)``.

    Returns:
        ``(cx, cy, rx, ry)`` in pixels, or None if the sphere is outside the frustum.
    """
    # TODO(#33): pinhole projection with the Source axis conventions.
    raise NotImplementedError("TODO(#33): project_sphere")


def smoke_mask_for_frame(
    tick: int,
    events: pd.DataFrame,
    eye: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    cfg: DictConfig,
    size: Sequence[int] = (150, 280),
) -> np.ndarray:
    """Boolean smoke mask of one frame.

    Args:
        tick: Frame tick.
        events: Smoke table (detonate/expired tick, x/y/z, thrower).
        eye: ``float[3]`` eye position.
        yaw_deg: Source yaw of the camera.
        pitch_deg: Source pitch of the camera.
        cfg: A validated cs2wm config (``eval.h2.*``).
        size: Image ``(H, W)``.

    Returns:
        ``bool[H, W]`` mask of pixels inside an active smoke, dilated.
    """
    # TODO(#33): union of project_sphere ellipses for smokes active at `tick`, then dilate.
    raise NotImplementedError("TODO(#33): smoke_mask_for_frame")
