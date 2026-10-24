"""Smoke and static masks for H2 (#33), from smoke events and the spectated player's eye pose.

Each smoke volume is a sphere (``eval.h2.smoke_radius_units``) centred on the detonation point. A pixel is in the
smoke if its camera ray hits the sphere; the union over the smokes active at the frame's tick, dilated by
``eval.h2.mask_dilate_px``, is the smoke mask. The static mask is its complement minus a viewmodel box, and
only exists for frames with no enemy in view.

Conventions (each is pinned by ``tests/test_smoke_masks.py``)
-------------------------------------------------------------
* **World axes** (Source): x forward/east, y left/north, z up. Units are Source units.
* **Angles**: yaw rotates about z, counter-clockwise positive (yaw 0 faces +x, yaw 90 faces +y). Pitch is positive
  looking DOWN, which is what the demo and the repo's labels use (``+y`` mouse bin = look down = Source pitch up).
  The camera has no roll.
* **View basis** (``view_basis``): ``forward = (cos p cos y, cos p sin y, -sin p)``, ``right = (sin y, -cos y, 0)``,
  ``up = (sin p cos y, sin p sin y, cos p)``. Facing +x, the right-hand side is -y.
* **Camera space**: ``(x, y, z) = (d . right, d . up, d . forward)`` for ``d = point - eye``; ``z`` is the depth.
* **Pixels**: ``u = W/2 + f_x x / z`` (columns, increasing to the right) and ``v = H/2 - f_y y / z`` (rows,
  increasing downwards). A pixel ``(row, col)`` is sampled at its centre ``(col + 0.5, row + 0.5)``.
* **Focal lengths**: per-axis pinhole as in H1 (``rotation.focal_lengths_px``): ``f_x`` from ``eval.h1.hfov_deg`` over
  the frame width, ``f_y`` from the capture aspect ratio over the frame height. The stored frames are squashed
  horizontally, so ``f_x != f_y``, and a sphere is projected with its own per-axis angular extent.
* **Sphere mask** (``sphere_mask``): exact per pixel. The ray of pixel ``(col, row)`` in camera space is
  ``normalise((col + 0.5 - W/2) / f_x, -(row + 0.5 - H/2) / f_y, 1)``; with ``c`` the sphere centre in camera space it
  hits the sphere iff ``b = ray . c > 0`` and ``b^2 - (|c|^2 - r^2) >= 0``, or the camera is inside the sphere
  (``|c| < r``), which masks the whole frame. A sphere that crosses the camera plane is therefore handled exactly
  (an approximating ellipse is not: it is far off for the near, large smokes that matter most).
* **Eye position** (``eye_position``): pawn origin + view offset. The offset is the tick table's view-offset column
  when ``eval.h2.view_offset_prop`` names one, else it is interpolated between the standing and the crouching
  height (64 and 46 units) by ``duck_amount``. CONFIRM(#14): the demoparser2 prop and whether those heights hold in
  CS2.
* **Frame to tick**: frame ``k`` shows the demo state at ``start_tick + k * ticks_per_frame`` (``render.tick_offset``
  only shifts when the render commands are scheduled, see ``pipeline/segment.py``).
* **Active smoke**: ``detonate_tick <= tick < expire_tick``.
* **Line-of-sight filter** (``eval.h2.max_los_units``): a smoke only counts if its centre is within that distance of
  the view axis (the ray from the eye along ``forward``; behind the eye, of the eye itself). Smokes far off to the
  side are the likeliest to be hidden behind a wall, so dropping them lowers the false positive rate.
* **Static mask**: not smoke, not in the viewmodel box (``eval.h2.viewmodel_box``, ``[x0, y0, x1, y1]`` as fractions
  of W and H), and only on frames where no living enemy is in view. An enemy is any other player in the tick table
  (the bot deathmatch has no teams to tell friends from enemies); it is in view when, within
  ``eval.h2.enemy_max_dist_units``, the origin, the middle or the head projects inside the frame grown by
  ``eval.h2.enemy_margin_px``. Walls are ignored here too, so a frame with an enemy behind a wall is dropped (safe).

Limitation: occlusion by walls and other geometry is ignored. A smoke behind a wall, or one that the sphere
over-covers (the real volume is irregular and shrinks as it dissipates), still yields mask pixels. Dilation and the
line-of-sight filter keep false positives low but do not remove them; the masks are geometric approximations.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

import cv2
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from cs2wm.eval import rotation

PLAYER_HEIGHT_UNITS = 72.0  # origin to the top of the head of a standing player
_LIMIT_PX = 1e5  # stand-in for an unbounded edge of a sphere that crosses the camera plane


def view_basis(yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the camera's ``(forward, right, up)`` unit vectors in world coordinates.

    Args:
        yaw_deg: Source yaw (counter-clockwise positive about z).
        pitch_deg: Source pitch (positive looks down).

    Returns:
        Three ``float[3]`` vectors; see the module docstring for the formulas.
    """
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    forward = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p)])
    right = np.array([math.sin(y), -math.cos(y), 0.0])
    up = np.array([math.sin(p) * math.cos(y), math.sin(p) * math.sin(y), math.cos(p)])
    return forward, right, up


def _camera_space(point: np.ndarray, eye: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    forward, right, up = view_basis(yaw_deg, pitch_deg)
    d = np.asarray(point, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    return np.array([d @ right, d @ up, d @ forward])


def _focal(size: tuple[int, int], hfov_deg: float, capture_hw: tuple[int, int]) -> tuple[float, float]:
    return rotation.focal_lengths_px(size[1], size[0], hfov_deg, capture_hw)


def project_point(
    point: np.ndarray,
    eye: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    hfov_deg: float,
    size: tuple[int, int],
    capture_hw: tuple[int, int] = rotation.DEFAULT_CAPTURE_HW,
) -> tuple[float, float, float] | None:
    """Project a world-space point to pixel coordinates.

    Args:
        point: ``float[3]`` world position.
        eye: ``float[3]`` eye position.
        yaw_deg: Source yaw of the camera.
        pitch_deg: Source pitch of the camera.
        hfov_deg: Horizontal field of view of the capture.
        size: Image ``(H, W)``.
        capture_hw: ``(H, W)`` of the capture, which sets the vertical FOV.

    Returns:
        ``(u, v, depth)`` with ``u`` the column and ``v`` the row coordinate (not clipped to the frame), or None if the
        point is not in front of the camera.
    """
    x, y, z = _camera_space(point, eye, yaw_deg, pitch_deg)
    if z <= 1e-6:
        return None
    f_x, f_y = _focal(size, hfov_deg, capture_hw)
    return size[1] / 2 + f_x * x / z, size[0] / 2 - f_y * y / z, float(z)


@functools.lru_cache(maxsize=8)
def _pixel_rays(size: tuple[int, int], hfov_deg: float, capture_hw: tuple[int, int]) -> np.ndarray:
    """Unit camera-space rays ``(right, up, forward)`` through the pixel centres: ``float[H, W, 3]``."""
    h, w = size
    f_x, f_y = _focal(size, hfov_deg, capture_hw)
    x = (np.arange(w) + 0.5 - w / 2) / f_x
    y = -(np.arange(h) + 0.5 - h / 2) / f_y
    rays = np.stack(np.broadcast_arrays(x[None, :], y[:, None], np.ones((h, w))), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    rays.flags.writeable = False
    return rays


def sphere_mask(
    center: np.ndarray,
    radius: float,
    eye: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    hfov_deg: float,
    size: tuple[int, int],
    capture_hw: tuple[int, int] = rotation.DEFAULT_CAPTURE_HW,
) -> np.ndarray:
    """Pixels whose camera ray hits a world-space sphere (exact ray/sphere test).

    Args:
        center: ``float[3]`` sphere centre (Source units).
        radius: Sphere radius in Source units.
        eye: ``float[3]`` eye position (X/Y/Z plus view offset).
        yaw_deg: Source yaw of the camera.
        pitch_deg: Source pitch of the camera (positive looks down).
        hfov_deg: Horizontal field of view.
        size: Image ``(H, W)``.
        capture_hw: ``(H, W)`` of the capture, which sets the vertical FOV.

    Returns:
        ``bool[H, W]``. All True if the camera is inside the sphere; all False if it is entirely behind the camera
        or outside the frustum.
    """
    c = _camera_space(center, eye, yaw_deg, pitch_deg)
    outside = float(c @ c) - radius * radius
    if outside < 0:
        return np.ones(size, dtype=bool)
    b = _pixel_rays((int(size[0]), int(size[1])), float(hfov_deg), tuple(capture_hw)) @ c
    return (b > 0) & (b * b >= outside)


def eye_position(
    origin: Sequence[float], duck_amount: float, cfg: DictConfig, view_offset: float | None = None
) -> np.ndarray:
    """Eye position of a player: the pawn origin plus the view offset.

    Args:
        origin: ``(X, Y, Z)`` of the pawn.
        duck_amount: 0 standing to 1 fully crouched (NaN is treated as standing).
        cfg: A validated config (``eval.h2.eye_height_*``).
        view_offset: Eye height above the origin if the demo provides it; otherwise it is derived from
            ``duck_amount``.

    Returns:
        ``float[3]``.
    """
    h = cfg.eval.h2
    if view_offset is None or not np.isfinite(view_offset):
        duck = 0.0 if not np.isfinite(duck_amount) else min(max(float(duck_amount), 0.0), 1.0)
        view_offset = float(h.eye_height_stand) + (float(h.eye_height_crouch) - float(h.eye_height_stand)) * duck
    eye = np.asarray(origin, dtype=np.float64).copy()
    eye[2] += view_offset
    return eye


def active_smokes(events: pd.DataFrame, tick: int) -> pd.DataFrame:
    """Smokes in the air at ``tick``: ``detonate_tick <= tick < expire_tick``."""
    return events[(events["detonate_tick"] <= tick) & (tick < events["expire_tick"])]


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not mask.any():
        return mask
    kernel = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _smokes_in_los(
    tick: int, events: pd.DataFrame, eye: np.ndarray, yaw_deg: float, pitch_deg: float, cfg: DictConfig
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Centres of the active smokes that pass and that fail the line-of-sight filter."""
    forward = view_basis(yaw_deg, pitch_deg)[0]
    eye = np.asarray(eye, dtype=np.float64)
    kept, dropped = [], []
    for center in active_smokes(events, tick)[["x", "y", "z"]].to_numpy(dtype=np.float64):
        d = center - eye
        along = float(d @ forward)
        # Distance from the smoke centre to the line of sight: to the ray in front of the eye, to the eye behind it.
        off_axis = float(np.linalg.norm(d - along * forward)) if along > 0 else float(np.linalg.norm(d))
        (kept if off_axis <= float(cfg.eval.h2.max_los_units) else dropped).append(center)
    return kept, dropped


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
        events: Smoke table (``cs2wm.demo.SMOKE_COLUMNS``: detonate/expire tick, x/y/z, thrower).
        eye: ``float[3]`` eye position.
        yaw_deg: Source yaw of the camera.
        pitch_deg: Source pitch of the camera.
        cfg: A validated cs2wm config (``eval.h2.*``).
        size: Image ``(H, W)``.

    Returns:
        ``bool[H, W]`` mask of pixels inside an active smoke, dilated.
    """
    h = cfg.eval.h2
    hw = (int(size[0]), int(size[1]))
    capture = tuple(int(v) for v in cfg.resolution.capture)
    mask = np.zeros(hw, dtype=bool)
    for center in _smokes_in_los(tick, events, eye, yaw_deg, pitch_deg, cfg)[0]:
        mask |= sphere_mask(
            center, float(h.smoke_radius_units), eye, yaw_deg, pitch_deg, float(cfg.eval.h1.hfov_deg), hw, capture
        )
    return _dilate(mask, int(h.mask_dilate_px))


def los_dropped_count(
    tick: int,
    events: pd.DataFrame,
    eye: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    cfg: DictConfig,
    size: Sequence[int] = (150, 280),
) -> int:
    """Number of active smokes that are in the frustum but left out of the mask by the line-of-sight filter.

    Those pixels end up in the static region (if no wall hides the smoke, they would be smoke), so this counts the
    possible contamination of the static region.
    """
    h = cfg.eval.h2
    hw = (int(size[0]), int(size[1]))
    capture = tuple(int(v) for v in cfg.resolution.capture)
    return sum(
        bool(
            sphere_mask(
                c, float(h.smoke_radius_units), eye, yaw_deg, pitch_deg, float(cfg.eval.h1.hfov_deg), hw, capture
            ).any()
        )
        for c in _smokes_in_los(tick, events, eye, yaw_deg, pitch_deg, cfg)[1]
    )


def viewmodel_mask(cfg: DictConfig, size: Sequence[int] = (150, 280)) -> np.ndarray:
    """Boolean mask of the viewmodel box (``eval.h2.viewmodel_box``, fractions ``x0, y0, x1, y1`` of W/H)."""
    x0, y0, x1, y1 = (float(v) for v in cfg.eval.h2.viewmodel_box)
    h, w = int(size[0]), int(size[1])
    mask = np.zeros((h, w), dtype=bool)
    mask[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)] = True
    return mask


def enemy_in_view(
    enemies: np.ndarray,
    eye: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    cfg: DictConfig,
    size: Sequence[int] = (150, 280),
) -> bool:
    """Whether any enemy is inside the view frustum within ``eval.h2.enemy_max_dist_units``.

    Args:
        enemies: ``float[k, 3]`` pawn origins of the other living players.
        eye: ``float[3]`` eye position of the camera.
        yaw_deg: Source yaw of the camera.
        pitch_deg: Source pitch of the camera.
        cfg: A validated config.
        size: Image ``(H, W)``.

    Returns:
        True if the origin, middle or head of any enemy projects inside the frame (plus margin).
    """
    h = cfg.eval.h2
    hw = (int(size[0]), int(size[1]))
    capture = tuple(int(v) for v in cfg.resolution.capture)
    margin, max_dist = float(h.enemy_margin_px), float(h.enemy_max_dist_units)
    eye = np.asarray(eye, dtype=np.float64)
    for origin in np.asarray(enemies, dtype=np.float64).reshape(-1, 3):
        if np.linalg.norm(origin - eye) > max_dist:
            continue
        for frac in (0.0, 0.5, 1.0):
            point = origin + np.array([0.0, 0.0, frac * PLAYER_HEIGHT_UNITS])
            proj = project_point(point, eye, yaw_deg, pitch_deg, float(cfg.eval.h1.hfov_deg), hw, capture)
            if proj is not None and -margin <= proj[0] <= hw[1] + margin and -margin <= proj[1] <= hw[0] + margin:
                return True
    return False


def static_mask_for_frame(
    smoke_mask: np.ndarray,
    enemies: np.ndarray,
    eye: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    cfg: DictConfig,
) -> np.ndarray | None:
    """Static-geometry mask of one frame.

    Args:
        smoke_mask: ``bool[H, W]`` smoke mask of the frame.
        enemies: ``float[k, 3]`` origins of the other living players at the frame's tick.
        eye: ``float[3]`` eye position.
        yaw_deg: Source yaw of the camera.
        pitch_deg: Source pitch of the camera.
        cfg: A validated config.

    Returns:
        ``bool[H, W]`` complement of the smoke mask minus the viewmodel box, or None if an enemy is in view (the frame
        does not qualify).
    """
    if enemy_in_view(enemies, eye, yaw_deg, pitch_deg, cfg, smoke_mask.shape):
        return None
    return ~smoke_mask & ~viewmodel_mask(cfg, smoke_mask.shape)
