"""Render-to-demo lag in ticks, from rendered frames and the spectated player's yaw (#88).

Frame ``k`` of a segment is meant to show the demo state at ``T_k = T0 + ticks_per_frame * k``. HLAE can show a
different tick (the offset moved with the Nov 11 CS2 update), which ``render.tick_offset`` compensates. This module
measures what is left, with the method of #73:

* the camera shift between consecutive frames comes from phase correlation on the upper 60% of the frame (the lower part
  holds the viewmodel and HUD, which do not move with the camera);
* it is correlated with the spectated player's d-yaw over the ticks ``(T_k + L, T_k + ticks_per_frame + L]`` for every
  lag ``L`` in ``-16..+16``; pairs with ``|d-yaw| >= 15`` degrees (respawns, spectator cuts) and pairs whose image shift is over ``MAX_SHIFT_FRAC`` of the frame
  width (the same cuts seen from the image side) are left out;
* the lag with the highest correlation is the answer, refined to sub-tick precision by a parabola through the peak.

Sign conventions: ``L > 0`` means the frame shows the state ``L`` ticks **after** its nominal tick (the render is
ahead of the labels); ``L < 0`` means it shows an earlier state, which ``render.tick_offset = -L`` compensates. In CS2
demos +yaw is a turn to the left, so the scene content moves right (``dx > 0``) and r is positive at the true lag. The
best lag is the maximum of r, never of ``|r|``. For pitch, +pitch is looking down, which moves the content up
(``dy < 0``), so pitch is correlated as ``-dy``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cs2wm.eval.rotation import estimate_shift_phasecorr

LAGS = tuple(range(-16, 17))
MAX_ABS_DELTA_DEG = 15.0  # pairs with a larger demo d-angle are spectator glitches, not camera motion
# Pairs whose image shift is over this fraction of the frame width are phase-correlation wraps (a spectator cut returns
# a shift in the hundreds of pixels), not camera motion; they would mask the real lag.
MAX_SHIFT_FRAC = 0.25
UPPER_FRACTION = 0.6  # the part of the frame (from the top) that the shift is measured on
_MIN_PAIRS = 3


@dataclass(frozen=True)
class Segment:
    """Frame-to-frame camera shifts of one stretch of consecutive frames of one spectated player.

    Attributes:
        dx: ``float[m - 1]`` horizontal content shift in pixels from frame ``k`` to ``k + 1`` (+ = right).
        dy: ``float[m - 1]`` vertical content shift in pixels (+ = down).
        start_tick: Demo tick ``T0`` of frame 0.
        ticks_per_frame: Demo ticks between two frames (``timing.tickrate // timing.fps``).
        yaw: Demo yaw in degrees of the spectated player, indexed by tick.
        pitch: Demo pitch in degrees indexed by tick, or None to skip the pitch estimate.
        max_shift_px: Pairs whose |dx| or |dy| exceeds this are left out (``MAX_SHIFT_FRAC`` of the frame width when
            made by ``from_frames``; no limit by default).
    """

    dx: np.ndarray
    dy: np.ndarray
    start_tick: int
    ticks_per_frame: int
    yaw: pd.Series
    pitch: pd.Series | None = None
    max_shift_px: float = math.inf

    @classmethod
    def from_frames(
        cls,
        frames_bgr: np.ndarray,
        start_tick: int,
        ticks_per_frame: int,
        yaw: pd.Series,
        pitch: pd.Series | None = None,
    ) -> Segment:
        """Measure the shifts of ``frames_bgr`` (``uint8[m, H, W, 3]``) and keep only those, not the frames."""
        dx, dy = frame_shifts(frames_bgr)
        return cls(dx, dy, int(start_tick), int(ticks_per_frame), yaw, pitch, MAX_SHIFT_FRAC * frames_bgr.shape[2])


@dataclass(frozen=True)
class LagResult:
    """The measured render-to-demo lag.

    Attributes:
        lag: Integer lag in ticks with the highest r.
        lag_subtick: ``lag`` refined by a parabola through the peak and its two neighbours.
        r: Peak correlation (at ``lag``).
        n_pairs: Frame pairs that entered the correlation at ``lag``.
        r_by_lag: Correlation per lag in ``-16..+16`` (NaN where there was too little data).
    """

    lag: int
    lag_subtick: float
    r: float
    n_pairs: int
    r_by_lag: dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        """One line for logs."""
        return f"lag {self.lag_subtick:+.2f} ticks (best integer {self.lag:+d}, r {self.r:.2f}, {self.n_pairs} pairs)"


def frame_shifts(frames_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Phase-correlation shifts between consecutive frames, on the upper 60% of each frame.

    Args:
        frames_bgr: ``uint8[m, H, W, 3]`` BGR frames in order.

    Returns:
        ``(dx, dy)``, each ``float[m - 1]``: the content shift in pixels from frame ``k`` to ``k + 1``.
    """
    rows = max(1, int(frames_bgr.shape[1] * UPPER_FRACTION))
    top = frames_bgr[:, :rows]
    shifts = [estimate_shift_phasecorr(top[k], top[k + 1]) for k in range(len(top) - 1)]
    return np.array([s.dx for s in shifts]), np.array([s.dy for s in shifts])


def player_angles(ticks: pd.DataFrame, user_id: int) -> tuple[pd.Series, pd.Series]:
    """Yaw and pitch series, indexed by tick, of one player from a demo tick table.

    Args:
        ticks: Tick table with ``tick``, ``user_id``, ``yaw`` and ``pitch`` columns.
        user_id: Demo ``user_id`` of the spectated player.

    Returns:
        ``(yaw, pitch)`` in degrees.
    """
    rows = ticks[ticks["user_id"] == user_id].drop_duplicates("tick").set_index("tick").sort_index()
    return rows["yaw"].astype(float), rows["pitch"].astype(float)


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap degrees to ``[-180, 180)``."""
    return (angle + 180.0) % 360.0 - 180.0


def _pairs(seg: Segment, lag: int, axis: str) -> tuple[np.ndarray, np.ndarray]:
    """``(image shift, demo d-angle)`` of every usable frame pair of a segment at one lag."""
    series = seg.yaw if axis == "yaw" else seg.pitch
    k = np.arange(len(seg.dx))
    a = seg.start_tick + seg.ticks_per_frame * k + lag
    b = a + seg.ticks_per_frame
    first, second = series.reindex(a).to_numpy(float), series.reindex(b).to_numpy(float)
    delta = second - first
    if axis == "yaw":
        delta = _wrap(delta)
    shift = seg.dx if axis == "yaw" else -seg.dy
    ok = (
        np.isfinite(delta)
        & np.isfinite(shift)
        & (np.abs(delta) < MAX_ABS_DELTA_DEG)
        & (np.abs(shift) <= seg.max_shift_px)
    )
    return shift[ok], delta[ok]


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < _MIN_PAIRS or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _peak(r_by_lag: dict[int, float]) -> tuple[int, float]:
    """Integer lag of the highest r and the parabolic sub-tick estimate (the integer lag at an edge or a gap)."""
    best = max(r_by_lag, key=r_by_lag.__getitem__)
    lo, hi = r_by_lag.get(best - 1, math.nan), r_by_lag.get(best + 1, math.nan)
    denom = lo - 2 * r_by_lag[best] + hi
    if not (math.isfinite(denom) and denom < 0):
        return best, float(best)
    return best, best + 0.5 * (lo - hi) / denom


def measure_lag(segments: Iterable[Segment], axis: str = "yaw") -> LagResult | None:
    """Pool the frame pairs of all segments and find the lag at which image shift and demo d-angle agree best.

    Args:
        segments: Segments to pool (one per spectated stretch; pairs never span two segments).
        axis: ``"yaw"`` or ``"pitch"`` (needs ``Segment.pitch``).

    Returns:
        The result, or None if there is no verdict: no pair at any lag, or no turning (the demo angle or the image
        shift is constant), so r is undefined. Whether r and the pair count are good enough is the caller's call.

    Raises:
        ValueError: If ``axis`` is not ``"yaw"`` or ``"pitch"``, or ``"pitch"`` is asked for a segment without pitch.
    """
    if axis not in ("yaw", "pitch"):
        raise ValueError(f"axis must be 'yaw' or 'pitch', got {axis!r}")
    segs: Sequence[Segment] = list(segments)
    if axis == "pitch" and any(s.pitch is None for s in segs):
        raise ValueError("pitch lag needs Segment.pitch on every segment")
    r_by_lag: dict[int, float] = {}
    counts: dict[int, int] = {}
    for lag in LAGS:
        parts = [_pairs(s, lag, axis) for s in segs]
        x = np.concatenate([p[0] for p in parts]) if parts else np.empty(0)
        y = np.concatenate([p[1] for p in parts]) if parts else np.empty(0)
        r_by_lag[lag], counts[lag] = _corr(x, y), len(x)
    valid = {lag: r for lag, r in r_by_lag.items() if math.isfinite(r)}
    if not valid:
        return None
    best, subtick = _peak(valid)
    return LagResult(best, subtick, r_by_lag[best], counts[best], r_by_lag)
