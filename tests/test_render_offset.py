"""Tests for cs2wm.eval.render_offset (#88): the render-to-demo lag from camera rotation vs demo yaw."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cs2wm.eval.render_offset import LAGS, Segment, frame_shifts, measure_lag, player_angles

H, W = 150, 280
TPF = 4  # ticks per frame
PX_PER_DEG = 2.0
N_FRAMES = 120
T0 = 1000
MARGIN = 120


def _texture() -> np.ndarray:
    """Random grey texture, big enough to crop a frame at any offset the tests use."""
    rng = np.random.default_rng(7)
    tex = rng.integers(0, 256, (H + 2 * MARGIN, W + 2 * MARGIN), dtype=np.uint8)
    return np.repeat(tex[:, :, None], 3, axis=2)


def _angle(ticks: np.ndarray, phase: float = 0.0) -> np.ndarray:
    """A smooth, turning angle in degrees: at most ~2 deg per tick, so under 15 deg per frame."""
    return 20 * np.sin(2 * np.pi * (ticks + phase) / 150) + 10 * np.sin(2 * np.pi * ticks / 53)


def synthetic_render(
    lag: int, start_tick: int = T0, wrap: bool = False, with_pitch: bool = False
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    """Frames whose frame k shows the demo state at tick ``T_k + lag``, and the demo's per-tick yaw and pitch."""
    ticks = np.arange(start_tick - 40, start_tick + TPF * N_FRAMES + 40)
    yaw_deg = _angle(ticks)
    pitch_deg = _angle(ticks, phase=31) / 4
    tex = _texture()
    frames = np.empty((N_FRAMES, H, W, 3), dtype=np.uint8)
    for k in range(N_FRAMES):
        state = start_tick + TPF * k + lag - ticks[0]
        # content moves right when yaw rises, and up when pitch (looking down) rises
        x0 = MARGIN - round(PX_PER_DEG * yaw_deg[state])
        y0 = MARGIN + (round(PX_PER_DEG * pitch_deg[state]) if with_pitch else 0)
        frames[k] = tex[y0 : y0 + H, x0 : x0 + W]
    shown = yaw_deg + 175 if wrap else yaw_deg  # the wrap case crosses +-180
    yaw = pd.Series(((shown + 180) % 360) - 180, index=ticks)
    return frames, yaw, pd.Series(pitch_deg, index=ticks)


def _segment(lag: int, start_tick: int = T0, wrap: bool = False, with_pitch: bool = False) -> Segment:
    frames, yaw, pitch = synthetic_render(lag, start_tick, wrap, with_pitch)
    return Segment.from_frames(frames, start_tick, TPF, yaw, pitch)


def _with(seg: Segment, **changes) -> Segment:
    fields = {"dx": seg.dx, "dy": seg.dy, "start_tick": seg.start_tick, "ticks_per_frame": seg.ticks_per_frame}
    return Segment(**{**fields, "yaw": seg.yaw, "pitch": seg.pitch, **changes})


@pytest.mark.parametrize("lag", [-5, 0, 3, 7])
def test_recovers_the_lag_of_synthetic_frames(lag: int) -> None:
    res = measure_lag([_segment(lag)])
    assert res is not None
    assert res.lag == lag
    assert abs(res.lag_subtick - lag) < 0.5
    assert res.r > 0.9 and res.n_pairs > 100
    assert set(res.r_by_lag) == set(LAGS)
    assert res.r_by_lag[lag] == res.r


def test_r_is_positive_at_the_true_lag_and_the_sign_is_not_flipped() -> None:
    res = measure_lag([_segment(3)])
    assert res is not None and res.r_by_lag[3] > 0.9
    # a render that is ahead (+3) must not be mistaken for one that is behind (-3)
    assert res.r_by_lag[-3] < res.r_by_lag[3]


def test_pools_segments_with_different_start_ticks() -> None:
    res = measure_lag([_segment(3, start_tick=1000), _segment(3, start_tick=5000)])
    one = measure_lag([_segment(3)])
    assert res is not None and one is not None
    assert res.lag == 3 and res.n_pairs == pytest.approx(2 * one.n_pairs, abs=4)


def test_yaw_wraps_around_180() -> None:
    res = measure_lag([_segment(2, wrap=True)])
    assert res is not None and res.lag == 2 and res.r > 0.9


def test_pitch_lag_is_measured_with_the_same_sign_convention() -> None:
    res = measure_lag([_segment(3, with_pitch=True)], axis="pitch")
    assert res is not None and res.lag == 3 and res.r > 0.8


def test_large_jumps_are_left_out() -> None:
    seg = _segment(3)
    yaw = seg.yaw.copy()
    yaw.loc[yaw.index >= T0 + 200] += 40.0  # a respawn-like jump: the pairs across it are dropped
    res = measure_lag([_with(seg, yaw=yaw)])
    assert res is not None and res.lag == 3 and res.r > 0.9


def test_no_turning_has_no_verdict() -> None:
    seg = _segment(3)
    assert measure_lag([_with(seg, yaw=pd.Series(90.0, index=seg.yaw.index))]) is None


def test_no_pairs_has_no_verdict() -> None:
    assert measure_lag([_with(_segment(3), start_tick=10_000_000)]) is None  # ticks not in the series
    assert measure_lag([]) is None


def test_unknown_axis_and_missing_pitch_raise() -> None:
    seg = _segment(0)
    with pytest.raises(ValueError, match="axis"):
        measure_lag([seg], axis="roll")
    with pytest.raises(ValueError, match="pitch"):
        measure_lag([_with(seg, pitch=None)], axis="pitch")


def test_frame_shifts_content_moving_right_is_positive() -> None:
    tex = _texture()
    a, b = tex[MARGIN : MARGIN + H, MARGIN : MARGIN + W], tex[MARGIN : MARGIN + H, MARGIN - 6 : MARGIN - 6 + W]
    dx, dy = frame_shifts(np.stack([a, b]))
    assert dx.shape == (1,) and dx[0] == pytest.approx(6.0, abs=0.3) and abs(dy[0]) < 0.3


def test_player_angles_picks_one_player_and_drops_duplicate_ticks() -> None:
    table = pd.DataFrame(
        {
            "tick": [1, 1, 2, 3, 1, 2],
            "user_id": [5, 5, 5, 5, 9, 9],
            "yaw": [10.0, 10.0, 11.0, 12.0, -1.0, -2.0],
            "pitch": [0.0, 0.0, 1.0, 2.0, 7.0, 8.0],
        }
    )
    yaw, pitch = player_angles(table, 5)
    assert list(yaw.index) == [1, 2, 3] and list(yaw) == [10.0, 11.0, 12.0] and list(pitch) == [0.0, 1.0, 2.0]
