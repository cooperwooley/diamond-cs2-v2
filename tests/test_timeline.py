"""Tests for the frame/tick helpers in cs2wm.timeline."""

import numpy as np
import pytest

from cs2wm.timeline import EpisodeSpec, frame_to_tick, make_episode_id, tick_to_frame


@pytest.fixture
def ep() -> EpisodeSpec:
    return EpisodeSpec(
        episode_id="dm-d1-s000-c00",
        demo_id="d1",
        kind="dm",
        player_name="p",
        user_id=1,
        steamid=0,
        start_tick=1000,
        n_frames=10,
        ticks_per_frame=4,
        segment_idx=0,
        chunk_idx=0,
    )


def test_frame_tick_window_end(ep: EpisodeSpec) -> None:
    assert ep.frame_tick(0) == 1000
    assert ep.frame_tick(10) == 1040
    assert ep.window(0) == (1000, 1004)
    assert ep.window(9) == (1036, 1040)
    assert ep.end_tick == 1040


def test_index_bounds(ep: EpisodeSpec) -> None:
    for bad in (-1, 11):
        with pytest.raises(IndexError):
            ep.frame_tick(bad)
    for bad in (-1, 10):
        with pytest.raises(IndexError):
            ep.window(bad)


def test_frame_to_tick_vectorised() -> None:
    out = frame_to_tick(np.array([0, 1, 5]), 1000, 4)
    np.testing.assert_array_equal(out, [1000, 1004, 1020])
    assert out.dtype == np.int64


def test_tick_to_frame_boundaries() -> None:
    t0, tpf = 1000, 4
    assert int(tick_to_frame(t0, t0, tpf)) == -1  # T0 itself is before the first window
    assert int(tick_to_frame(t0 + 1, t0, tpf)) == 0
    assert int(tick_to_frame(t0 + 4, t0, tpf)) == 0  # T_1 belongs to frame 0
    assert int(tick_to_frame(t0 + 5, t0, tpf)) == 1
    assert int(tick_to_frame(t0 - 3, t0, tpf)) == -1
    assert int(tick_to_frame(t0 - 4, t0, tpf)) == -2


def test_tick_to_frame_vectorised() -> None:
    out = tick_to_frame(np.array([1000, 1001, 1004, 1005, 1008]), 1000, 4)
    np.testing.assert_array_equal(out, [-1, 0, 0, 1, 1])
    assert out.dtype == np.int64


def test_tick_to_frame_inverts_frame_to_tick() -> None:
    k = np.arange(0, 500)
    for t0, tpf in ((0, 4), (1234567, 4), (17, 2)):
        np.testing.assert_array_equal(tick_to_frame(frame_to_tick(k, t0, tpf) + 1, t0, tpf), k)
        np.testing.assert_array_equal(tick_to_frame(frame_to_tick(k + 1, t0, tpf), t0, tpf), k)


def test_make_episode_id() -> None:
    assert make_episode_id("dm", "abc", 3, 7) == "dm-abc-s003-c07"
    assert make_episode_id("smoke", "x", 12, 0) == "smoke-x-s012-c00"
