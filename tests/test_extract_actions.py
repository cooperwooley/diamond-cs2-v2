"""Tests for the pure action extractor (#11) on synthetic per-tick tables."""

from __future__ import annotations

import json
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from cs2wm.actions import ActionSpec
from cs2wm.config import load_config
from cs2wm.pipeline.extract_actions import EpisodeQA, extract_episode
from cs2wm.timeline import EpisodeSpec

T0 = 1000
TPF = 4
N = 80  # >= episodes.min_frames (64)
RIFLE, PISTOL = "AK-47", "Glock-18"

CFG = load_config()
SPEC = ActionSpec.from_config(CFG)
KEY = {name: i for i, name in enumerate(SPEC.keys)}
LMB, RMB = SPEC.idx_lmb, SPEC.idx_rmb


def make_episode(n: int = N, start: int = T0) -> EpisodeSpec:
    return EpisodeSpec(
        episode_id="dm-demo-s000-c00",
        demo_id="demo",
        kind="dm",
        player_name="bot",
        user_id=1,
        steamid=0,
        start_tick=start,
        n_frames=n,
        ticks_per_frame=TPF,
        segment_idx=0,
        chunk_idx=0,
    )


def make_ticks(
    ep: EpisodeSpec,
    *,
    x: Callable[[np.ndarray], np.ndarray] | float = 0.0,
    y: Callable[[np.ndarray], np.ndarray] | float = 0.0,
    z: Callable[[np.ndarray], np.ndarray] | float = 0.0,
    yaw: Callable[[np.ndarray], np.ndarray] | float = 0.0,
    pitch: Callable[[np.ndarray], np.ndarray] | float = 0.0,
    airborne: Callable[[np.ndarray], np.ndarray] | bool = False,
    duck: Callable[[np.ndarray], np.ndarray] | float = 0.0,
    weapon: Callable[[np.ndarray], list] | str | None = RIFLE,
    pad: int = 8,
    **extra: Callable[[np.ndarray], np.ndarray],
) -> pd.DataFrame:
    """One row per tick from ``T0 - pad`` to ``T_n + pad``; every argument is a constant or f(tick)."""
    t = np.arange(ep.start_tick - pad, ep.end_tick + pad + 1)

    def val(v, dtype=float):
        out = v(t) if callable(v) else np.full(t.shape, v)
        return np.asarray(out, dtype=dtype)

    wp = weapon(t) if callable(weapon) else [weapon] * len(t)
    df = pd.DataFrame(
        {
            "tick": t,
            "X": val(x),
            "Y": val(y),
            "Z": val(z),
            "yaw": val(yaw),
            "pitch": val(pitch),
            "is_airborne": val(airborne, bool),
            "duck_amount": val(duck),
            "active_weapon_name": pd.Series(wp, dtype=object),
        }
    )
    for name, f in extra.items():
        df[name] = f(t)
    return df


def ev(*ticks: int, **cols) -> pd.DataFrame:
    return pd.DataFrame({"tick": list(ticks), **{k: v for k, v in cols.items()}})


def run(ticks: pd.DataFrame, events=None, ep: EpisodeSpec | None = None, cfg=CFG):
    return extract_episode(ticks, events or {}, ep or make_episode(), cfg)


def x_bin(a: np.ndarray) -> np.ndarray:
    return a[:, SPEC.mouse_x_slice].argmax(axis=1)


def y_bin(a: np.ndarray) -> np.ndarray:
    return a[:, SPEC.mouse_y_slice].argmax(axis=1)


X0 = SPEC.mouse_x_units.index(0)
Y0 = SPEC.mouse_y_units.index(0)


def keys_of(a: np.ndarray, k: int) -> set[str]:
    return {name for name, i in KEY.items() if a[k, i] == 1}


def vel(vx: float, vy: float) -> dict:
    """Constant horizontal velocity in units/s as position functions of tick."""
    return {"x": lambda t: vx * (t - T0) / 64.0, "y": lambda t: vy * (t - T0) / 64.0}


# --- 1. standing still ------------------------------------------------------------------------------------------


def test_standing_still_is_all_zero_and_passes() -> None:
    a, qa = run(make_ticks(make_episode()))
    assert a.shape == (N, 51)
    assert a[:, : SPEC.idx_rmb + 1].sum() == 0
    assert (x_bin(a) == X0).all() and (y_bin(a) == Y0).all()
    assert qa.passed and qa.reasons == []
    assert qa.stats["n_frames"] == N
    assert qa.stats["missing_tick_frac"] == 0.0
    assert qa.stats["fire_recall"] == 1.0
    assert qa.stats["rate_mouse_x_nonzero"] == 0.0


# --- 2. WASD ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("yaw", "forward_dir", "right_dir"),
    [
        (90.0, (0, 1), (1, 0)),
        (0.0, (1, 0), (0, -1)),
        (-135.0, (math.cos(math.radians(-135)), math.sin(math.radians(-135))), (-math.sqrt(0.5), math.sqrt(0.5))),
    ],
)
def test_wasd_projection(yaw: float, forward_dir: tuple, right_dir: tuple) -> None:
    ep = make_episode()
    # Right is (sin, -cos): for yaw=-135 that is (-sqrt(.5), +sqrt(.5)).
    cases = {
        "w": (forward_dir, 1),
        "s": (forward_dir, -1),
        "d": (right_dir, 1),
        "a": (right_dir, -1),
    }
    for key, ((dx, dy), sign) in cases.items():
        a, qa = run(make_ticks(ep, yaw=yaw, **vel(sign * 250 * dx, sign * 250 * dy)))
        assert all(keys_of(a, k) == {key} for k in range(N)), (yaw, key)
        assert qa.passed, qa.reasons


def test_wasd_diagonal_gives_two_keys() -> None:
    a, _ = run(make_ticks(make_episode(), yaw=90.0, **vel(180, 180)))
    assert keys_of(a, 5) == {"w", "d"}


def test_speed_below_deadzone_gives_no_key() -> None:
    a, _ = run(make_ticks(make_episode(), yaw=90.0, **vel(0, 25)))
    assert keys_of(a, 5) == set()
    a, _ = run(make_ticks(make_episode(), yaw=90.0, **vel(0, 35)))
    assert keys_of(a, 5) == {"w"}


# --- 3-5. mouse -------------------------------------------------------------------------------------------------


def test_turn_right_and_left_bins() -> None:
    ep = make_episode()
    a, qa = run(make_ticks(ep, yaw=lambda t: -0.1375 * (t - T0)))
    assert {SPEC.mouse_x_units[b] for b in x_bin(a)} == {10}
    assert (y_bin(a) == Y0).all() and qa.passed
    a, _ = run(make_ticks(ep, yaw=lambda t: 0.1375 * (t - T0)))
    assert {SPEC.mouse_x_units[b] for b in x_bin(a)} == {-10}


def test_yaw_wrap_across_180() -> None:
    ep = make_episode(n=N)

    def yaw(t):  # 179 at T_10, +2 degrees per frame wraps to -179 at T_11, then holds
        raw = 179.0 + np.clip((t - (T0 + 40)) / 4.0, 0, 1) * 2.0
        return (raw + 180) % 360 - 180

    a, qa = run(make_ticks(ep, yaw=yaw))
    assert SPEC.mouse_x_units[x_bin(a)[10]] == -30  # -36.4 units -> nearest centre -30
    assert qa.stats["mouse_saturation_frac"] == 0.0
    assert qa.stats["max_abs_dyaw_deg"] == pytest.approx(2.0)
    assert qa.passed


def test_pitch_down_bin() -> None:
    a, _ = run(make_ticks(make_episode(), pitch=lambda t: 0.055 * (t - T0)))  # 0.22 deg/frame
    assert {SPEC.mouse_y_units[b] for b in y_bin(a)} == {4}
    a, _ = run(make_ticks(make_episode(), pitch=lambda t: -0.055 * (t - T0)))
    assert {SPEC.mouse_y_units[b] for b in y_bin(a)} == {-4}


# --- 6. event alignment -----------------------------------------------------------------------------------------


def lmb_frames(a: np.ndarray) -> list[int]:
    return np.flatnonzero(a[:, LMB]).tolist()


def test_event_window_boundaries() -> None:
    ep = make_episode()
    t1 = ep.frame_tick(1)
    ticks = make_ticks(ep)
    a, _ = run(ticks, {"weapon_fire": ev(t1)})
    assert lmb_frames(a) == [0]
    a, _ = run(ticks, {"weapon_fire": ev(t1 + 1)})
    assert lmb_frames(a) == [1]
    a, qa = run(ticks, {"weapon_fire": ev(T0)})
    assert lmb_frames(a) == []
    assert qa.stats["n_fire_events"] == 0 and qa.stats["fire_recall"] == 1.0
    a, qa = run(ticks, {"weapon_fire": ev(ep.end_tick)})  # last tick belongs to the last frame
    assert lmb_frames(a) == [N - 1]
    assert qa.stats["n_fire_events"] == 1
    a, qa = run(ticks, {"weapon_fire": ev(ep.end_tick + 1)})
    assert lmb_frames(a) == [] and qa.stats["n_fire_events"] == 0


def test_fire_recall_is_one_by_construction() -> None:
    # Every event in (T0, T_n] maps to a frame in [0, n) and sets LMB, so recall cannot drop below 1 unless the
    # alignment regresses: that is exactly what the metric guards against.
    ep = make_episode()
    shots = list(range(T0 + 1, ep.end_tick + 1, 7))
    _, qa = run(make_ticks(ep), {"weapon_fire": ev(*shots)})
    assert qa.stats["n_fire_events"] == len(shots)
    assert qa.stats["fire_recall"] == 1.0


def test_recall_gate_reports_when_threshold_unreachable() -> None:
    cfg = OmegaConf.create(OmegaConf.to_container(CFG))  # unvalidated copy
    cfg.qa.min_fire_recall = 1.5
    ep = make_episode()
    _, qa = run(make_ticks(ep), {"weapon_fire": ev(T0 + 2)}, cfg=cfg)
    assert not qa.passed
    assert qa.reasons == ["fire_recall 1.000 < 1.5"]


# --- 7. hold-fill -----------------------------------------------------------------------------------------------


def test_hold_fill_bridges_close_shots() -> None:
    ep = make_episode()
    shots = [ep.frame_tick(2) + 1 + 6 * i for i in range(6)]  # 6-tick cadence, frames 2..
    a, _ = run(make_ticks(ep), {"weapon_fire": ev(*shots, weapon=[RIFLE] * 6)})
    first = 2
    last = int(-((T0 - shots[-1]) // TPF) - 1)
    assert lmb_frames(a) == list(range(first, last + 1))


def test_hold_fill_does_not_bridge_wide_gap() -> None:
    ep = make_episode()
    t_a, t_b = ep.frame_tick(2) + 1, ep.frame_tick(2) + 41  # 40 ticks > 0.2 s * 64
    a, _ = run(make_ticks(ep), {"weapon_fire": ev(t_a, t_b, weapon=[RIFLE, RIFLE])})
    assert lmb_frames(a) == [2, 12]


def test_hold_fill_does_not_bridge_different_weapons() -> None:
    ep = make_episode()
    t_a, t_b = ep.frame_tick(2) + 1, ep.frame_tick(2) + 9
    a, _ = run(make_ticks(ep), {"weapon_fire": ev(t_a, t_b, weapon=[RIFLE, PISTOL])})
    assert lmb_frames(a) == [2, 4]


def test_hold_fill_without_weapon_column_and_clipping() -> None:
    ep = make_episode()
    # Shots straddling the start: a pre-episode shot bridges into frame 0 but is not itself counted.
    a, qa = run(make_ticks(ep), {"weapon_fire": ev(T0 - 5, T0 + 3)})
    assert lmb_frames(a) == [0]
    assert qa.stats["n_fire_events"] == 1
    # Straddling the end clips to n - 1.
    a, _ = run(make_ticks(ep), {"weapon_fire": ev(ep.end_tick - 3, ep.end_tick + 5)})
    assert lmb_frames(a) == [N - 1]


def test_hold_fill_pair_entirely_outside_the_episode_marks_nothing() -> None:
    ep = make_episode()
    # Two close shots well before T0 (e.g. the end of a burst of the previous episode) must not wrap to frame 0.
    a, qa = run(make_ticks(ep), {"weapon_fire": ev(T0 - 40, T0 - 39)})
    assert lmb_frames(a) == [] and qa.stats["n_fire_events"] == 0
    a, _ = run(make_ticks(ep), {"weapon_fire": ev(ep.end_tick + 20, ep.end_tick + 21)})
    assert lmb_frames(a) == []


# --- 8. jump ----------------------------------------------------------------------------------------------------


def test_jump_from_event() -> None:
    ep = make_episode()
    a, _ = run(make_ticks(ep), {"player_jump": ev(ep.frame_tick(3) + 2)})
    assert [k for k in range(N) if "space" in keys_of(a, k)] == [3]


def test_jump_from_state_takeoff() -> None:
    ep = make_episode()
    t_up = ep.frame_tick(5)  # on the ground at T_5, airborne after
    a, _ = run(
        make_ticks(
            ep,
            airborne=lambda t: t > t_up,
            z=lambda t: np.where(t > t_up, 2.0 * (t - t_up), 0.0),  # 8 units/frame = 128 u/s > 100
        )
    )
    assert "space" in keys_of(a, 5)
    assert "space" not in keys_of(a, 6)  # already airborne


def test_slow_takeoff_is_not_a_jump_and_ledge_fall_is_not_a_jump() -> None:
    ep = make_episode()
    t_up = ep.frame_tick(5)
    slow = make_ticks(ep, airborne=lambda t: t > t_up, z=lambda t: np.where(t > t_up, 1.0 * (t - t_up), 0.0))
    a, _ = run(slow)
    assert "space" not in keys_of(a, 5)
    fall = make_ticks(ep, airborne=lambda t: t > t_up, z=lambda t: np.where(t > t_up, -3.0 * (t - t_up), 0.0))
    a, _ = run(fall)
    assert not a[:, KEY["space"]].any()


# --- 9. crouch --------------------------------------------------------------------------------------------------


def test_crouch_rising_held_and_release() -> None:
    ep = make_episode()
    t_s, t_e = ep.frame_tick(4), ep.frame_tick(8)

    def duck(t):  # ramp 0 -> 1 over frames 4..8, hold to frame 14, ramp down to 0 by frame 18
        up = np.clip((t - t_s) / (t_e - t_s), 0, 1)
        down = np.clip((t - ep.frame_tick(14)) / (ep.frame_tick(18) - ep.frame_tick(14)), 0, 1)
        return up - down

    a, _ = run(make_ticks(ep, duck=duck))
    ctrl = a[:, KEY["ctrl"]]
    assert ctrl[4:8].all()  # rising
    assert ctrl[8:14].all()  # held fully ducked
    assert not ctrl[14:].any()  # falling (and standing)
    assert not ctrl[:4].any()


# --- 10. weapon switch ------------------------------------------------------------------------------------------


def switch_ticks(ep: EpisodeSpec, new: object, at_frame: int = 3):
    t_sw = ep.frame_tick(at_frame) + 2
    return make_ticks(ep, weapon=lambda t: [RIFLE if v < t_sw else new for v in t])


@pytest.mark.parametrize(("new", "slot"), [(PISTOL, "2"), ("Knife", "3"), ("Smoke Grenade", "3"), ("M4A1", "1")])
def test_weapon_switch_slot(new: str, slot: str) -> None:
    ep = make_episode()
    a, _ = run(switch_ticks(ep, new))
    assert [(k, keys_of(a, k)) for k in range(N) if keys_of(a, k)] == [(3, {slot})]


@pytest.mark.parametrize("new", [None, float("nan")])
def test_missing_weapon_names_give_no_switch(new: object) -> None:
    a, _ = run(switch_ticks(make_episode(), new))
    assert not a[:, [KEY["1"], KEY["2"], KEY["3"]]].any()


# --- 11. shift --------------------------------------------------------------------------------------------------


def test_shift_from_is_walking() -> None:
    ep = make_episode()
    t_w = ep.frame_tick(10)
    a, _ = run(make_ticks(ep, is_walking=lambda t: t > t_w))
    shift = a[:, KEY["shift"]]
    assert not shift[:10].any() and shift[10:].all()


def test_shift_absent_is_zero() -> None:
    a, _ = run(make_ticks(make_episode()))
    assert not a[:, KEY["shift"]].any()


# --- 12. reload / zoom / grenade --------------------------------------------------------------------------------


def test_reload_zoom_grenade_events() -> None:
    ep = make_episode()
    ev_map = {
        "weapon_reload": ev(ep.frame_tick(1) + 1),
        "weapon_zoom": ev(ep.frame_tick(2) + 1),
        "grenade_thrown": ev(ep.frame_tick(3) + 1),
    }
    a, qa = run(make_ticks(ep), ev_map)
    assert np.flatnonzero(a[:, KEY["r"]]).tolist() == [1]
    assert np.flatnonzero(a[:, RMB]).tolist() == [2]
    assert lmb_frames(a) == [3]
    assert qa.stats["n_fire_events"] == 0  # a throw is not a weapon_fire event


def test_empty_and_unknown_events_are_ignored() -> None:
    a, _ = run(make_ticks(make_episode()), {"weapon_fire": ev(), "weapon_zoom": pd.DataFrame({"tick": []}), "x": ev(5)})
    assert not a[:, : SPEC.idx_rmb + 1].any()


# --- 13. QA failures --------------------------------------------------------------------------------------------


def test_qa_missing_ticks_fail() -> None:
    ep = make_episode()
    t = make_ticks(ep)
    # Drop 5% of ticks, but never the sampled ones (T_k), so only the fraction gates.
    drop = (t["tick"] - T0) % TPF != 0
    drop &= (t["tick"] - T0) % 20 == 1
    t = t[~drop]
    _, qa = run(t)
    assert qa.stats["missing_tick_frac"] > 0.01
    assert not qa.passed and any(r.startswith("missing_tick_frac") for r in qa.reasons)


def test_qa_nan_position_fails_but_output_is_finite() -> None:
    ep = make_episode()
    t = make_ticks(ep, **vel(0, 250), yaw=90.0)
    t.loc[t["tick"] == ep.frame_tick(10), "X"] = np.nan
    a, qa = run(t)
    assert np.isfinite(a).all()
    assert qa.stats["nan_frames"] == 2  # frames 9 and 10 both use T_10
    assert not qa.passed and any(r.startswith("nan_frames") for r in qa.reasons)
    assert keys_of(a, 9) == set() and keys_of(a, 10) == set()
    assert keys_of(a, 11) == {"w"}
    assert np.isfinite(list(qa.stats.values())).all()


def test_qa_nan_at_final_sample_hits_last_frame() -> None:
    ep = make_episode()
    t = make_ticks(ep, pad=0)
    t.loc[t["tick"] == ep.end_tick, "yaw"] = np.nan
    a, qa = run(t)
    assert qa.stats["nan_frames"] == 1 and np.isfinite(a).all()


def test_qa_teleport_fails() -> None:
    ep = make_episode()
    t_tp = ep.frame_tick(20)
    a, qa = run(make_ticks(ep, y=lambda t: np.where(t > t_tp, 500.0, 0.0)))
    assert qa.stats["max_disp_per_frame"] == pytest.approx(500.0)
    assert not qa.passed and any(r.startswith("max_disp_per_frame") for r in qa.reasons)
    assert "w" in keys_of(a, 20) or "d" in keys_of(a, 20) or "a" in keys_of(a, 20) or "s" in keys_of(a, 20)


def test_qa_yaw_snap_fails() -> None:
    ep = make_episode()
    t_snap = ep.frame_tick(20)
    a, qa = run(make_ticks(ep, yaw=lambda t: np.where(t > t_snap, 90.0, 0.0)))
    assert qa.stats["max_abs_dyaw_deg"] == pytest.approx(90.0)
    assert any(r.startswith("max_abs_dyaw_deg") for r in qa.reasons)
    assert x_bin(a)[20] == 0 or x_bin(a)[20] == len(SPEC.mouse_x_units) - 1


def test_qa_mouse_saturation_fails() -> None:
    ep = make_episode()
    # Pitch jumps of 20 deg (~364 units) on 3 of 4 frames; the y extreme bin is 200 units (11 deg).
    _, qa = run(make_ticks(ep, pitch=lambda t: np.where(((t - T0) // TPF) % 4 == 0, 0.0, 20.0)))
    assert qa.stats["mouse_saturation_frac"] > 0.02
    assert any(r.startswith("mouse_saturation_frac") for r in qa.reasons)


def test_qa_short_episode_fails() -> None:
    ep = make_episode(n=10)
    _, qa = run(make_ticks(ep), ep=ep)
    assert not qa.passed
    assert qa.reasons == ["n_frames 10.000 < 64"]


# --- 14. nearest-tick fill --------------------------------------------------------------------------------------


def test_nearest_tick_fill_within_limit() -> None:
    ep = make_episode()
    t = make_ticks(ep, **vel(0, 250), yaw=90.0)
    gap = t["tick"].isin([ep.frame_tick(10), ep.frame_tick(10) + 1])  # nearest present is 2 ticks away
    t = t[~gap]
    a, qa = run(t)
    assert qa.stats["nan_frames"] == 0
    assert all(keys_of(a, k) == {"w"} for k in range(N))


def test_nearest_tick_fill_beyond_limit_is_nan() -> None:
    ep = make_episode()
    t = make_ticks(ep)
    t = t[~t["tick"].between(ep.frame_tick(10) - 3, ep.frame_tick(10) + 3)]  # 4 ticks to nearest
    a, qa = run(t)
    assert qa.stats["nan_frames"] == 2
    assert not qa.passed and np.isfinite(a).all()


def test_no_ticks_at_all_gives_all_nan_frames() -> None:
    ep = make_episode()
    t = make_ticks(ep).iloc[0:0]
    a, qa = run(t)
    assert qa.stats["nan_frames"] == N and np.isfinite(a).all() and not qa.passed


def test_duplicate_ticks_keep_first_counted_and_fail_qa() -> None:
    ep = make_episode()
    t = make_ticks(ep, **vel(0, 250), yaw=90.0)
    bad = t[t["tick"].isin([ep.frame_tick(3), ep.frame_tick(3) + 1])].copy()
    bad["Y"] = 1e6  # later duplicates must be ignored
    a, qa = run(pd.concat([t, bad], ignore_index=True))
    assert qa.stats["dup_ticks"] == 2
    assert keys_of(a, 3) == {"w"}
    assert not qa.passed and "dup_ticks 2 > 0" in qa.reasons


def test_two_players_merged_fail_qa() -> None:
    ep = make_episode()
    p1 = make_ticks(ep, **vel(0, 250), yaw=90.0)
    v2 = vel(0, 240)
    p2 = make_ticks(ep, x=lambda t: 500.0 + v2["x"](t), y=lambda t: 500.0 + v2["y"](t), yaw=90.0)
    _, qa = run(pd.concat([p1, p2], ignore_index=True))
    assert qa.passed is False
    assert qa.stats["dup_ticks"] > 0
    assert any(r.startswith("dup_ticks") for r in qa.reasons)


def test_rows_outside_episode_are_ignored_and_object_airborne_ok() -> None:
    ep = make_episode()
    t = make_ticks(ep, pad=500)
    t["is_airborne"] = t["is_airborne"].astype(object)
    t.loc[0, "is_airborne"] = None
    _, qa = run(t)
    assert qa.passed and qa.stats["missing_tick_frac"] == 0.0


def test_missing_required_column_raises() -> None:
    with pytest.raises(ValueError, match="duck_amount"):
        run(make_ticks(make_episode()).drop(columns="duck_amount"))


# --- 15. button cross-check -------------------------------------------------------------------------------------


def test_button_crosscheck_agreement() -> None:
    ep = make_episode()
    t = make_ticks(
        ep,
        **vel(0, 250),
        yaw=90.0,
        FORWARD=lambda t: np.ones(t.shape, bool),
        FIRE=lambda t: np.zeros(t.shape, bool),
    )
    _, qa = run(t, {"weapon_fire": ev(ep.frame_tick(5) + 1)})
    assert qa.stats["xcheck_FORWARD"] == 1.0
    assert qa.stats["xcheck_FIRE"] == pytest.approx((N - 1) / N)  # the one labelled shot has no FIRE button
    assert "xcheck_BACK" not in qa.stats
    assert qa.passed  # never gating


def test_button_crosscheck_disagreement_is_reported_not_gating() -> None:
    ep = make_episode()
    t = make_ticks(ep, BACK=lambda t: np.ones(t.shape, bool))
    _, qa = run(t)
    assert qa.stats["xcheck_BACK"] == 0.0
    assert qa.passed


# --- 16. output format ------------------------------------------------------------------------------------------


def test_output_format() -> None:
    ep = make_episode()
    a, qa = run(
        make_ticks(ep, yaw=lambda t: 0.05 * (t - T0), pitch=lambda t: -0.02 * (t - T0), **vel(100, 100)),
        {"weapon_fire": ev(ep.frame_tick(2) + 1)},
    )
    assert a.dtype == np.float32 and a.shape == (N, 51)
    assert set(np.unique(a)) <= {0.0, 1.0}
    assert (a[:, SPEC.mouse_x_slice].sum(axis=1) == 1).all()
    assert (a[:, SPEC.mouse_y_slice].sum(axis=1) == 1).all()
    for name in (*SPEC.keys, *SPEC.clicks):
        assert f"rate_{name}" in qa.stats
    assert {"rate_mouse_x_nonzero", "rate_mouse_y_nonzero"} <= set(qa.stats)
    assert all(isinstance(v, float) for v in qa.stats.values())


def test_rates_match_labels() -> None:
    ep = make_episode()
    _, qa = run(make_ticks(ep, yaw=90.0, **vel(0, 250)))
    assert qa.stats["rate_w"] == 1.0 and qa.stats["rate_s"] == 0.0


# --- 17. JSON ---------------------------------------------------------------------------------------------------


def test_to_json_round_trip_and_nan_to_null() -> None:
    qa = EpisodeQA("e1", False, ["fire_recall 0.950 < 0.98"], {"b": 1.0, "a": float("nan"), "c": float("inf")})
    line = qa.to_json()
    assert "\n" not in line
    d = json.loads(line)
    assert d == {
        "episode_id": "e1",
        "passed": False,
        "reasons": ["fire_recall 0.950 < 0.98"],
        "stats": {"a": None, "b": 1.0, "c": None},
    }
    assert list(d["stats"]) == ["a", "b", "c"]


def test_extracted_qa_serialises() -> None:
    _, qa = run(make_ticks(make_episode()))
    d = json.loads(qa.to_json())
    assert d["episode_id"] == "dm-demo-s000-c00" and d["passed"] is True
