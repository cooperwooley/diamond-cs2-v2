"""Tests for cs2wm.play.input: pygame-like events to the 51-dim action, headless."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from cs2wm.actions import ActionSpec, decode, encode_batch, yaw_pitch_delta_to_mouse_deg
from cs2wm.config import load_config
from cs2wm.play.input import CMD_QUIT, CMD_RESET, CMD_TOGGLE_OVERLAY, InputBindings, InputMapper

KEYDOWN, KEYUP, MBDOWN, MBUP, MOTION, QUIT = 1, 2, 3, 4, 5, 6
CODES = {"w": 10, "a": 11, "s": 12, "d": 13, "space": 14, "ctrl": 15, "shift": 16, "1": 17, "2": 18, "3": 19, "r": 20}
K_4, K_RCTRL, K_BACKSPACE, K_F1, K_ESC = 21, 22, 23, 24, 25
FOCUSLOST = 7
BINDINGS = InputBindings(
    keydown=KEYDOWN,
    keyup=KEYUP,
    mousebuttondown=MBDOWN,
    mousebuttonup=MBUP,
    mousemotion=MOTION,
    quit=QUIT,
    keys={**{code: name for name, code in CODES.items()}, K_4: "3", K_RCTRL: "ctrl"},
    key_reset=K_BACKSPACE,
    key_overlay=K_F1,
    key_quit=K_ESC,
    windowfocuslost=FOCUSLOST,
)


def down(code: int) -> SimpleNamespace:
    return SimpleNamespace(type=KEYDOWN, key=code)


def up(code: int) -> SimpleNamespace:
    return SimpleNamespace(type=KEYUP, key=code)


def motion(dx: float, dy: float) -> SimpleNamespace:
    return SimpleNamespace(type=MOTION, rel=(dx, dy))


def click(button: int, pressed: bool = True) -> SimpleNamespace:
    return SimpleNamespace(type=MBDOWN if pressed else MBUP, button=button)


@pytest.fixture(scope="module")
def spec() -> ActionSpec:
    return ActionSpec.from_config(load_config())


def make_mapper(sensitivity: float = 2.5) -> InputMapper:
    cfg = load_config(overrides=[f"play.sensitivity={sensitivity}"])
    return InputMapper(cfg, BINDINGS)


def feed(mapper: InputMapper, *events) -> None:
    for e in events:
        mapper.handle_event(e)


def keys_of(mapper: InputMapper) -> tuple[str, ...]:
    return decode(mapper.frame_action(), mapper.spec).keys


def test_idle_action_is_zero_mouse_bins(spec) -> None:
    action = make_mapper().frame_action()
    assert action.shape == (51,) and action.dtype == np.float32
    d = decode(action, spec)
    assert d.keys == () and not d.lmb and not d.rmb and d.mouse_x_deg == 0.0 and d.mouse_y_deg == 0.0
    assert action.sum() == 2  # exactly the two idle mouse bins


def test_held_keys_persist_across_frames_until_released() -> None:
    m = make_mapper()
    feed(m, down(CODES["w"]), down(CODES["shift"]))
    assert keys_of(m) == ("w", "shift")
    assert keys_of(m) == ("w", "shift")
    feed(m, up(CODES["w"]))
    assert keys_of(m) == ("shift",)


def test_right_ctrl_counts_as_ctrl() -> None:
    m = make_mapper()
    feed(m, down(K_RCTRL))
    assert keys_of(m) == ("ctrl",)


@pytest.mark.parametrize(
    ("press", "slot_idx"), [("1", 7), ("2", 8), ("3", 9), (K_4, 9)], ids=["slot1", "slot2", "slot3", "key4_to_idx9"]
)
def test_slot_keys_map_to_indices(press, slot_idx) -> None:
    m = make_mapper()
    feed(m, down(CODES[press] if isinstance(press, str) else press))
    action = m.frame_action()
    assert np.flatnonzero(action[:11]).tolist() == [slot_idx]


def test_slot_is_a_one_frame_pulse_like_the_labels() -> None:
    m = make_mapper()
    feed(m, down(CODES["2"]))
    assert keys_of(m) == ("2",)
    assert keys_of(m) == ()  # still held, but the labels only mark the frame of the weapon change


def test_slot_tapped_within_one_frame_is_not_lost() -> None:
    m = make_mapper()
    feed(m, down(CODES["1"]), up(CODES["1"]))
    assert keys_of(m) == ("1",)


def test_newer_slot_replaces_older_slot() -> None:
    m = make_mapper()
    feed(m, down(CODES["1"]), down(K_4), down(CODES["2"]))
    assert keys_of(m) == ("2",)
    feed(m, down(CODES["2"]), down(K_4))
    assert keys_of(m) == ("3",)


@pytest.mark.parametrize("pair", [("w", "s"), ("a", "d")])
def test_opposing_movement_cancels(pair) -> None:
    m = make_mapper()
    feed(m, down(CODES[pair[0]]), down(CODES[pair[1]]), down(CODES["shift"]))
    assert keys_of(m) == ("shift",)
    feed(m, up(CODES[pair[1]]))
    assert keys_of(m) == (pair[0], "shift")


def test_clicks() -> None:
    m = make_mapper()
    feed(m, click(1), click(3), click(2))  # middle button ignored
    d = decode(m.frame_action(), m.spec)
    assert d.lmb and d.rmb
    d = decode(m.frame_action(), m.spec)
    assert d.lmb and not d.rmb  # LMB is held (labels hold-fill fire); RMB is a one-frame pulse
    feed(m, click(1, pressed=False))
    assert not decode(m.frame_action(), m.spec).lmb


def test_mouse_motion_accumulates_over_the_frame_then_resets(spec) -> None:
    m = make_mapper(sensitivity=2.5)
    # Last-event-wins (DIAMOND) would give +1 count -> 0.055 deg -> unit 2 bin; summing gives 21 counts.
    feed(m, motion(10, -3), motion(5, 1), motion(5, 1), motion(1, 0))
    assert (m.state.mouse_dx_counts, m.state.mouse_dy_counts) == (21, -1)
    d = decode(m.frame_action(), spec)
    assert d.mouse_x_deg == pytest.approx(20 * 0.055)  # 21 counts * 0.055 = 1.155 deg = 21 units -> nearest bin 20
    assert d.mouse_y_deg == pytest.approx(-0.055 * 2)  # -1 unit ties between -2 and 0 and goes to the lower bin
    assert (m.state.mouse_dx_counts, m.state.mouse_dy_counts) == (0, 0)
    d = decode(m.frame_action(), spec)
    assert d.mouse_x_deg == 0.0 and d.mouse_y_deg == 0.0


def test_mouse_signs(spec) -> None:
    m = make_mapper()
    feed(m, motion(100, 40))  # right and down on screen
    d = decode(m.frame_action(), spec)
    assert d.mouse_x_deg > 0 and d.mouse_y_deg > 0


@pytest.mark.parametrize("sensitivity", [0.5, 1.0, 2.5, 3.1])
@pytest.mark.parametrize("counts", [-400, -37, -5, -1, 0, 1, 3, 12, 80, 900])
def test_counts_to_degrees_to_bin(spec, sensitivity, counts) -> None:
    m = make_mapper(sensitivity)
    feed(m, motion(counts, 0))
    # Exact reference: nearest centre in rational arithmetic, ties to the lower bin (min() keeps the first minimum).
    deg = counts * Fraction("0.022") * Fraction(str(sensitivity))
    centers = [Fraction(str(u)) * Fraction("0.055") for u in load_config().actions.mouse.x_units]
    expected = min(range(len(centers)), key=lambda i: abs(centers[i] - deg))
    assert int(np.argmax(m.frame_action()[spec.mouse_x_slice])) == expected


@pytest.mark.parametrize("sensitivity", [0.5, 1.0, 2.5, 3.1])
def test_play_bin_equals_label_bin_for_the_same_degree_delta(spec, sensitivity) -> None:
    counts = np.arange(-1200, 1201, 7)
    deg_x = counts * 0.022 * sensitivity
    deg_y = counts[::-1] * 0.022 * sensitivity
    # Label side: Source yaw/pitch deltas -> mouse degrees -> encode_batch, as extract_actions does.
    mx, my = yaw_pitch_delta_to_mouse_deg(-deg_x, deg_y)
    n = len(counts)
    labels = encode_batch(np.zeros((n, 11), bool), np.zeros(n, bool), np.zeros(n, bool), mx, my, spec)
    m = make_mapper(sensitivity)
    for i, (cx, cy) in enumerate(zip(counts, counts[::-1])):
        feed(m, motion(cx, cy))
        np.testing.assert_array_equal(m.frame_action(), labels[i])


def test_exact_pz_units_at_sensitivity_2_5(spec) -> None:
    # At sensitivity 2.5 one count is one P&Z unit, so every bin centre is hit exactly.
    for idx, unit in enumerate(load_config().actions.mouse.x_units):
        m = make_mapper(2.5)
        feed(m, motion(unit, 0))
        assert int(np.argmax(m.frame_action()[spec.mouse_x_slice])) == idx


def test_pixel_ties_go_to_the_lower_bin(spec) -> None:
    m = make_mapper(2.5)
    feed(m, motion(3, 0))  # 3 units: midway between the 2 and 4 bins
    assert decode(m.frame_action(), spec).mouse_x_deg == pytest.approx(2 * 0.055)


def test_huge_motion_clips_to_extreme_bin(spec) -> None:
    m = make_mapper()
    feed(m, motion(-1e6, 1e6))
    d = decode(m.frame_action(), spec)
    assert d.mouse_x_deg == pytest.approx(spec.mouse_x_centers_deg[0])
    assert d.mouse_y_deg == pytest.approx(spec.mouse_y_centers_deg[-1])


def test_control_keys_return_commands_and_leave_the_action_alone() -> None:
    m = make_mapper()
    assert m.handle_event(down(K_BACKSPACE)) == CMD_RESET
    assert m.handle_event(down(K_F1)) == CMD_TOGGLE_OVERLAY
    assert m.handle_event(down(K_ESC)) == CMD_QUIT
    assert m.handle_event(SimpleNamespace(type=QUIT)) == CMD_QUIT
    assert m.handle_event(up(K_F1)) is None
    assert m.handle_event(down(999)) is None
    assert keys_of(m) == ()


def test_reset_forgets_held_input() -> None:
    m = make_mapper()
    feed(m, down(CODES["w"]), click(1), motion(50, 50))
    m.reset()
    action = m.frame_action()
    assert action[:13].sum() == 0


def tap(name: str) -> tuple[SimpleNamespace, SimpleNamespace]:
    if name in ("lmb", "rmb"):
        button = 1 if name == "lmb" else 3
        return click(button), click(button, pressed=False)
    return down(CODES[name]), up(CODES[name])


@pytest.mark.parametrize("name", ["lmb", "rmb", "space", "r", "w", "ctrl", "shift"])
def test_tap_shorter_than_a_frame_is_not_lost(spec, name) -> None:
    m = make_mapper()
    feed(m, *tap(name))
    d = decode(m.frame_action(), spec)
    assert (name in d.keys) or (name == "lmb" and d.lmb) or (name == "rmb" and d.rmb)
    after = decode(m.frame_action(), spec)  # released: gone from the next frame
    assert after.keys == () and not after.lmb and not after.rmb


def test_tap_that_cancels_an_opposite_still_cancels() -> None:
    m = make_mapper()
    feed(m, down(CODES["s"]), *tap("w"))
    assert keys_of(m) == ()
    assert keys_of(m) == ("s",)


@pytest.mark.parametrize("name", ["space", "r", "rmb"])
def test_pulse_keys_fire_once_while_held(spec, name) -> None:
    m = make_mapper()
    feed(m, tap(name)[0])
    on = []
    for _ in range(5):
        d = decode(m.frame_action(), spec)
        on.append(name in d.keys or (name == "rmb" and d.rmb))
    assert on == [True, False, False, False, False]
    feed(m, tap(name)[1], *tap(name))  # release, press again: a new pulse
    d = decode(m.frame_action(), spec)
    assert name in d.keys or d.rmb


def test_os_key_repeat_does_not_retrigger_a_pulse(spec) -> None:
    m = make_mapper()
    feed(m, down(CODES["space"]))
    m.frame_action()
    feed(m, down(CODES["space"]), down(CODES["space"]))
    assert keys_of(m) == ()


@pytest.mark.parametrize("name", ["lmb", "w", "ctrl", "shift"])
def test_state_keys_stay_on_while_held(spec, name) -> None:
    m = make_mapper()
    feed(m, tap(name)[0])
    for _ in range(4):
        d = decode(m.frame_action(), spec)
        assert (name in d.keys) or (name == "lmb" and d.lmb)


def test_focus_loss_drops_held_input() -> None:
    m = make_mapper()
    feed(m, down(CODES["w"]), click(1), motion(9, 9))
    assert m.handle_event(SimpleNamespace(type=FOCUSLOST)) is None
    assert m.frame_action()[:13].sum() == 0


def test_drop_mouse_keeps_keys(spec) -> None:
    m = make_mapper()
    feed(m, down(CODES["w"]), motion(500, 500))
    m.drop_mouse()
    d = decode(m.frame_action(), spec)
    assert d.keys == ("w",) and d.mouse_x_deg == 0.0
