"""Tests for cs2wm.eval.rotation (#32): shift estimators on synthetic shifted images, sign convention, pinhole maths."""

import math

import cv2
import numpy as np
import pytest

from cs2wm.eval import rotation as rot

H, W = 150, 280
PAD = 24


def texture(seed: int = 0, h: int = H + 2 * PAD, w: int = W + 2 * PAD) -> np.ndarray:
    """Smooth random BGR texture, big enough that shifted crops are real (non-periodic) views."""
    rng = np.random.default_rng(seed)
    img = rng.random((h, w, 3)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 2.5)
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)


def view(big: np.ndarray, sx: float, sy: float, h: int = H, w: int = W) -> np.ndarray:
    """The central ``h x w`` window of ``big`` whose content has moved by ``(sx, sy)`` pixels (+x right, +y down)."""
    m = np.float32([[1, 0, sx], [0, 1, sy]])
    warped = cv2.warpAffine(big, m, (big.shape[1], big.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    y0, x0 = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
    return warped[y0 : y0 + h, x0 : x0 + w]


@pytest.fixture(scope="module")
def big() -> np.ndarray:
    return texture()


@pytest.mark.parametrize("sx,sy", [(0, 0), (3, 0), (-3, 0), (0, 2), (0, -2), (5, -4), (-8, 6), (1.5, 0), (-0.5, 0.75)])
def test_phasecorr_recovers_shift_and_sign(big, sx, sy) -> None:
    dx, dy, response = rot.estimate_shift_phasecorr(view(big, 0, 0), view(big, sx, sy))
    assert dx == pytest.approx(sx, abs=0.15) and dy == pytest.approx(sy, abs=0.15)
    assert response > 0


def test_phasecorr_sign_convention_integer_roll(big) -> None:
    """Content moving right (np.roll along +x) is +dx; moving down is +dy."""
    a = view(big, 0, 0)
    dx, dy, _ = rot.estimate_shift_phasecorr(a, np.roll(a, 4, axis=1))
    assert dx > 3.5 and abs(dy) < 0.3
    dx, dy, _ = rot.estimate_shift_phasecorr(a, np.roll(a, 3, axis=0))
    assert dy > 2.5 and abs(dx) < 0.3


@pytest.mark.parametrize("sx,sy", [(0, 0), (3, 0), (-3, 0), (0, 2), (0, -2), (4, -3), (1.5, 0.5)])
def test_flow_recovers_shift_and_sign(big, sx, sy) -> None:
    dx, dy, response = rot.estimate_shift_flow(view(big, 0, 0), view(big, sx, sy))
    assert dx == pytest.approx(sx, abs=0.3) and dy == pytest.approx(sy, abs=0.3)
    assert math.isnan(response)


@pytest.mark.parametrize("sx,sy", [(2, 0), (-4, 3), (0.5, -1.5)])
def test_flow_and_phasecorr_agree(big, sx, sy) -> None:
    a, b = view(big, 0, 0), view(big, sx, sy)
    pc, fl = rot.estimate_shift_phasecorr(a, b), rot.estimate_shift_flow(a, b, border=10)
    assert pc.dx == pytest.approx(fl.dx, abs=0.4) and pc.dy == pytest.approx(fl.dy, abs=0.4)


@pytest.mark.parametrize("fn", [rot.estimate_shift_phasecorr, rot.estimate_shift_flow])
def test_low_res_shift(fn) -> None:
    """At 30x56 a 1 px shift is recovered too (the robustness-check resolution)."""
    big = cv2.resize(texture(1, 330, 480), (480 // 5, 330 // 5), interpolation=cv2.INTER_AREA)
    big = cv2.GaussianBlur(big, (0, 0), 0.8)
    for sx, sy in [(1.0, 0.0), (-1.0, 0.5), (0.0, -1.0)]:
        got = fn(view(big, 0, 0, 30, 56), view(big, sx, sy, 30, 56))
        assert got.dx == pytest.approx(sx, abs=0.35) and got.dy == pytest.approx(sy, abs=0.35)


def test_flow_border_validation_and_effect(big) -> None:
    a, b = view(big, 0, 0), view(big, 2, 0)
    with pytest.raises(ValueError, match="border"):
        rot.estimate_shift_flow(a, b, border=75)
    assert rot.estimate_shift_flow(a, b, border=20).dx == pytest.approx(2, abs=0.3)


def test_flow_divergence_sign_for_zoom(big) -> None:
    a = view(big, 0, 0)
    c = (W / 2, H / 2)
    zoom_in = cv2.warpAffine(a, cv2.getRotationMatrix2D(c, 0, 1.04), (W, H), borderMode=cv2.BORDER_REFLECT)
    zoom_out = cv2.warpAffine(a, cv2.getRotationMatrix2D(c, 0, 0.96), (W, H), borderMode=cv2.BORDER_REFLECT)
    assert rot.flow_divergence(a, zoom_in, border=10) > 0.02
    assert rot.flow_divergence(a, zoom_out, border=10) < -0.02
    assert abs(rot.flow_divergence(a, view(big, 3, 0), border=10)) < 0.01


def test_grayscale_input_is_accepted(big) -> None:
    a, b = view(big, 0, 0), view(big, 2, 0)
    gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    assert rot.estimate_shift_phasecorr(gray(a), gray(b)).dx == pytest.approx(2, abs=0.15)


# --- pinhole conversion ------------------------------------------------------------------------------------------


def test_focal_lengths_match_hand_computation() -> None:
    # hfov 106.26 deg = 2 atan(4/3): tan(hfov/2) = 4/3, tan(vfov/2) = 4/3 * 720/1280 = 0.75.
    hfov = math.degrees(2 * math.atan(4 / 3))
    f_x, f_y = rot.focal_lengths_px(280, 150, hfov)
    assert f_x == pytest.approx(140 / (4 / 3)) == pytest.approx(105.0)
    assert f_y == pytest.approx(75 / 0.75) == pytest.approx(100.0)
    f_x, f_y = rot.focal_lengths_px(56, 30, hfov)
    assert (f_x, f_y) == (pytest.approx(21.0), pytest.approx(20.0))


def test_shift_to_degrees_is_per_axis_with_repo_signs() -> None:
    hfov = math.degrees(2 * math.atan(4 / 3))
    yaw, pitch = rot.shift_to_degrees(-10.5, 0.0, 280, 150, hfov)  # content moved left => turned right
    assert yaw == pytest.approx(math.degrees(math.atan(10.5 / 105)))
    assert yaw > 0 and pitch == 0
    yaw, pitch = rot.shift_to_degrees(0.0, -10.0, 280, 150, hfov)  # content moved up => looked down
    assert pitch == pytest.approx(math.degrees(math.atan(10 / 100))) and pitch > 0 and yaw == 0
    # The same pixel count means more degrees vertically: f_y < f_x (not hfov / width for both axes).
    yaw, pitch = rot.shift_to_degrees(-5.0, -5.0, 280, 150, hfov)
    assert pitch > yaw


def test_shift_to_degrees_accepts_arrays() -> None:
    yaw, pitch = rot.shift_to_degrees(np.array([-2.0, 0.0, 2.0]), np.zeros(3), 280, 150, 106.26)
    assert yaw[0] > 0 and yaw[1] == 0 and yaw[2] < 0 and yaw[0] == pytest.approx(-yaw[2])
    assert yaw.shape == pitch.shape == (3,)


def test_theoretical_slope_is_small_angle_limit() -> None:
    sx, sy = rot.theoretical_slopes(280, 150, 106.26)
    yaw, pitch = rot.shift_to_degrees(-0.01, -0.01, 280, 150, 106.26)
    assert yaw / 0.01 == pytest.approx(sx, rel=1e-4) and pitch / 0.01 == pytest.approx(sy, rel=1e-4)
    assert sy > sx
