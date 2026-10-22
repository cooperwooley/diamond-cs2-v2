"""Tests for cs2wm.actions."""

from fractions import Fraction

import numpy as np
import pytest

from cs2wm.actions import (
    ActionSpec,
    bin_index,
    counts_to_deg,
    decode,
    encode,
    encode_batch,
    slot_for_weapon,
    wrap180,
    yaw_pitch_delta_to_mouse_deg,
)
from cs2wm.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def spec(cfg) -> ActionSpec:
    return ActionSpec.from_config(cfg)


def test_layout(spec: ActionSpec) -> None:
    assert spec.keys == ("w", "a", "s", "d", "space", "ctrl", "shift", "1", "2", "3", "r")
    assert spec.clicks == ("lmb", "rmb")
    assert spec.dim == 51
    assert (spec.idx_lmb, spec.idx_rmb) == (11, 12)
    assert spec.mouse_x_slice == slice(13, 36)
    assert spec.mouse_y_slice == slice(36, 51)
    assert spec.key_index("r") == 10


def test_key_index_unknown_lists_valid(spec: ActionSpec) -> None:
    with pytest.raises(KeyError, match="valid keys"):
        spec.key_index("q")


def test_unit_to_degree_centres(spec: ActionSpec) -> None:
    x = spec.mouse_x_centers_deg
    assert x.dtype == np.float64
    assert x[spec.mouse_x_units.index(60)] == pytest.approx(3.3)
    assert x[-1] == pytest.approx(55.0)
    assert x[0] == pytest.approx(-55.0)
    assert spec.mouse_y_centers_deg[-1] == pytest.approx(11.0)


def test_sign_conventions() -> None:
    x = yaw_pitch_delta_to_mouse_deg(-2.0, 0.0)[0]
    assert x == 2.0  # yaw decreasing is a right turn
    assert yaw_pitch_delta_to_mouse_deg(2.0, 0.0)[0] == -2.0
    assert yaw_pitch_delta_to_mouse_deg(0.0, 1.0)[1] == 1.0  # pitch increasing is looking down


def test_yaw_wrap() -> None:
    # 179 -> -179 is a 2 degree left turn, not a 358 degree right turn.
    x, _ = yaw_pitch_delta_to_mouse_deg(-179.0 - 179.0, 0.0)
    assert x == pytest.approx(-2.0)


def test_yaw_pitch_vectorised() -> None:
    x, y = yaw_pitch_delta_to_mouse_deg(np.array([-2.0, -358.0]), np.array([1.0, -1.0]))
    np.testing.assert_allclose(x, [2.0, -2.0])
    np.testing.assert_allclose(y, [1.0, -1.0])


def test_wrap180() -> None:
    assert float(wrap180(190.0)) == -170.0
    assert float(wrap180(-180.0)) == -180.0
    assert float(wrap180(180.0)) == -180.0
    np.testing.assert_allclose(wrap180(np.array([0.0, 360.0, -358.0])), [0.0, 0.0, 2.0])


def test_bin_index_exact_centres(spec: ActionSpec) -> None:
    c = spec.mouse_x_centers_deg
    np.testing.assert_array_equal(bin_index(c, c), np.arange(len(c)))
    out = bin_index(0.0, c)
    assert out.dtype == np.int64
    assert int(out) == spec.mouse_x_units.index(0)


def test_bin_index_clips(spec: ActionSpec) -> None:
    c = spec.mouse_x_centers_deg
    assert int(bin_index(1e6, c)) == len(c) - 1
    assert int(bin_index(-1e6, c)) == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_bin_index_non_finite(spec: ActionSpec, bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        bin_index(bad, spec.mouse_x_centers_deg)
    with pytest.raises(ValueError, match="finite"):
        bin_index(np.array([0.0, bad]), spec.mouse_x_centers_deg)


def _pz_reference(units_value: float, units: tuple[float, ...]) -> int:
    """Pearce & Zhu: ``min(possibles, key=lambda v: abs(v - x))``; the first minimum wins."""
    best = min(units, key=lambda v: abs(v - units_value))
    return units.index(best)


def test_bin_index_ties_go_to_lower_index(spec: ActionSpec) -> None:
    c = spec.mouse_x_centers_deg
    u = spec.mouse_x_units
    for i in (3, 15):  # one negative and one positive midpoint
        assert _pz_reference((u[i] + u[i + 1]) / 2, u) == i
        assert int(bin_index((c[i] + c[i + 1]) / 2, c)) == i
        assert int(bin_index((c[i] + c[i + 1]) / 2 + 1e-6, c)) == i + 1


@pytest.mark.parametrize("axis", ["x", "y"])
def test_bin_index_matches_pz_reference(spec: ActionSpec, axis: str) -> None:
    units = getattr(spec, f"mouse_{axis}_units")
    centres = getattr(spec, f"mouse_{axis}_centers_deg")
    rng = np.random.default_rng(0)
    # Half-integer unit values land on many exact midpoints in unit space.
    values = np.concatenate([rng.integers(-2400, 2400, 2000) / 2, rng.uniform(-1500, 1500, 2000)])
    got = bin_index(values * spec.deg_per_unit, centres)
    want = np.array([_pz_reference(float(v), units) for v in values])
    np.testing.assert_array_equal(got, want)


def _pz_reference_exact(u: Fraction, units: tuple[float, ...]) -> int:
    """P&Z nearest-bin rule on an exact rational; the first minimum wins."""
    exact = [Fraction(str(v)) for v in units]
    return exact.index(min(exact, key=lambda v: abs(v - u)))


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("sens", [2.5, 1.0, 1.7])
def test_bin_index_integer_count_sweep_matches_pz(spec: ActionSpec, axis: str, sens: float) -> None:
    """Live play: integer mouse counts hit exact midpoints (sens 2.5: every odd count), and float error must not decide."""
    units = getattr(spec, f"mouse_{axis}_units")
    centres = getattr(spec, f"mouse_{axis}_centers_deg")
    counts = range(-1500, 1501)
    got = [int(bin_index(counts_to_deg(c, sens, 0.022), centres)) for c in counts]
    want = [_pz_reference_exact(Fraction(c) * Fraction(str(sens)) / Fraction(5, 2), units) for c in counts]
    assert got == want


@pytest.mark.parametrize("axis", ["x", "y"])
def test_bin_index_integer_unit_label_sweep_matches_pz(spec: ActionSpec, axis: str) -> None:
    """Label path: ``units * deg_per_unit`` for integer units bins like P&Z."""
    units = getattr(spec, f"mouse_{axis}_units")
    centres = getattr(spec, f"mouse_{axis}_centers_deg")
    values = range(-1500, 1501)
    got = [int(bin_index(u * spec.deg_per_unit, centres)) for u in values]
    want = [_pz_reference_exact(Fraction(u), units) for u in values]
    assert got == want


def test_counts_to_deg_sensitivity_2_5_is_pz_unit(spec: ActionSpec) -> None:
    assert counts_to_deg(1, 2.5, 0.022) == pytest.approx(spec.deg_per_unit)
    assert counts_to_deg(60, 2.5) == pytest.approx(3.3)


def test_encode_decode_roundtrip(spec: ActionSpec) -> None:
    vec = encode(["w", "shift", "2"], True, False, 3.3, -0.11, spec)
    assert vec.dtype == np.float32
    assert vec.shape == (51,)
    assert vec[spec.mouse_x_slice].sum() == 1
    assert vec[spec.mouse_y_slice].sum() == 1
    d = decode(vec, spec)
    assert d.keys == ("w", "shift", "2")
    assert d.lmb and not d.rmb
    assert d.mouse_x_deg == pytest.approx(3.3)
    assert d.mouse_y_deg == pytest.approx(-0.11)


def test_encode_unknown_key(spec: ActionSpec) -> None:
    with pytest.raises(KeyError):
        encode(["q"], False, False, 0.0, 0.0, spec)


def test_encode_does_not_filter_forbidden_combos(spec: ActionSpec) -> None:
    vec = encode(["w", "s"], False, False, 0.0, 0.0, spec)
    assert vec[0] == 1 and vec[2] == 1


def test_decode_wrong_length(spec: ActionSpec) -> None:
    with pytest.raises(ValueError, match="shape"):
        decode(np.zeros(50), spec)


def test_encode_batch_equals_stacked_encode(spec: ActionSpec) -> None:
    rng = np.random.default_rng(1)
    n = 64
    km = rng.random((n, 11)) < 0.3
    lmb, rmb = rng.random(n) < 0.5, rng.random(n) < 0.5
    mx, my = rng.normal(0, 8, n), rng.normal(0, 3, n)
    got = encode_batch(km, lmb, rmb, mx, my, spec)
    want = np.stack(
        [encode([k for k, on in zip(spec.keys, km[i]) if on], lmb[i], rmb[i], mx[i], my[i], spec) for i in range(n)]
    )
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, want)


def test_encode_batch_validates_shapes(spec: ActionSpec) -> None:
    with pytest.raises(ValueError, match="key_matrix"):
        encode_batch(np.zeros((3, 10), bool), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), spec)
    with pytest.raises(ValueError, match="lmb"):
        encode_batch(np.zeros((3, 11), bool), np.zeros(2), np.zeros(3), np.zeros(3), np.zeros(3), spec)


@pytest.mark.parametrize(
    ("weapon", "slot"),
    [
        ("AK-47", "1"),
        ("weapon_m4a1_silencer", "1"),
        ("Glock-18", "2"),
        ("USP-S", "2"),
        ("Knife", "3"),
        ("Smoke Grenade", "3"),
        (None, None),
        ("", None),
    ],
)
def test_slot_for_weapon(cfg, weapon: str | None, slot: str | None) -> None:
    assert slot_for_weapon(weapon, cfg) == slot


@pytest.mark.parametrize(
    ("weapon", "slot"),
    [
        ("weapon_ak47", "1"),
        ("weapon_sg556", "1"),
        ("weapon_elite", "2"),
        ("weapon_fiveseven", "2"),
        ("weapon_hkp2000", "2"),
        ("weapon_revolver", "2"),
        ("weapon_knife_t", "3"),
        ("weapon_smokegrenade", "3"),
        ("weapon_c4", "3"),
    ],
)
def test_slot_for_weapon_internal_names(cfg, weapon: str, slot: str) -> None:
    assert slot_for_weapon(weapon, cfg) == slot


@pytest.mark.parametrize(
    "name",
    ["knife", "knife_t", "bayonet", "knife_karambit", "flashbang", "hegrenade", "smokegrenade", "molotov"]
    + ["incgrenade", "decoy", "taser", "c4"],
)
def test_no_slot_substring_matches_knife_or_utility(cfg, name: str) -> None:
    assert slot_for_weapon(name, cfg) == "3"
    assert slot_for_weapon(f"weapon_{name}", cfg) == "3"


def test_play_extract_consistency(spec: ActionSpec) -> None:
    """Live input (counts at the player's sensitivity) and extracted labels (degrees) bin identically."""
    rng = np.random.default_rng(2)
    centres = spec.mouse_x_centers_deg
    for sens in (0.5, 1.0, 2.5, 3.7):
        for deg in rng.uniform(-60, 60, 500):
            counts = deg / (0.022 * sens)
            assert int(bin_index(counts_to_deg(counts, sens), centres)) == int(bin_index(deg, centres))
