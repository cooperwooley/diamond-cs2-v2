"""Tests for the H2 smoke masks (#33): camera conventions, sphere projection, per-frame and static masks."""

import math

import numpy as np
import pandas as pd
import pytest

from cs2wm.config import load_config
from cs2wm.eval import smoke_masks as sm

H, W = 150, 280
SIZE = (H, W)
HFOV = 106.26
EYE = np.array([0.0, 0.0, 64.0])
R = 144.0


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def cfg_plain():
    return load_config(overrides=["eval.h2.mask_dilate_px=0"])


def sphere(center, yaw=0.0, pitch=0.0, eye=EYE):
    return sm.project_sphere(np.array(center, dtype=float), R, eye, yaw, pitch, HFOV, SIZE)


def smokes(*rows, detonate=0, expire=1000):
    return pd.DataFrame(
        [
            {
                "entity_id": i,
                "detonate_tick": detonate,
                "expire_tick": expire,
                "x": x,
                "y": y,
                "z": z,
                "thrower_user_id": 1,
            }
            for i, (x, y, z) in enumerate(rows)
        ]
    )


# --- camera model --------------------------------------------------------------------------------------------------


def test_view_basis_is_orthonormal_and_follows_the_source_convention() -> None:
    for yaw, pitch in [(0, 0), (90, 0), (-135, 30), (200, -60), (10, 89)]:
        f, r, u = sm.view_basis(yaw, pitch)
        assert np.allclose([f @ f, r @ r, u @ u, f @ r, f @ u, r @ u], [1, 1, 1, 0, 0, 0], atol=1e-12)
        assert np.allclose(np.cross(u, r), f)  # (right, up, forward) is a left-handed triple, as in camera space
    f, r, u = sm.view_basis(0, 0)
    assert np.allclose(f, [1, 0, 0]) and np.allclose(r, [0, -1, 0]) and np.allclose(u, [0, 0, 1])
    assert np.allclose(sm.view_basis(90, 0)[0], [0, 1, 0])  # yaw is counter-clockwise: 90 faces +y
    assert np.allclose(sm.view_basis(0, 90)[0], [0, 0, -1])  # pitch is positive looking down
    assert np.allclose(sm.view_basis(0, 30)[0], [math.cos(math.radians(30)), 0, -0.5])


def test_smoke_straight_ahead_is_a_centred_ellipse() -> None:
    cx, cy, rx, ry = sphere([1000, 0, 64])
    assert (cx, cy) == (pytest.approx(W / 2), pytest.approx(H / 2))
    f_x, f_y = 140 / math.tan(math.radians(HFOV / 2)), 75 / (math.tan(math.radians(HFOV / 2)) * 720 / 1280)
    alpha = math.asin(R / 1000)
    assert rx == pytest.approx(f_x * math.tan(alpha)) and ry == pytest.approx(f_y * math.tan(alpha))
    assert rx > ry  # f_x > f_y: the stored frames are squashed horizontally, so a sphere is wider than tall


def test_smoke_to_the_right_is_on_the_right_half() -> None:
    # Facing +x, the right-hand side is -y in Source.
    cx, cy, rx, ry = sphere([1000, -600, 64])
    assert cx - rx > W / 2 and cy == pytest.approx(H / 2)
    mask = sm.rasterize_ellipse((cx, cy, rx, ry), SIZE)
    assert mask[:, W // 2 :].any() and not mask[:, : W // 2].any()
    left = sphere([1000, 600, 64])  # +y is the left
    assert left[0] == pytest.approx(W - cx) and left[2] == pytest.approx(rx)


def test_yaw_rotates_counter_clockwise() -> None:
    # Facing +y (yaw 90), +x is on the right and -x on the left.
    assert sphere([600, 1000, 64], yaw=90)[0] > W / 2
    assert sphere([-600, 1000, 64], yaw=90)[0] < W / 2
    assert sphere([0, 1000, 64], yaw=90)[0] == pytest.approx(W / 2)
    assert sphere([-1000, 0, 64], yaw=180)[0] == pytest.approx(W / 2)


def test_smoke_above_is_on_the_upper_half() -> None:
    cx, cy, rx, ry = sphere([1000, 0, 500])
    assert cx == pytest.approx(W / 2) and cy + ry < H / 2  # rows grow downwards
    mask = sm.rasterize_ellipse((cx, cy, rx, ry), SIZE)
    assert mask[: H // 2].any() and not mask[H // 2 :].any()
    below = sphere([1000, 0, -400])
    assert below[1] > H / 2


def test_smoke_behind_the_camera_has_no_mask(cfg) -> None:
    assert sphere([-1000, 0, 64]) is None
    assert sphere([-300, 200, 64]) is None
    assert not sm.smoke_mask_for_frame(10, smokes((-1000, 0, 64)), EYE, 0.0, 0.0, cfg).any()
    assert sphere([-1000, 0, 64], yaw=180) is not None  # turn around and it is there


def test_looking_down_at_a_smoke_below_is_centred() -> None:
    cx, cy, *_ = sphere([0, 0, -300], pitch=90)
    assert (cx, cy) == (pytest.approx(W / 2), pytest.approx(H / 2))
    # Looking down, "up" on the screen is the heading: a smoke ahead of the player is in the upper half.
    assert sphere([300, 0, -300], pitch=90)[1] < H / 2
    assert sphere([0, -300, -300], pitch=90)[0] > W / 2  # and -y stays on the right


def test_positive_pitch_looks_down() -> None:
    # A smoke at eye height is above the screen centre once the camera looks down, below it when it looks up.
    assert sphere([1000, 0, 64], pitch=20)[1] < H / 2
    assert sphere([1000, 0, 64], pitch=-20)[1] > H / 2
    # The exact angle: looking down 20 degrees puts the horizon 20 degrees above the centre.
    f_y = 75 / (math.tan(math.radians(HFOV / 2)) * 720 / 1280)
    assert H / 2 - sphere([1000, 0, 64], pitch=20)[1] == pytest.approx(f_y * math.tan(math.radians(20)), rel=0.05)


def test_ellipse_shrinks_with_distance_by_the_angular_radius() -> None:
    near, far = sphere([1000, 0, 64]), sphere([2000, 0, 64])
    assert near[2] > far[2] and near[3] > far[3]
    assert near[2] / far[2] == pytest.approx(math.tan(math.asin(R / 1000)) / math.tan(math.asin(R / 2000)))
    assert near[2] / far[2] == pytest.approx(2.0, rel=0.03)  # about 1 / distance


def test_camera_inside_the_sphere_masks_the_whole_frame(cfg_plain) -> None:
    assert sphere([100, 0, 64])[2:] == (math.inf, math.inf)
    mask = sm.smoke_mask_for_frame(10, smokes((100, 0, 64)), EYE, 0.0, 0.0, cfg_plain)
    assert mask.all()
    assert sm.smoke_mask_for_frame(10, smokes((60, 60, 0)), EYE, 270.0, 45.0, cfg_plain).all()  # any viewing direction


def test_a_sphere_that_crosses_the_camera_plane_covers_its_visible_side(cfg_plain) -> None:
    centre = (-50.0, -150.0, 64.0)  # outside the sphere (distance 158), centre behind, mostly to the right
    ellipse = sphere(centre)
    assert ellipse is not None
    mask = sm.smoke_mask_for_frame(10, smokes(centre), EYE, 0.0, 0.0, cfg_plain)
    assert mask[:, -1].all() and not mask[:, 0].any() and mask.any()


def test_sphere_outside_the_frustum_is_none() -> None:
    assert sphere([1000, -3000, 64]) is None  # far to the right of the 106 degree view
    assert sphere([1000, 0, 3000]) is None


# --- eye position --------------------------------------------------------------------------------------------------


def test_eye_position_standing_crouching_and_given(cfg) -> None:
    assert np.allclose(sm.eye_position((10, 20, 30), 0.0, cfg), [10, 20, 94])
    assert np.allclose(sm.eye_position((10, 20, 30), 1.0, cfg), [10, 20, 76])
    assert np.allclose(sm.eye_position((10, 20, 30), 0.5, cfg), [10, 20, 85])
    assert np.allclose(sm.eye_position((10, 20, 30), float("nan"), cfg), [10, 20, 94])
    assert np.allclose(sm.eye_position((10, 20, 30), 1.0, cfg, view_offset=60.0), [10, 20, 90])  # given wins
    assert np.allclose(sm.eye_position((10, 20, 30), 0.0, cfg, view_offset=float("nan")), [10, 20, 94])


# --- per-frame masks -----------------------------------------------------------------------------------------------


def test_active_window_is_detonate_inclusive_expire_exclusive(cfg_plain) -> None:
    ev = smokes((1000, 0, 64), detonate=100, expire=200)

    def n(tick):
        return sm.smoke_mask_for_frame(tick, ev, EYE, 0.0, 0.0, cfg_plain).sum()

    assert n(99) == 0 and n(100) > 0 and n(199) > 0 and n(200) == 0


def test_mask_is_the_union_of_ellipses_and_matches_the_projection(cfg_plain) -> None:
    a, b = (1000.0, 0.0, 64.0), (1200.0, -700.0, 64.0)
    both = sm.smoke_mask_for_frame(10, smokes(a, b), EYE, 0.0, 0.0, cfg_plain)
    only_a = sm.smoke_mask_for_frame(10, smokes(a), EYE, 0.0, 0.0, cfg_plain)
    only_b = sm.smoke_mask_for_frame(10, smokes(b), EYE, 0.0, 0.0, cfg_plain)
    assert np.array_equal(both, only_a | only_b) and only_a.any() and only_b.any() and not (only_a & only_b).any()
    assert np.array_equal(only_a, sm.rasterize_ellipse(sphere(a), SIZE))
    assert both.shape == SIZE and both.dtype == bool


def test_dilation_grows_the_mask_by_the_configured_pixels(cfg, cfg_plain) -> None:
    ev = smokes((1000, 0, 64))
    plain = sm.smoke_mask_for_frame(10, ev, EYE, 0.0, 0.0, cfg_plain)
    dilated = sm.smoke_mask_for_frame(10, ev, EYE, 0.0, 0.0, cfg)
    assert cfg.eval.h2.mask_dilate_px == 2
    assert (dilated & plain).sum() == plain.sum() and dilated.sum() > plain.sum()
    rows, cols = np.where(plain)
    rows2, cols2 = np.where(dilated)
    assert (rows.min() - rows2.min(), rows2.max() - rows.max()) == (2, 2)
    assert (cols.min() - cols2.min(), cols2.max() - cols.max()) == (2, 2)


def test_line_of_sight_filter_drops_smokes_far_from_the_view_axis(cfg_plain) -> None:
    ev = smokes((1000, -600, 64))  # 600 units off the axis
    assert sm.smoke_mask_for_frame(10, ev, EYE, 0.0, 0.0, cfg_plain).any()
    tight = load_config(overrides=["eval.h2.mask_dilate_px=0", "eval.h2.max_los_units=500"])
    assert not sm.smoke_mask_for_frame(10, ev, EYE, 0.0, 0.0, tight).any()


def test_a_synthetic_scene_two_smokes_one_expired(cfg) -> None:
    ev = pd.concat(
        [smokes((1000, 0, 64), detonate=0, expire=100), smokes((900, -500, 64), detonate=50, expire=300)],
        ignore_index=True,
    )
    early = sm.smoke_mask_for_frame(10, ev, EYE, 0.0, 0.0, cfg)
    both = sm.smoke_mask_for_frame(60, ev, EYE, 0.0, 0.0, cfg)
    late = sm.smoke_mask_for_frame(150, ev, EYE, 0.0, 0.0, cfg)
    assert early[:, W // 2].any() and not early[:, -1].any()
    assert both.sum() > max(early.sum(), late.sum()) and (both & early).sum() == early.sum()
    assert not late[:, :170].any() and late[:, 170:230].any()  # only the right-hand smoke is left
    assert not sm.smoke_mask_for_frame(400, ev, EYE, 0.0, 0.0, cfg).any()
    assert not sm.smoke_mask_for_frame(10, ev.iloc[0:0], EYE, 0.0, 0.0, cfg).any()


# --- static masks --------------------------------------------------------------------------------------------------


def test_viewmodel_box_is_a_bottom_fraction_box(cfg) -> None:
    box = sm.viewmodel_mask(cfg, SIZE)
    x0, y0, x1, _ = cfg.eval.h2.viewmodel_box
    assert box[int(y0 * H) + 1, int(x0 * W) + 1] and box[-1, int((x0 + x1) / 2 * W)]
    assert not box[0].any() and not box[:, 0].any() and not box[: int(y0 * H) - 1].any()
    custom = load_config(overrides=["eval.h2.viewmodel_box=[0.0,0.5,0.5,1.0]"])
    assert sm.viewmodel_mask(custom, SIZE).sum() == (H // 2) * (W // 2)


def test_static_mask_is_the_complement_minus_the_viewmodel(cfg) -> None:
    smoke = sm.smoke_mask_for_frame(10, smokes((1000, 0, 64)), EYE, 0.0, 0.0, cfg)
    static = sm.static_mask_for_frame(smoke, np.zeros((0, 3)), EYE, 0.0, 0.0, cfg)
    box = sm.viewmodel_mask(cfg, SIZE)
    assert np.array_equal(static, ~smoke & ~box)
    assert not (static & smoke).any() and not (static & box).any() and static.any()


def test_static_mask_excludes_frames_with_an_enemy_in_the_frustum(cfg) -> None:
    smoke = np.zeros(SIZE, dtype=bool)

    def static(enemy, yaw=0.0):
        return sm.static_mask_for_frame(smoke, np.array([enemy], dtype=float), EYE, yaw, 0.0, cfg)

    assert static([800, 100, 0]) is None  # ahead, a bit to the left
    assert static([-800, 0, 0]) is not None  # directly behind
    assert static([800, 100, 0], yaw=180) is not None  # same enemy once the camera turns away
    assert static([-800, 0, 0], yaw=180) is None
    assert static([3000, 0, 0]) is not None  # in view but beyond eval.h2.enemy_max_dist_units
    assert static([200, 900, 0]) is not None  # 77 degrees to the left: outside the 53 degree half FOV
    assert static([-800, 0, 0]).sum() > 0.5 * H * W


def test_enemy_at_the_frame_edge_uses_the_margin_and_the_head(cfg) -> None:
    # An enemy whose feet are below the frame but whose head is inside still counts.
    assert sm.enemy_in_view(np.array([[300.0, 0.0, -200.0]]), EYE, 0.0, 0.0, cfg, SIZE)
    far_below = np.array([[300.0, 0.0, -2000.0]])
    assert not sm.enemy_in_view(far_below, EYE, 0.0, 0.0, cfg, SIZE)
    assert not sm.enemy_in_view(np.zeros((0, 3)), EYE, 0.0, 0.0, cfg, SIZE)
    # Just outside the right edge but within the margin: x_cam / z = tan(53.13) * (1 + 8 / 140).
    z = 1000.0
    edge_y = -z * math.tan(math.radians(HFOV / 2)) * (1 + 8 / 140)
    assert sm.enemy_in_view(np.array([[z, edge_y, 0.0]]), EYE, 0.0, 0.0, cfg, SIZE)
    assert not sm.enemy_in_view(np.array([[z, edge_y * 1.2, 0.0]]), EYE, 0.0, 0.0, cfg, SIZE)


def test_project_point_pixel_conventions() -> None:
    u, v, depth = sm.project_point(np.array([1000.0, -500.0, 64.0 + 250.0]), EYE, 0.0, 0.0, HFOV, SIZE)
    assert depth == pytest.approx(1000.0)
    assert u == pytest.approx(W / 2 + 140 * 0.5 / math.tan(math.radians(HFOV / 2)))
    assert v < H / 2
    assert sm.project_point(np.array([-1.0, 0.0, 64.0]), EYE, 0.0, 0.0, HFOV, SIZE) is None
