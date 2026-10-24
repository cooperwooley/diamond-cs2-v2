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


def sphere(center, yaw=0.0, pitch=0.0, eye=EYE, radius=R):
    """Exact (undilated) mask of one sphere."""
    return sm.sphere_mask(np.array(center, dtype=float), radius, eye, yaw, pitch, HFOV, SIZE)


def extent(mask):
    """``(left, right, top, bottom)`` of a mask's bounding box in pixels."""
    rows, cols = np.where(mask)
    return cols.min(), cols.max() + 1, rows.min(), rows.max() + 1


def centroid(mask):
    rows, cols = np.where(mask)
    return cols.mean() + 0.5, rows.mean() + 0.5


def reference_mask(center, eye, yaw, pitch, radius=R, step=0.5):
    """Brute force, independent of the camera-space algebra: march every pixel's world-space ray in small steps."""
    f_x, f_y = 140 / math.tan(math.radians(HFOV / 2)), 75 / (math.tan(math.radians(HFOV / 2)) * 720 / 1280)
    fwd, right, up = sm.view_basis(yaw, pitch)
    cols, rows = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = ((cols - W / 2) / f_x)[..., None] * right + (-(rows - H / 2) / f_y)[..., None] * up + fwd
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    eye, center = np.asarray(eye, float), np.asarray(center, float)
    hit = np.zeros((H, W), dtype=bool)
    for t in np.arange(0.0, np.linalg.norm(center - eye) + radius + 1, step):
        hit |= np.linalg.norm(eye + t * d - center, axis=-1) <= radius
    return hit


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


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


def test_smoke_straight_ahead_is_a_centred_disc() -> None:
    m = sphere([1000, 0, 64])
    cx, cy = centroid(m)
    assert (cx, cy) == (pytest.approx(W / 2, abs=0.01), pytest.approx(H / 2, abs=0.01))
    f_x, f_y = 140 / math.tan(math.radians(HFOV / 2)), 75 / (math.tan(math.radians(HFOV / 2)) * 720 / 1280)
    alpha = math.asin(R / 1000)
    left, right, top, bottom = extent(m)
    assert (right - left) / 2 == pytest.approx(f_x * math.tan(alpha), abs=1)
    assert (bottom - top) / 2 == pytest.approx(f_y * math.tan(alpha), abs=1)


def test_smoke_to_the_right_is_on_the_right_half() -> None:
    # Facing +x, the right-hand side is -y in Source.
    m = sphere([1000, -600, 64])
    assert m[:, W // 2 :].any() and not m[:, : W // 2].any() and centroid(m)[1] == pytest.approx(H / 2, abs=0.5)
    left = sphere([1000, 600, 64])  # +y is the left
    assert np.array_equal(left, m[:, ::-1])


def test_yaw_rotates_counter_clockwise() -> None:
    # Facing +y (yaw 90), +x is on the right and -x on the left.
    assert centroid(sphere([600, 1000, 64], yaw=90))[0] > W / 2
    assert centroid(sphere([-600, 1000, 64], yaw=90))[0] < W / 2
    assert centroid(sphere([0, 1000, 64], yaw=90))[0] == pytest.approx(W / 2, abs=0.01)
    assert centroid(sphere([-1000, 0, 64], yaw=180))[0] == pytest.approx(W / 2, abs=0.01)


def test_smoke_above_is_on_the_upper_half() -> None:
    m = sphere([1000, 0, 500])
    assert centroid(m)[0] == pytest.approx(W / 2, abs=0.01) and m[: H // 2].any() and not m[H // 2 :].any()
    assert centroid(sphere([1000, 0, -400]))[1] > H / 2  # rows grow downwards


def test_smoke_behind_the_camera_has_no_mask(cfg) -> None:
    assert not sphere([-1000, 0, 64]).any()
    assert not sphere([-300, 200, 64]).any()
    assert not sm.smoke_mask_for_frame(10, smokes((-1000, 0, 64)), EYE, 0.0, 0.0, cfg).any()
    assert sphere([-1000, 0, 64], yaw=180).any()  # turn around and it is there


def test_looking_down_at_a_smoke_below_is_centred() -> None:
    cx, cy = centroid(sphere([0, 0, -300], pitch=90))
    assert (cx, cy) == (pytest.approx(W / 2, abs=0.01), pytest.approx(H / 2, abs=0.01))
    # Looking down, "up" on the screen is the heading: a smoke ahead of the player is in the upper half.
    assert centroid(sphere([300, 0, -300], pitch=90))[1] < H / 2
    assert centroid(sphere([0, -300, -300], pitch=90))[0] > W / 2  # and -y stays on the right


def test_positive_pitch_looks_down() -> None:
    # A smoke at eye height is above the screen centre once the camera looks down, below it when it looks up.
    assert centroid(sphere([1000, 0, 64], pitch=20))[1] < H / 2
    assert centroid(sphere([1000, 0, 64], pitch=-20))[1] > H / 2
    # The exact angle: looking down 20 degrees puts the horizon 20 degrees above the centre.
    f_y = 75 / (math.tan(math.radians(HFOV / 2)) * 720 / 1280)
    assert H / 2 - centroid(sphere([1000, 0, 64], pitch=20))[1] == pytest.approx(
        f_y * math.tan(math.radians(20)), rel=0.05
    )


def test_smoke_shrinks_with_distance_by_the_angular_radius() -> None:
    def width(d):
        left, right, *_ = extent(sphere([d, 0, 64]))
        return right - left

    assert width(1000) > width(2000)
    assert width(1000) / width(2000) == pytest.approx(  # whole-pixel widths of 30 and 16 px: allow a pixel
        math.tan(math.asin(R / 1000)) / math.tan(math.asin(R / 2000)), rel=0.08
    )
    assert sphere([1000, 0, 64]).sum() > 3 * sphere([2000, 0, 64]).sum()  # area about 1 / distance^2


def test_camera_inside_the_sphere_masks_the_whole_frame(cfg_plain) -> None:
    assert sphere([100, 0, 64]).all()
    assert sm.smoke_mask_for_frame(10, smokes((100, 0, 64)), EYE, 0.0, 0.0, cfg_plain).all()
    assert sm.smoke_mask_for_frame(10, smokes((60, 60, 0)), EYE, 270.0, 45.0, cfg_plain).all()  # any viewing direction


def test_a_sphere_that_crosses_the_camera_plane_covers_its_visible_side() -> None:
    m = sphere([-50.0, -150.0, 64.0])  # outside the sphere (distance 158), centre behind, mostly to the right
    assert m[:, -1].all() and not m[:, 0].any() and m.any()


def test_sphere_outside_the_frustum_has_no_mask() -> None:
    assert not sphere([1000, -3000, 64]).any()  # far to the right of the 106 degree view
    assert not sphere([1000, 0, 3000]).any()


NEAR_CASES = [  # (centre relative to an eye at the origin, yaw, pitch)
    ((150, -150, 100), 0, 0),  # the reviewed 0 px case
    ((100, -200, 60), 0, 0),
    ((60, -200, 0), 0, 0),
    ((150, 0, 0), 0, 0),  # 150 units straight ahead
    ((200, 40, 30), 0, 0),
    ((0, 250, -80), 90, 10),
    ((-50, -150, 64), 0, 0),  # crosses the camera plane
    ((-20, -150, 100), 0, 0),  # crosses the camera plane
    ((300, 0, 0), 0, 0),
    ((0, 0, -250), 0, 90),
    ((180, 120, 90), 20, -15),
]


@pytest.mark.parametrize(("center", "yaw", "pitch"), NEAR_CASES)
def test_near_smokes_match_the_brute_force_reference(center, yaw, pitch) -> None:
    eye = np.zeros(3)
    got = sphere(center, yaw, pitch, eye=eye)
    ref = reference_mask(center, eye, yaw, pitch)
    assert ref.sum() > 300
    assert iou(got, ref) > 0.95, (center, int(got.sum()), int(ref.sum()))


def test_near_smoke_iou_summary_and_reviewed_pixel_counts() -> None:
    eye = np.zeros(3)
    ious = [iou(sphere(c, y, p, eye=eye), reference_mask(c, eye, y, p)) for c, y, p in NEAR_CASES]
    print(f"IoU vs brute force over {len(ious)} near cases: min {min(ious):.4f}, mean {np.mean(ious):.4f}")
    assert min(ious) > 0.95
    # The cases an inscribed-ellipse model got wrong (it gave 0 px for the first): the exact count is the reference's.
    for center in ((150, -150, 100), (100, -200, 60), (60, -200, 0)):
        exact, ref = sphere(center, eye=eye), reference_mask(center, eye, 0, 0)
        print(f"{center}: exact {int(exact.sum())} px, reference {int(ref.sum())} px")
        assert exact.sum() > 2000 and iou(exact, ref) > 0.99


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
    assert np.array_equal(only_a, sphere(a))
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


def test_los_dropped_count_reports_only_smokes_in_the_frustum(cfg_plain) -> None:
    tight = load_config(overrides=["eval.h2.mask_dilate_px=0", "eval.h2.max_los_units=500"])
    ev = smokes((1000, -600, 64), (1000, 0, 64), (1000, -3000, 64))  # off-axis in view, on axis, off-axis out of view

    def n(cfg):
        return sm.los_dropped_count(10, ev, EYE, 0.0, 0.0, cfg)

    assert n(tight) == 1 and n(cfg_plain) == 0  # the third is out of the frustum, so it is not contamination
    assert sm.los_dropped_count(2000, ev, EYE, 0.0, 0.0, tight) == 0  # none active
    only_on_axis = sm.smoke_mask_for_frame(10, ev, EYE, 0.0, 0.0, tight)
    assert np.array_equal(only_on_axis, sm.smoke_mask_for_frame(10, ev.iloc[[1]], EYE, 0.0, 0.0, tight))


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
