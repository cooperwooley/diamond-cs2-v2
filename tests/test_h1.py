"""Tests for the H1 controllability eval (#32): suites, ground-truth validation, batched rollouts, statistics, e2e."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from agent import Agent
from cs2wm import cli
from cs2wm.actions import ACTION_DIM, ActionSpec, decode, encode
from cs2wm.config import load_config
from cs2wm.eval import h1_controllability as h1
from cs2wm.eval import rotation
from cs2wm.pipeline.pack import write_episode
from cs2wm.play.agent import DEFAULT_AGENT_CONFIG_PATH, build_play_agent
from cs2wm.play.world_model import WorldModel, sample_from_noise

H, W = 150, 280
LOW_H, LOW_W = 30, 56


@pytest.fixture(scope="module")
def cfg():
    return load_config(
        overrides=["pack.fullres_mod=1", "eval.h1.gt_pairs=60", "eval.h1.bootstrap=100", "eval.h1.skip_initial_steps=0"]
    )


@pytest.fixture(scope="module")
def spec(cfg):
    return ActionSpec.from_config(cfg)


# --- scripted suites -----------------------------------------------------------------------------------------------


def test_default_suite_has_the_issue_scripts(cfg, spec) -> None:
    scripts = h1.ScriptedSuite.from_config(cfg).scripts(spec)
    names = [s.name for s in scripts]
    assert len(scripts) == 12 + 8 + 4 + 1 == len(set(names))
    assert h1.ScriptedSuite.from_config(cfg) == h1.ScriptedSuite()
    assert names[:2] == ["yaw+2", "yaw-2"] and names[-1] == "noop" and "pitch-20" in names and "key_d" in names


def test_scripts_decode_to_the_commanded_bin(spec) -> None:
    for s in h1.ScriptedSuite().scripts(spec):
        d = decode(s.action, spec)
        assert (s.commanded_yaw_deg, s.commanded_pitch_deg) == (d.mouse_x_deg, d.mouse_y_deg)
        assert not d.lmb and not d.rmb and s.action.shape == (ACTION_DIM,)
        if s.kind == "yaw":
            assert s.commanded_yaw_deg == pytest.approx(int(s.name[3:]) * 0.055) and s.commanded_pitch_deg == 0
            assert d.keys == ()
        elif s.kind == "pitch":
            assert s.commanded_pitch_deg == pytest.approx(int(s.name[5:]) * 0.055) and s.commanded_yaw_deg == 0
        elif s.kind == "key":
            assert d.keys == (s.name[4:],) and s.commanded_yaw_deg == s.commanded_pitch_deg == 0
        else:
            assert d.keys == () and s.commanded_yaw_deg == s.commanded_pitch_deg == 0
    # Signs: +x = turn right, +y = look down.
    by_name = {s.name: s for s in h1.ScriptedSuite().scripts(spec)}
    assert by_name["yaw+10"].commanded_yaw_deg > 0 > by_name["yaw-10"].commanded_yaw_deg
    assert by_name["pitch+10"].commanded_pitch_deg > 0 > by_name["pitch-10"].commanded_pitch_deg
    np.testing.assert_array_equal(by_name["noop"].action, encode([], False, False, 0.0, 0.0, spec))


def test_suite_options_and_invalid_units(spec) -> None:
    small = h1.ScriptedSuite(yaw_units=(4,), pitch_units=(), keys=("w",), rollout_frames=3, noop=False)
    assert [s.name for s in small.scripts(spec)] == ["yaw+4", "yaw-4", "key_w"]
    with pytest.raises(ValueError, match="not a bin centre"):
        h1.ScriptedSuite(yaw_units=(3,)).scripts(spec)
    with pytest.raises(ValueError, match="not a bin centre"):
        h1.ScriptedSuite(pitch_units=(60,)).scripts(spec)  # a mouse-x bin, not a mouse-y bin


# --- statistics ----------------------------------------------------------------------------------------------------


def test_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    v = np.random.default_rng(0).normal(2.0, 1.0, 50)
    a, b = h1.bootstrap_ci(v, 1000, seed=3), h1.bootstrap_ci(v, 1000, seed=3)
    assert a == b and h1.bootstrap_ci(v, 1000, seed=4) != a
    mean, lo, hi = a
    assert mean == pytest.approx(v.mean()) and lo < mean < hi
    assert hi - lo == pytest.approx(2 * 1.96 * v.std() / math.sqrt(len(v)), rel=0.25)
    assert all(math.isnan(x) for x in h1.bootstrap_ci(np.array([np.nan]), 10, 0))
    assert h1.bootstrap_ci(np.full(5, 1.5), 50, 0) == (1.5, 1.5, 1.5)


def test_axis_metrics_known_r_slope_and_sign_accuracy() -> None:
    rng = np.random.default_rng(1)
    cmd = np.repeat([-3.0, -1.0, 0.0, 1.0, 3.0], 200)
    noise = rng.normal(size=cmd.shape)
    design = np.stack([np.ones_like(cmd), cmd], 1)
    noise -= design @ np.linalg.lstsq(design, noise, rcond=None)[0]  # orthogonal to [1, cmd]: exact r below
    noise *= 1.5 / noise.std()
    est = 0.8 * cmd + 0.1 + noise
    m = h1.axis_metrics(cmd, est)
    assert m["pearson_r"] == pytest.approx(0.8 * cmd.std() / math.hypot(0.8 * cmd.std(), 1.5))
    assert m["slope"] == pytest.approx(0.8) and m["intercept"] == pytest.approx(0.1)
    assert m["n"] == 1000 and m["n_nonzero"] == 800


def test_sign_accuracy_ignores_zero_commands_and_counts_zero_estimates_as_wrong() -> None:
    cmd = np.array([0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 2.0])
    est = np.array([5.0, -5.0, 0.3, -0.3, -0.2, 0.0, 0.1])
    m = h1.axis_metrics(cmd, est)
    assert m["sign_accuracy"] == pytest.approx(3 / 5) and m["n_nonzero"] == 5
    assert math.isnan(h1.axis_metrics(np.zeros(5), np.arange(5.0))["sign_accuracy"])
    flat = h1.axis_metrics(np.arange(5.0), np.ones(5))
    assert math.isnan(flat["pearson_r"]) and math.isnan(flat["slope"])


def synthetic_records(spec, n_ctx: int = 6, frames: int = 4):
    """Rows for the yaw/pitch scripts (estimate = 0.9 x commanded + small noise), W/S and no-op."""
    scripts = h1.ScriptedSuite(yaw_units=(4,), pitch_units=(2,), keys=("w", "s"), rollout_frames=frames + 1).scripts(
        spec
    )
    rng = np.random.default_rng(0)
    ctx, scr = np.meshgrid(np.arange(n_ctx), np.arange(len(scripts)))
    rec = {"context": ctx.ravel().repeat(frames), "script": scr.ravel().repeat(frames)}
    rec["frame"] = np.tile(np.arange(1, frames + 1), n_ctx * len(scripts))
    cmd_y = np.array([s.commanded_yaw_deg for s in scripts])[rec["script"]]
    cmd_p = np.array([s.commanded_pitch_deg for s in scripts])[rec["script"]]
    est = {k: 0.9 * c + rng.normal(0, 0.01, len(c)) for k, c in (("yaw", cmd_y), ("pitch", cmd_p))}
    sign = {"key_w": 1.0, "key_s": -1.0}
    rec["divergence"] = np.array([sign.get(scripts[i].name, np.nan) * 0.01 for i in rec["script"]])
    return scripts, rec, est


def test_model_metrics_per_bin_pooled_and_noop_drift(spec) -> None:
    scripts, rec, est = synthetic_records(spec)
    m = h1.model_metrics(rec, est, scripts, 6, 200, 0)
    assert m["yaw"]["pearson_r"] > 0.99 and m["yaw"]["slope"] == pytest.approx(0.9, abs=0.05)
    assert m["yaw"]["sign_accuracy"] == 1.0 and m["yaw"]["n_nonzero"] == 2 * 6 * 4
    assert m["pitch"]["pearson_r"] > 0.95 and m["pitch"]["n_nonzero"] == 2 * 6 * 4
    bins = {b["script"]: b for b in m["per_bin"]}
    assert list(bins) == ["yaw+4", "yaw-4", "pitch+2", "pitch-2", "key_w", "key_s", "noop"]
    assert bins["pitch+2"]["pitch"]["mean"] == pytest.approx(0.9 * 2 * 0.055, abs=0.01)
    up = bins["yaw+4"]
    assert up["yaw"]["mean"] == pytest.approx(0.9 * 4 * 0.055, abs=0.01)
    assert up["yaw"]["ci_lo"] <= up["yaw"]["mean"] <= up["yaw"]["ci_hi"] and up["n_contexts"] == 6
    assert bins["yaw-4"]["yaw"]["mean"] < 0 < bins["yaw+4"]["yaw"]["mean"]
    drift = m["noop_drift"]
    assert 0 < drift["mean_abs_dyaw_deg"]["mean"] < 0.03 and drift["mean_abs_dpitch_deg"]["ci_hi"] < 0.03
    assert m == h1.model_metrics(rec, est, scripts, 6, 200, 0)  # bootstrap is seeded


def test_divergence_metrics_signs(spec) -> None:
    scripts, rec, _ = synthetic_records(spec)
    d = h1.divergence_metrics(rec, scripts, 6, 100, 0)
    assert d["w"]["mean"] == pytest.approx(0.01) and d["w"]["sign_accuracy"] == 1.0
    assert d["s"]["mean"] == pytest.approx(-0.01) and d["s"]["sign_accuracy"] == 1.0
    assert h1.divergence_metrics(rec, scripts[:2], 6, 100, 0) == {}


def test_json_clean_replaces_nan_and_numpy() -> None:
    out = h1._json_clean({"a": float("nan"), "b": np.float32(1.5), "c": (np.int64(2), float("inf")), "d": {"e": 1}})
    assert out == {"a": None, "b": 1.5, "c": [2, None], "d": {"e": 1}}
    json.dumps(out, allow_nan=False)


# --- ground truth: data, validation --------------------------------------------------------------------------------


def periodic_texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tiled = np.tile(rng.random((H, W, 3)).astype(np.float32), (3, 3, 1))  # blur the tiling, keep the middle tile
    img = cv2.GaussianBlur(tiled, (0, 0), 2.5)[H : 2 * H, W : 2 * W]
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)


def shifted(tex: np.ndarray, sx: float, sy: float) -> np.ndarray:
    m = np.float32([[1, 0, sx], [0, 1, sy]])
    return cv2.warpAffine(tex, m, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)


def write_synthetic_episode(root: Path, name: str, cfg, spec, n: int, seed: int, demo: str = "demo0", ok: bool = True):
    """An episode whose frame k -> k+1 content shift is proportional to the labeled bin of action k.

    With ``ok`` False the labels are unrelated to the (static) frames.
    """
    rng = np.random.default_rng(seed)
    tex = periodic_texture(seed)
    slope_x, slope_y = rotation.theoretical_slopes(W, H, float(cfg.eval.h1.hfov_deg), tuple(cfg.resolution.capture))
    yaw = rng.choice(spec.mouse_x_centers_deg[8:15], size=n)  # -10..+10 units
    pitch = rng.choice(spec.mouse_y_centers_deg[4:11], size=n)  # -10..+10 units
    frames, pos = [], np.zeros(2)
    for k in range(n):
        frames.append(shifted(tex, *pos) if ok else tex)
        pos += (-yaw[k] / slope_x, -pitch[k] / slope_y)  # a right turn moves the content left, looking down moves it up
    actions = np.stack([encode([], False, False, yaw[k], pitch[k], spec) for k in range(n)])
    meta = {
        "episode_id": name,
        "demo_id": demo,
        "kind": "dm",
        "player": None,
        "start_tick": 0,
        "cs2_build": "1",
        "demoparser2_version": "0.42.0",
        "cs2wm_git_sha": "abc1234",
    }
    write_episode(np.stack(frames), actions, meta, root / "low", cfg, fullres_dir=root / "full")
    return yaw, pitch


@pytest.fixture(scope="module")
def gt_root(tmp_path_factory, cfg, spec):
    root = tmp_path_factory.mktemp("gt")
    for i in range(3):
        write_synthetic_episode(root, f"ep{i}", cfg, spec, 30, seed=i, demo=f"demo{i % 2}")
    return root


def test_validate_estimator_recovers_the_theoretical_slope_on_clean_synthetic_gt(gt_root, cfg) -> None:
    low, full = h1.list_episodes(gt_root / "low"), h1.list_episodes(gt_root / "full")
    val = h1.validate_estimator(low, full, cfg, seed=0)
    assert val["passed"] is True and val["threshold_r"] == 0.8 and val["estimator"] == "phasecorr"
    hi = val["resolutions"]["150x280"]
    assert hi["available"] and hi["n_pairs"] == 60
    for axis in ("yaw", "pitch"):
        fit = hi["estimators"]["phasecorr"][axis]
        assert fit["r"] > 0.99 and fit["passed"] is True
        assert fit["slope_deg_per_px"] == pytest.approx(fit["theoretical_slope_deg_per_px"], rel=0.1)
        assert fit["n"] == fit["n_all"]
        flow = hi["estimators"]["flow"][axis]
        assert flow["r"] > 0.9
    lo = val["resolutions"]["30x56"]["estimators"]["phasecorr"]["yaw"]
    assert lo["r"] > 0.8 and lo["slope_deg_per_px"] > 0
    # A 30x56 pixel is 5x as large and sub-pixel shifts are underestimated there: the fitted slope is larger still.
    assert lo["slope_deg_per_px"] > 4 * hi["estimators"]["phasecorr"]["yaw"]["slope_deg_per_px"]


def test_validate_estimator_sign_pins_label_direction(gt_root, cfg) -> None:
    """Right turn (+yaw label) gives a negative dx: the fitted slope would be negative if the sign were wrong."""
    gt = h1.collect_gt_shifts(h1.list_episodes(gt_root / "full"), 40, 0, ActionSpec.from_config(cfg))
    dx, dy = gt["shifts"]["phasecorr"]
    assert np.corrcoef(dx, gt["label_yaw"])[0, 1] < -0.99 and np.corrcoef(dy, gt["label_pitch"])[0, 1] < -0.99


def test_validation_without_fullres_falls_back_to_theory(gt_root, cfg) -> None:
    val = h1.validate_estimator(h1.list_episodes(gt_root / "low"), None, cfg, seed=0)
    assert val["resolutions"]["150x280"]["available"] is False
    conv, warnings = h1.build_conversion(val, cfg)
    th = rotation.theoretical_slopes(W, H, 106.26, (720, 1280))
    assert conv["fullres"] == {
        "yaw_deg_per_px": pytest.approx(th[0]),
        "pitch_deg_per_px": pytest.approx(th[1]),
        "yaw_source": "theory",
        "pitch_source": "theory",
    }
    assert conv["lowres"]["yaw_source"] == "fitted" and conv["lowres"]["yaw_deg_per_px"] > 0 and warnings == []


def test_failed_validation_is_flagged(tmp_path, cfg, spec) -> None:
    for i in range(2):
        write_synthetic_episode(tmp_path, f"bad{i}", cfg, spec, 20, seed=i, ok=False)
    val = h1.validate_estimator(h1.list_episodes(tmp_path / "low"), h1.list_episodes(tmp_path / "full"), cfg, 0)
    assert val["passed"] is False
    fit = val["resolutions"]["150x280"]["estimators"]["phasecorr"]["yaw"]
    assert not fit["passed"]
    conv, warnings = h1.build_conversion(val, cfg)
    assert conv["fullres"]["yaw_source"] == "theory" or fit["slope_deg_per_px"] > 0
    if conv["fullres"]["yaw_source"] == "theory":
        assert any("unusable" in w for w in warnings)


def test_fit_axis_range_filter() -> None:
    label = np.array([-1.0, 0.0, 1.0, 2.0, 40.0])
    px = np.array([-2.0, 0.0, 2.0, 4.0, -10.0])  # the outlier (40 deg) is excluded from the fit by max_abs_deg
    fit = h1.fit_axis(label, px, 5.0)
    assert fit["n"] == 4 and fit["n_all"] == 5 and fit["r"] == pytest.approx(1.0) and fit["slope_deg_per_px"] == 0.5
    assert fit["r_all"] < 0.5


def test_list_episodes_split_json(gt_root, tmp_path) -> None:
    low = gt_root / "low"
    assert [p.stem for p in h1.list_episodes(low)] == ["ep0", "ep1", "ep2"]
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"demo0": "test", "demo1": "train"}))
    assert [p.stem for p in h1.list_episodes(low, split)] == ["ep0", "ep2"]
    split.write_text(json.dumps({"train": ["demo0"], "test": ["demo1"]}))
    assert [p.stem for p in h1.list_episodes(low, split)] == ["ep1"]
    split.write_text(json.dumps({"demo0": "train"}))
    with pytest.raises(ValueError, match="test demo"):
        h1.list_episodes(low, split)
    with pytest.raises(FileNotFoundError):
        h1.list_episodes(tmp_path / "nothing")


def test_sample_contexts_deterministic_spread_and_shaped(gt_root) -> None:
    files = h1.list_episodes(gt_root / "low")
    a = h1.sample_contexts(files, 5, 4, seed=1)
    b = h1.sample_contexts(files, 5, 4, seed=1)
    assert [(c.episode, c.start) for c in a] == [(c.episode, c.start) for c in b]
    assert [(c.episode, c.start) for c in a] != [(c.episode, c.start) for c in h1.sample_contexts(files, 5, 4, seed=2)]
    assert {c.episode for c in a[:3]} == {"ep0", "ep1", "ep2"}  # first pass visits every episode
    for c in a:
        assert c.frames.shape == (4, 3, LOW_H, LOW_W) and c.actions.shape == (4, ACTION_DIM)
        assert -1 <= c.frames.min() and c.frames.max() <= 1 and 0 <= c.start <= 26
    torch.testing.assert_close(a[0].frames, b[0].frames)
    with pytest.raises(ValueError, match="at least 1"):
        h1.sample_contexts(files, 0, 4, 0)
    with pytest.raises(ValueError, match="at least 40"):
        h1.sample_contexts(files, 1, 40, 0)


# --- batched rollouts ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agent_cfg(tiny_model_dotlist):
    return OmegaConf.merge(OmegaConf.load(DEFAULT_AGENT_CONFIG_PATH), OmegaConf.from_dotlist(tiny_model_dotlist()))


def make_agent(agent_cfg, seed: int):
    """A tiny agent with random weights. DIAMOND zero-initialises its output layers, which would make an untrained
    model ignore its inputs (and every seed identical), so those are randomised too."""
    torch.manual_seed(seed)
    agent = Agent(instantiate(agent_cfg, num_actions=ACTION_DIM))
    for p in agent.parameters():
        if p.ndim > 1 and not p.abs().max() > 0:
            torch.nn.init.normal_(p, std=0.1)
    return agent.eval().requires_grad_(False)


@pytest.fixture(scope="module")
def agent(agent_cfg):
    return make_agent(agent_cfg, 0)


def gens(*seeds: int) -> list[torch.Generator]:
    return [torch.Generator().manual_seed(s) for s in seeds]


def random_batch(b: int, t: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    frames = torch.rand(b, 4, 3, LOW_H, LOW_W, generator=g) * 2 - 1
    ctx_actions = torch.zeros(b, 4, ACTION_DIM)
    actions = torch.zeros(b, t, ACTION_DIM)
    actions[torch.arange(b), :, torch.arange(b) % 11] = 1.0  # a different key per stream
    actions[:, :, 11 + 2 + 12] = 1.0  # a non-zero mouse bin
    return frames, ctx_actions, actions


@pytest.mark.parametrize("steps", [1, 3])
def test_batched_rollout_equals_independent_rollouts(agent, steps) -> None:
    # The denoiser quantises its output to uint8 levels (2/255 apart), so float differences between a batched and
    # an unbatched convolution can flip a rare pixel by a level (and the flip feeds the next step): compare to within
    # a few levels on at most a few percent of the pixels.
    world = WorldModel(agent, steps, use_upsampler=True)
    frames, ctx_actions, actions = random_batch(3, 3)
    low, full = world.rollout(frames, ctx_actions, actions, gens(11, 12, 13))
    assert low.shape == (3, 3, 3, LOW_H, LOW_W) and full.shape == (3, 3, 3, H, W)
    for i in range(3):
        low_i, full_i = world.rollout(frames[i : i + 1], ctx_actions[i : i + 1], actions[i : i + 1], gens(11 + i))
        for got, want in ((low[i : i + 1], low_i), (full[i : i + 1], full_i)):
            diff = (got - want).abs()
            assert diff.max() <= 4 * 2 / 255 and (diff > 1e-4).float().mean() < 0.03
    # A stream's noise (and so its frames) does not depend on its neighbours or its position in the batch.
    low_r, _ = world.rollout(frames.flip(0), ctx_actions.flip(0), actions.flip(0), gens(13, 12, 11))
    assert (low_r.flip(0) - low).abs().max() <= 4 * 2 / 255


def test_rollout_depends_on_the_action_and_the_model(agent, agent_cfg) -> None:
    """Guards the tests above against passing vacuously with a model that ignores its inputs."""
    world = WorldModel(agent, 1, use_upsampler=True)
    frames, ctx_actions, actions = random_batch(1, 2)
    base = world.rollout(frames, ctx_actions, actions, gens(1))[0]
    other_action = actions.clone()
    other_action[:, :, 11 + 2 + 12] = 0.0
    other_action[:, :, 11 + 2 + 20] = 1.0
    assert not torch.allclose(base, world.rollout(frames, ctx_actions, other_action, gens(1))[0], atol=1e-3)
    other = WorldModel(make_agent(agent_cfg, 99), 1, use_upsampler=True)
    assert not torch.allclose(base, other.rollout(frames, ctx_actions, actions, gens(1))[0], atol=1e-3)


def test_rollout_noise_is_seeded_per_stream_and_changes_with_the_seed(agent) -> None:
    world = WorldModel(agent, 1, use_upsampler=True)
    frames, ctx_actions, actions = random_batch(2, 2)
    a = world.rollout(frames, ctx_actions, actions, gens(1, 2))[1]
    assert torch.equal(a, world.rollout(frames, ctx_actions, actions, gens(1, 2))[1])
    assert not torch.equal(a, world.rollout(frames, ctx_actions, actions, gens(1, 3))[1])
    assert torch.equal(a[0], world.rollout(frames, ctx_actions, actions, gens(1, 3))[1][0])


class Replay:
    """A ``randn`` that hands out pre-drawn noise in order."""

    def __init__(self, tensors):
        self.tensors = list(tensors)

    def __call__(self, *shape):
        t = self.tensors.pop(0)
        assert t.shape[1:] == shape
        return t


@pytest.mark.parametrize("steps", [1, 3])
def test_noise_driven_step_equals_the_stock_world_model_step(agent, steps) -> None:
    """Same maths as ``DiffusionSampler.sample``, including the conditioning noise: B=1 vs ``WorldModel.step``."""
    world = WorldModel(agent, steps, use_upsampler=True)
    frames, _, _ = random_batch(1, 1)
    acts = np.zeros((4, ACTION_DIM), np.float32)
    action = np.zeros(ACTION_DIM, np.float32)
    action[0] = action[11 + 2 + 5] = 1.0
    world.reset(frames[0], acts)
    # Draw order of the stock path: initial noise, then per denoising step the (discarded) offset noise and the
    # conditioning noise, then the upsampler's initial noise.
    torch.manual_seed(5)
    x0 = torch.randn(1, 3, LOW_H, LOW_W)
    cond = []
    for _ in range(steps):
        torch.randn(1, 12, 1, 1)
        cond.append(torch.randn(1, 12, LOW_H, LOW_W))
    up = torch.randn(1, 3, H, W)
    torch.manual_seed(5)
    expected = world.step(action).clone()
    expected_low = world.low_res.clone()

    ctx_obs, ctx_act = frames[:1].clone(), torch.zeros(1, 4, ACTION_DIM)
    full0 = world._bicubic(frames[:1, -1]).clamp(-1, 1)
    _, _, low, full = world._advance(
        ctx_obs, ctx_act, full0, torch.from_numpy(action)[None], Replay([x0, *cond]), Replay([up])
    )
    torch.testing.assert_close(low[0], expected_low, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(full[0], expected, atol=1e-5, rtol=1e-5)


def test_sample_from_noise_matches_stock_sampler_without_conditioning_noise(agent) -> None:
    from cs2wm.play.world_model import make_upsampler_sampler

    sampler = make_upsampler_sampler(agent.upsampler, 3)  # s_cond = 0: the draw orders coincide
    cond = torch.rand(2, 2, 3, H, W)
    torch.manual_seed(9)
    stock, _ = sampler.sample(cond, None)
    torch.manual_seed(9)
    noise = torch.randn(2, 3, H, W)
    torch.testing.assert_close(sample_from_noise(sampler, cond, None, Replay([noise])), stock)


def test_rollout_leaves_single_stream_state_alone_and_validates(agent) -> None:
    world = WorldModel(agent, 1, use_upsampler=True)
    frames, ctx_actions, actions = random_batch(2, 1)
    with pytest.raises(RuntimeError):
        world.step(np.zeros(ACTION_DIM, np.float32))
    world.rollout(frames, ctx_actions, actions, gens(1, 2))
    assert world._ctx_obs is None
    with pytest.raises(ValueError, match="same batch size"):
        world.rollout(frames, ctx_actions, actions, gens(1))
    with pytest.raises(ValueError, match="need 4 context"):
        world.rollout(frames[:, :2], ctx_actions[:, :2], actions, gens(1, 2))


def test_job_generator_is_stable_per_context_script_and_seed() -> None:
    def draw(*args):
        return torch.randn(3, generator=h1.job_generator(*args))

    assert torch.equal(draw(0, 1, "yaw+2"), draw(0, 1, "yaw+2"))
    for other in ((1, 1, "yaw+2"), (0, 2, "yaw+2"), (0, 1, "yaw-2")):
        assert not torch.equal(draw(0, 1, "yaw+2"), draw(*other))


# --- end to end ----------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory, agent_cfg):
    root = tmp_path_factory.mktemp("ckpts")
    out = {}
    for i, name in enumerate(("v2", "v1", "shuffled")):
        torch.save(make_agent(agent_cfg, i + 1).state_dict(), root / f"{name}.pt")
        out[name] = root / f"{name}.pt"
    return out


@pytest.fixture(scope="module")
def tiny_suite():
    return h1.ScriptedSuite(yaw_units=(4,), pitch_units=(2,), keys=("w",), rollout_frames=3)


def run(ckpts, gt_root, out, cfg, agent_cfg, suite, **kw):
    def factory(name, ckpt):
        return build_play_agent(ckpt, ckpt if name == "v1" else ckpts["v1"], agent_cfg=agent_cfg)

    kw = {"contexts": 2, "seed": 0, "batch_size": 5, "device": "cpu", **kw}
    return h1.run_h1(
        dict(ckpts),
        cfg,
        out,
        data_dir=gt_root / "low",
        fullres_dir=gt_root / "full",
        suite=suite,
        agent_factory=factory,
        **kw,
    )


def no_nan_constant(x):
    raise AssertionError(f"non-finite number {x} in summary.json")


def test_run_h1_end_to_end(ckpts, gt_root, tmp_path, cfg, agent_cfg, tiny_suite, capsys) -> None:
    summary = run(ckpts, gt_root, tmp_path, cfg, agent_cfg, tiny_suite)
    on_disk = json.loads((tmp_path / "summary.json").read_text(), parse_constant=no_nan_constant)
    assert on_disk["schema_version"] == 1 and on_disk["seed"] == 0
    assert set(on_disk) >= {"config", "data", "models", "scripts", "estimator_validation", "conversion", "results"}
    assert on_disk["config"]["rollout_frames"] == 3 and on_disk["config"]["contexts"] == 2
    assert on_disk["config"]["hfov_deg"] == 106.26 and on_disk["data"]["n_episodes"] == 3
    assert len(on_disk["data"]["contexts"]) == 2 and len(on_disk["scripts"]) == 6
    assert (
        on_disk["estimator_validation"]["passed"] is True and on_disk["conversion"]["fullres"]["yaw_source"] == "fitted"
    )
    assert set(on_disk["models"]) == set(on_disk["results"]) == {"v2", "v1", "shuffled"}
    assert on_disk["models"]["v1"]["upsampler"]["path"] == str(ckpts["v1"])
    assert (
        on_disk["models"]["v2"]["upsampler"]["path"]
        == str(ckpts["v1"])
        == on_disk["models"]["shuffled"]["upsampler"]["path"]
    )
    assert len(on_disk["models"]["v2"]["checkpoint"]["sha256_16"]) == 16
    assert {"python", "torch", "numpy", "opencv", "scipy", "matplotlib", "git_sha"} <= set(on_disk["versions"])
    for res in on_disk["results"].values():
        for key in ("fullres", "lowres"):
            r = res[key]
            assert set(r) == {"estimator", "yaw", "pitch", "per_bin", "noop_drift"}
            assert {"n", "pearson_r", "slope", "intercept", "sign_accuracy", "n_nonzero"} == set(r["yaw"])
            assert r["yaw"]["n"] == 2 * 6 * 2 and r["yaw"]["n_nonzero"] == 2 * 2 * 2 and len(r["per_bin"]) == 6
            assert {"mean", "ci_lo", "ci_hi"} == set(r["per_bin"][0]["yaw"])
            assert set(r["noop_drift"]) == {"mean_abs_dyaw_deg", "mean_abs_dpitch_deg"}
        assert set(res["ws_divergence"]) == {"w"}
    assert summary["results"]["v2"]["fullres"]["yaw"]["n"] == 24
    for model in ("v2", "v1", "shuffled"):
        lines = (tmp_path / f"{model}_frames.csv").read_text().splitlines()
        assert len(lines) == 1 + 2 * 6 * 2
        header = lines[0].split(",")
        assert header[:8] == ["context", "episode", "start", "script", "kind", "frame", "cmd_yaw_deg", "cmd_pitch_deg"]
        row = dict(zip(header, lines[1].split(",")))
        assert row["script"] == "yaw+4" and float(row["cmd_yaw_deg"]) == pytest.approx(0.22)
        assert float(row["est_yaw_deg"]) == pytest.approx(
            -on_disk["conversion"]["fullres"]["yaw_deg_per_px"] * float(row["dx_px"])
        )
        assert row["divergence"] == ""
        assert any(line.split(",")[3] == "key_w" and line.split(",")[-1] != "" for line in lines[1:])
        for ext in ("pdf", "png"):
            assert (tmp_path / f"{model}_scatter.{ext}").stat().st_size > 0
    for ext in ("pdf", "png"):
        assert (tmp_path / f"per_bin_curves.{ext}").stat().st_size > 0
    assert set(on_disk["files"]) >= {"summary.json", "v2_frames.csv", "per_bin_curves.pdf", "shuffled_scatter.png"}
    assert "v2" in capsys.readouterr().out


def test_run_h1_is_reproducible_and_seed_dependent(ckpts, gt_root, tmp_path, cfg, agent_cfg, tiny_suite) -> None:
    a = run(ckpts, gt_root, tmp_path / "a", cfg, agent_cfg, tiny_suite)
    b = run(ckpts, gt_root, tmp_path / "b", cfg, agent_cfg, tiny_suite)
    c = run(ckpts, gt_root, tmp_path / "c", cfg, agent_cfg, tiny_suite, seed=1)
    assert a["results"] == b["results"] and a["estimator_validation"] == b["estimator_validation"]
    assert (tmp_path / "a" / "v2_frames.csv").read_text() == (tmp_path / "b" / "v2_frames.csv").read_text()
    assert a["results"] != c["results"]


def test_batch_size_does_not_change_results(ckpts, gt_root, tmp_path, cfg, agent_cfg, tiny_suite) -> None:
    a = run(ckpts, gt_root, tmp_path / "a", cfg, agent_cfg, tiny_suite, batch_size=1)
    b = run(ckpts, gt_root, tmp_path / "b", cfg, agent_cfg, tiny_suite, batch_size=12)
    ra, rb = a["results"]["v2"]["fullres"], b["results"]["v2"]["fullres"]
    for axis in ("yaw", "pitch"):
        assert ra[axis]["slope"] == pytest.approx(rb[axis]["slope"], abs=0.05)  # see the uint8 note above
    ca = (tmp_path / "a" / "v2_frames.csv").read_text().splitlines()
    cb = (tmp_path / "b" / "v2_frames.csv").read_text().splitlines()
    assert [r.split(",")[:6] for r in ca] == [r.split(",")[:6] for r in cb]


def test_run_h1_flags_a_failed_validation_and_errors_on_bad_input(
    ckpts, tmp_path, cfg, agent_cfg, tiny_suite, spec, capsys
) -> None:
    for i in range(2):
        write_synthetic_episode(tmp_path / "bad", f"bad{i}", cfg, spec, 12, seed=i, ok=False)
    with pytest.raises(ValueError, match="data_dir"):
        h1.run_h1(dict(ckpts), cfg, tmp_path / "o")
    with pytest.raises(ValueError, match="rollout_frames"):
        run(ckpts, tmp_path / "bad", tmp_path / "o", cfg, agent_cfg, h1.ScriptedSuite(rollout_frames=1))
    summary = run(ckpts, tmp_path / "bad", tmp_path / "o", cfg, agent_cfg, tiny_suite)
    assert summary["estimator_validation"]["passed"] is False
    assert any("ESTIMATOR VALIDATION FAILED" in w for w in summary["warnings"])
    assert "ESTIMATOR VALIDATION FAILED" in capsys.readouterr().err
    assert json.loads((tmp_path / "o" / "summary.json").read_text())["warnings"] == summary["warnings"]


def test_run_h1_without_fullres_data_uses_theory_and_says_so(
    ckpts, gt_root, tmp_path, cfg, agent_cfg, tiny_suite
) -> None:
    def factory(name, ckpt):
        return build_play_agent(ckpt, ckpts["v1"], agent_cfg=agent_cfg)

    s = h1.run_h1(
        {"v2": ckpts["v2"]}, cfg, tmp_path, 1, 0, data_dir=gt_root / "low", suite=tiny_suite,
        agent_factory=factory, device="cpu", batch_size=4,
    )  # fmt: skip
    assert s["conversion"]["fullres"]["yaw_source"] == "theory"
    assert any("150x280" in w and "not validated" in w for w in s["warnings"])
    assert s["estimator_validation"]["passed"] is True  # only the 30x56 ground truth gates


def test_default_factory_gives_v1_its_own_upsampler(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(h1, "build_play_agent", lambda ckpt, up, device: calls.append((ckpt, up, device)) or "agent")
    factory = h1.make_agent_factory(Path("v1.pt"), torch.device("cpu"))
    assert factory("v2", Path("v2.pt")) == "agent" and factory("shuffled", Path("s.pt")) == "agent"
    assert factory("v1", Path("v1.pt")) == "agent"
    assert calls == [
        (Path(c), Path(u), torch.device("cpu")) for c, u in (("v2.pt", "v1.pt"), ("s.pt", "v1.pt"), ("v1.pt", "v1.pt"))
    ]
    with pytest.raises(ValueError, match="--upsampler-ckpt"):
        h1.make_agent_factory(None, torch.device("cpu"))("v2", Path("v2.pt"))
    assert h1.make_agent_factory(None, torch.device("cpu"))("v1", Path("v1.pt")) == "agent"


# --- cli -----------------------------------------------------------------------------------------------------------


def test_cli_h1_passes_options(monkeypatch, tmp_path) -> None:
    seen = {}

    def fake_run(models, cfg, out, contexts, seed, **kw):
        seen.update(models=models, out=out, contexts=contexts, seed=seed, cfg=cfg, **kw)
        return {}

    monkeypatch.setattr(h1, "run_h1", fake_run)
    argv = ["eval", "h1", "--models", "v2=a.pt", "v1=b.pt", "shuffled=c.pt", "--contexts", "7", "--seed", "3"]
    argv += ["--data", "d", "--split-json", "s.json", "--fullres-data", "f", "--upsampler-ckpt", "u.pt"]
    argv += ["--out", str(tmp_path), "--device", "cpu", "--batch-size", "4", "--denoising-steps", "3"]
    assert cli.main(argv) == 0
    assert seen["models"] == {"v2": Path("a.pt"), "v1": Path("b.pt"), "shuffled": Path("c.pt")}
    assert (seen["contexts"], seen["seed"], seen["out"]) == (7, 3, tmp_path)
    assert (seen["data_dir"], seen["split_json"], seen["fullres_dir"]) == (Path("d"), Path("s.json"), Path("f"))
    assert (seen["upsampler_ckpt"], seen["device"], seen["batch_size"], seen["denoising_steps"]) == (
        Path("u.pt"),
        "cpu",
        4,
        3,
    )
    assert seen["v1_name"] == "v1"
    cli.main(["eval", "h1", "--models", "v2=a.pt", "--data", "d", "--v1-name", "base"])
    assert seen["v1_name"] == "base"
    assert seen["out"] == Path("results/h1") and seen["contexts"] == 50 and seen["batch_size"] == 16
    assert seen["denoising_steps"] == 1


def test_cli_h1_requires_data() -> None:
    with pytest.raises(SystemExit):
        cli.main(["eval", "h1", "--models", "v2=a.pt"])


# --- calibration: regression dilution ------------------------------------------------------------------------------


def test_calibration_is_not_attenuated_by_noise_in_the_pixel_shift() -> None:
    """Noise on px (r about 0.8) leaves px-on-label calibrated; label-on-px would be about r^2 = 0.64 low."""
    rng = np.random.default_rng(0)
    true_px_per_deg = 2.0
    label = rng.choice(np.array([-3.3, -1.65, -0.55, -0.22, 0.0, 0.0, 0.22, 0.55, 1.65, 3.3]), size=40000)
    clean = true_px_per_deg * label
    px = clean + rng.normal(0, 0.75 * clean.std(), label.shape)
    fit = h1.fit_axis(label, px, max_abs_deg=5.0)
    assert fit["r"] == pytest.approx(0.8, abs=0.02)
    assert fit["slope_deg_per_px"] == pytest.approx(1 / true_px_per_deg, rel=0.03)
    assert fit["slope_px_per_deg"] == pytest.approx(true_px_per_deg, rel=0.03)
    assert fit["slope_deg_per_px_ols_label_on_px"] == pytest.approx(0.64 / true_px_per_deg, rel=0.06)  # diluted
    assert fit["slope_deg_per_px_ols_label_on_px"] < 0.7 * fit["slope_deg_per_px"]
    # Truncating on |label| selects on the regressor and leaves the calibration alone.
    cut = h1.fit_axis(label, px, max_abs_deg=1.0)
    assert cut["n"] < len(label) and cut["slope_deg_per_px"] == pytest.approx(1 / true_px_per_deg, rel=0.05)


def test_rollout_slope_regresses_estimated_on_commanded() -> None:
    """The commanded value is exact, so the slope must not be diluted by noise on the estimate."""
    rng = np.random.default_rng(1)
    cmd = rng.choice(np.array([-3.3, -0.55, 0.0, 0.55, 3.3]), size=40000)
    est = cmd + rng.normal(0, 1.0, cmd.shape)
    assert h1.axis_metrics(cmd, est)["slope"] == pytest.approx(1.0, rel=0.03)  # the reverse regression would give ~0.8


# --- stub worlds through the whole estimation path -----------------------------------------------------------------


class StubWorld:
    """A world model whose generated frames translate by an amount set by the action.

    The action held at step ``t`` moves the content between generated frames ``t - 1`` and ``t`` (a right turn moves it
    left) by ``commanded degrees / deg_per_px``; steps before ``move_from`` do not move at all.
    """

    def __init__(self, spec, deg_per_px, move_from: int = 0, seed: int = 0):
        self.spec, self.deg_per_px, self.move_from = spec, deg_per_px, move_from
        self.tex = periodic_texture(seed)

    def rollout(self, frames, ctx_actions, actions, generators):
        lows, fulls = [], []
        for stream in actions:
            pos, bgr = np.zeros(2), []
            for t, act in enumerate(stream.numpy()):
                d = decode(act, self.spec)
                if t >= self.move_from and t > 0:
                    pos += (-d.mouse_x_deg / self.deg_per_px[0], -d.mouse_y_deg / self.deg_per_px[1])
                bgr.append(shifted(self.tex, *pos))
            full = h1.frames_to_tensor(np.stack(bgr))
            fulls.append(full)
            lows.append(torch.nn.functional.interpolate(full, size=(LOW_H, LOW_W), mode="area"))
        return torch.stack(lows), torch.stack(fulls)


def stub_contexts(n: int) -> list:
    return [
        h1.Context(i, f"ep{i}", 0, torch.zeros(4, 3, LOW_H, LOW_W), np.zeros((4, ACTION_DIM), np.float32))
        for i in range(n)
    ]


def stub_run(world, spec, suite, n_ctx: int = 2):
    scripts = suite.scripts(spec)
    rec = h1.rollout_shifts(world, stub_contexts(n_ctx), scripts, suite.rollout_frames, 0, 4, h1.phasecorr_estimator)
    return scripts, rec


def theory_conversion(cfg):
    th = rotation.theoretical_slopes(W, H, float(cfg.eval.h1.hfov_deg), tuple(cfg.resolution.capture))
    lo = rotation.theoretical_slopes(LOW_W, LOW_H, float(cfg.eval.h1.hfov_deg), tuple(cfg.resolution.capture))
    return th, {
        "fullres": {"yaw_deg_per_px": th[0], "pitch_deg_per_px": th[1]},
        "lowres": {"yaw_deg_per_px": lo[0], "pitch_deg_per_px": lo[1]},
    }


def test_a_perfectly_controllable_stub_is_positive_and_has_slope_one(cfg, spec) -> None:
    """Right turn -> content moves left -> positive d-yaw: sign, alignment and calibration end to end."""
    th, conversion = theory_conversion(cfg)
    suite = h1.ScriptedSuite(yaw_units=(4, 10, 60), pitch_units=(4, 10), keys=(), rollout_frames=6)
    scripts, rec = stub_run(StubWorld(spec, th), spec, suite)
    est = h1._estimated_degrees(rec, conversion)["fullres"]
    yaw, pitch = (
        h1.axis_metrics(_cmd(rec, scripts, "yaw"), est["yaw"]),
        h1.axis_metrics(_cmd(rec, scripts, "pitch"), est["pitch"]),
    )
    for m in (yaw, pitch):
        assert m["pearson_r"] > 0.99 and m["sign_accuracy"] == 1.0
        assert m["slope"] == pytest.approx(1.0, abs=0.08)
    assert yaw["slope"] > 0 and yaw["n_nonzero"] == 6 * 2 * 5
    right = rec["script"] == [s.name for s in scripts].index("yaw+10")
    assert (rec["dx"][right] < -0.8).all() and (est["yaw"][right] > 0).all()  # a right turn is content moving left
    down = rec["script"] == [s.name for s in scripts].index("pitch+10")
    assert (rec["dy"][down] < -0.3).all() and (est["pitch"][down] > 0).all()  # looking down moves it up


def _cmd(rec, scripts, axis):
    return np.array([getattr(s, f"commanded_{axis}_deg") for s in scripts])[rec["script"]]


def test_a_stub_that_starts_moving_at_step_t_pins_the_pair_to_the_action(cfg, spec) -> None:
    """Pair ``frame = t`` is generated frames t-1 -> t, driven by the action at step t."""
    th, _ = theory_conversion(cfg)
    suite = h1.ScriptedSuite(yaw_units=(60,), pitch_units=(), keys=(), rollout_frames=7, noop=False)
    _, rec = stub_run(StubWorld(spec, th, move_from=3), spec, suite, n_ctx=1)
    right = rec["script"] == 0
    still, moving = right & (rec["frame"] < 3), right & (rec["frame"] >= 3)
    assert still.sum() == 2 and moving.sum() == 4
    assert (np.abs(rec["dx"][still]) < 0.2).all()
    assert (rec["dx"][moving] < -4).all()  # 60 units = 3.3 deg = about 6 px per step
    left = rec["script"] == 1
    assert (rec["dx"][left & (rec["frame"] >= 3)] > 4).all() and (
        np.abs(rec["dx"][left & (rec["frame"] < 3)]) < 0.2
    ).all()


# --- skip_initial_steps, held_out, v1 name, GT sampling ------------------------------------------------------------


def test_skip_initial_steps_excludes_early_pairs_from_the_metrics_only(ckpts, gt_root, tmp_path, agent_cfg) -> None:
    cfg = load_config(
        overrides=["pack.fullres_mod=1", "eval.h1.gt_pairs=30", "eval.h1.bootstrap=20", "eval.h1.skip_initial_steps=2"]
    )
    suite = h1.ScriptedSuite(yaw_units=(4,), pitch_units=(), keys=(), rollout_frames=4)
    s = run(ckpts, gt_root, tmp_path, cfg, agent_cfg, suite)
    assert s["config"]["skip_initial_steps"] == 2
    assert s["results"]["v2"]["fullres"]["yaw"]["n"] == 2 * 3 * 2  # contexts x scripts x frames 2..3
    lines = [row.split(",") for row in (tmp_path / "v2_frames.csv").read_text().splitlines()]
    header, rows = lines[0], lines[1:]
    assert len(rows) == 2 * 3 * 3  # every pair stays in the CSV
    assert {int(r[header.index("frame")]): int(r[header.index("in_stats")]) for r in rows} == {1: 0, 2: 1, 3: 1}
    too_much = load_config(overrides=["eval.h1.skip_initial_steps=4"])
    with pytest.raises(ValueError, match="skip_initial_steps"):
        run(ckpts, gt_root, tmp_path / "x", too_much, agent_cfg, suite)


def test_default_skip_is_the_conditioning_length() -> None:
    assert load_config().eval.h1.skip_initial_steps == 4


def test_missing_split_json_is_flagged_not_held_out(
    ckpts, gt_root, tmp_path, cfg, agent_cfg, tiny_suite, capsys
) -> None:
    s = run(ckpts, gt_root, tmp_path / "a", cfg, agent_cfg, tiny_suite)
    assert s["data"]["held_out"] is False and any("NOT HELD OUT" in w for w in s["warnings"])
    assert "NOT HELD OUT" in capsys.readouterr().err
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"demo0": "test", "demo1": "test"}))
    s = run(ckpts, gt_root, tmp_path / "b", cfg, agent_cfg, tiny_suite, split_json=split)
    assert s["data"]["held_out"] is True and not any("NOT HELD OUT" in w for w in s["warnings"])
    assert json.loads((tmp_path / "b" / "summary.json").read_text())["data"]["held_out"] is True


def test_v1_name_selects_the_own_upsampler_model_and_is_printed(
    ckpts, gt_root, tmp_path, cfg, agent_cfg, tiny_suite, capsys
) -> None:
    models = {"base": ckpts["v1"], "v2": ckpts["v2"]}
    calls = []

    def factory(name, ckpt):
        calls.append(name)
        return build_play_agent(ckpt, ckpts["v1"], agent_cfg=agent_cfg)

    s = h1.run_h1(models, cfg, tmp_path, 1, 0, data_dir=gt_root / "low", suite=tiny_suite, agent_factory=factory,
                  device="cpu", v1_name="base")  # fmt: skip
    assert s["models"]["base"]["upsampler"]["path"] == str(ckpts["v1"]) == s["models"]["v2"]["upsampler"]["path"]
    out = capsys.readouterr().out
    assert f"base: denoiser {ckpts['v1']}, upsampler {ckpts['v1']} (its own checkpoint)" in out
    assert f"v2: denoiser {ckpts['v2']}, upsampler {ckpts['v1']}" in out
    # Without a model of that name and without --upsampler-ckpt there is no upsampler for v2: fail before any work.
    with pytest.raises(ValueError, match="upsampler"):
        h1.run_h1(
            {"v2": ckpts["v2"]}, cfg, tmp_path / "e", 1, 0, data_dir=gt_root / "low", suite=tiny_suite, device="cpu"
        )
    assert calls == ["base", "v2"]
    seen = []
    monkeypatched = h1.build_play_agent
    try:
        h1.build_play_agent = lambda ckpt, up, device: seen.append((ckpt, up)) or "agent"
        h1.make_agent_factory(Path("u.pt"), torch.device("cpu"), v1_name="base")("base", Path("b.pt"))
        h1.make_agent_factory(Path("u.pt"), torch.device("cpu"), v1_name="base")("v1", Path("x.pt"))
    finally:
        h1.build_play_agent = monkeypatched
    assert seen == [(Path("b.pt"), Path("b.pt")), (Path("x.pt"), Path("u.pt"))]


def test_allocate_pairs_is_an_equal_share_capped_by_capacity() -> None:
    rng = np.random.default_rng(0)
    assert h1.allocate_pairs([10, 10, 10, 10], 12, rng).tolist() == [3, 3, 3, 3]
    assert h1.allocate_pairs([2, 50, 50], 30, rng).tolist() == [2, 14, 14]
    assert h1.allocate_pairs([3, 3], 100, rng).tolist() == [3, 3]
    assert h1.allocate_pairs([0, 5], 3, rng).tolist() == [0, 3]
    uneven = np.array([h1.allocate_pairs([9, 9, 9], 7, np.random.default_rng(s)) for s in range(30)])
    assert (uneven.sum(1) == 7).all() and uneven.min() == 2 and uneven.max() == 3
    assert (uneven[:, 0] == 3).any() and (uneven[:, 2] == 3).any()  # the remainder is not given to the first files


def test_collect_gt_shifts_takes_an_equal_share_from_every_episode(gt_root, cfg, spec) -> None:
    files = h1.list_episodes(gt_root / "low")
    gt = h1.collect_gt_shifts(files, 30, 0, spec)
    assert np.bincount(gt["episode"]).tolist() == [10, 10, 10]  # not 29 + 1 + 0 from the first files in order
    again = h1.collect_gt_shifts(files, 30, 0, spec)
    np.testing.assert_array_equal(gt["label_yaw"], again["label_yaw"])
    assert len(h1.collect_gt_shifts(files, 1000, 0, spec)["label_yaw"]) == 3 * 29
