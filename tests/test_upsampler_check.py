"""Tests for the v1 upsampler compatibility check (#27): metrics, frame sampling, verdict and a tiny end-to-end run."""

import json
import math

import cv2
import h5py
import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from agent import Agent
from cs2wm import cli
from cs2wm.actions import ACTION_DIM
from cs2wm.config import load_config
from cs2wm.eval import upsampler_check as uc
from cs2wm.pipeline.pack import to_low_res, write_episode
from cs2wm.play.agent import DEFAULT_AGENT_CONFIG_PATH

H, W = 150, 280


def smooth_frames(n: int, seed: int) -> np.ndarray:
    """Smooth BGR content (JPEG-friendly), different per frame so t-1 and t differ."""
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, size=(n, 1, 1, 3))
    ramp = np.linspace(0, 40, W, dtype=np.int64)[None, None, :, None]
    return np.clip(base + ramp, 0, 255).astype(np.uint8) * np.ones((n, H, W, 3), dtype=np.uint8)


def fake_lpips(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().flatten(1).mean(1)


# --- metrics ----------------------------------------------------------------------------------------------------


def test_psnr_known_value_and_identity() -> None:
    x = torch.full((2, 3, 16, 16), 0.5)
    torch.testing.assert_close(uc.psnr(x + 0.1, x), torch.full((2,), 20.0))  # mse 0.01 -> 20 dB
    assert uc.psnr(x, x).tolist() == [100.0, 100.0]


def test_ssim_identical_is_one() -> None:
    x = torch.rand(2, 3, 32, 40)
    torch.testing.assert_close(uc.ssim(x, x), torch.ones(2))


def test_ssim_constant_images_match_closed_form() -> None:
    a, b = 0.5, 0.25
    got = uc.ssim(torch.full((1, 3, 16, 16), a), torch.full((1, 3, 16, 16), b))
    c1 = 0.01**2  # the variance/covariance terms vanish for constant images
    torch.testing.assert_close(got, torch.tensor([(2 * a * b + c1) / (a * a + b * b + c1)]))


def test_ssim_is_symmetric_and_drops_with_noise() -> None:
    torch.manual_seed(0)
    x = torch.rand(3, 3, 32, 32)
    y = (x + 0.2 * torch.randn_like(x)).clamp(0, 1)
    torch.testing.assert_close(uc.ssim(x, y), uc.ssim(y, x))
    assert (uc.ssim(x, y) < 0.9).all()


def test_ssim_gaussian_window_is_normalised() -> None:
    win = uc._gaussian_window(3, torch.device("cpu"))
    assert win.shape == (3, 1, 11, 11)
    torch.testing.assert_close(win[0].sum(), torch.tensor(1.0))
    assert win[0, 0, 5, 5] == win[0].max()


# --- sampling ---------------------------------------------------------------------------------------------------


def test_sample_frame_indices_deterministic_spread_and_valid() -> None:
    lengths = np.array([10, 1, 30, 5])
    got = uc.sample_frame_indices(lengths, 12, seed=3)
    assert got == uc.sample_frame_indices(lengths, 12, seed=3)
    assert got != uc.sample_frame_indices(lengths, 12, seed=4)
    assert len(got) == len(set(got)) == 12
    assert all(1 <= t < lengths[e] for e, t in got)
    assert {e for e, _ in got} == {0, 2, 3}  # the 1-frame episode has no (t-1, t) pair
    assert {e for e, _ in got[:3]} == {0, 2, 3}  # round-robin: every episode shows up in the first pass


def test_sample_frame_indices_caps_at_available_and_rejects_empty() -> None:
    assert len(uc.sample_frame_indices(np.array([3, 4]), 100, seed=0)) == 2 + 3
    with pytest.raises(ValueError, match="at least 2"):
        uc.sample_frame_indices(np.array([1, 1]), 5, seed=0)


# --- verdict ----------------------------------------------------------------------------------------------------


def summary(up: float, bi: float) -> dict:
    return {
        "upsampler": {"lpips": {"mean": up, "std": 0.0}},
        "bicubic": {"lpips": {"mean": bi, "std": 0.0}},
    }


def test_judge_pass() -> None:
    v = uc.judge(summary(0.11, 0.30), summary(0.10, 0.30))
    assert v["passed"] is True and v["summary"].startswith("PASS")
    assert v["criteria"]["v2_vs_v1_lpips"]["ratio"] == pytest.approx(1.1)


def test_judge_boundary_and_better_than_v1_still_pass() -> None:
    assert uc.judge(summary(0.115, 0.3), summary(0.1, 0.3))["passed"] is True  # exactly +15%
    assert uc.judge(summary(0.05, 0.3), summary(0.1, 0.3))["passed"] is True


def test_judge_fails_when_v2_is_much_worse_than_v1() -> None:
    v = uc.judge(summary(0.12, 0.30), summary(0.10, 0.30))
    assert v["passed"] is False and "above v1" in v["summary"]
    assert v["criteria"]["v2_beats_bicubic"]["passed"] is True


def test_judge_fails_when_not_better_than_bicubic() -> None:
    v = uc.judge(summary(0.10, 0.10), summary(0.10, 0.30))
    assert v["passed"] is False and "bicubic" in v["summary"]
    assert v["criteria"]["v2_vs_v1_lpips"]["passed"] is True


def test_judge_without_v1_is_incomplete_unless_it_already_fails() -> None:
    assert uc.judge(summary(0.1, 0.3), None)["passed"] is None
    assert uc.judge(summary(0.1, 0.3), None)["summary"].startswith("INCOMPLETE")
    assert uc.judge(summary(0.3, 0.1), None)["passed"] is False


def test_judge_zero_v1_lpips() -> None:
    assert math.isinf(uc.judge(summary(0.1, 0.3), summary(0.0, 0.3))["criteria"]["v2_vs_v1_lpips"]["ratio"])


# --- end to end -------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agent_cfg(tiny_model_dotlist):
    return OmegaConf.merge(OmegaConf.load(DEFAULT_AGENT_CONFIG_PATH), OmegaConf.from_dotlist(tiny_model_dotlist()))


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory, agent_cfg):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "agent_epoch_00149.pt"
    torch.save(Agent(instantiate(agent_cfg, num_actions=ACTION_DIM)).state_dict(), path)
    return path


@pytest.fixture(scope="module")
def v2_dir(tmp_path_factory):
    cfg = load_config(overrides=["pack.fullres_mod=1"])
    root = tmp_path_factory.mktemp("v2")
    for i, n in enumerate([5, 4, 6]):
        meta = {
            "episode_id": f"ep{i}",
            "demo_id": "demo0",
            "kind": "dm",
            "player": None,
            "start_tick": 0,
            "cs2_build": "1",
            "demoparser2_version": "0.42.0",
            "cs2wm_git_sha": "abc1234",
        }
        actions = np.zeros((n, ACTION_DIM), np.float32)
        write_episode(smooth_frames(n, i), actions, meta, root / "low", cfg, fullres_dir=root / "full")
    return root / "full"


@pytest.fixture(scope="module")
def v1_dir(tmp_path_factory):
    """A v1-layout file: 1000 raw frames, no attrs (compressed so the fixture stays small)."""
    d = tmp_path_factory.mktemp("v1") / "test"
    d.mkdir()
    frames = smooth_frames(1, 9)[0]
    with h5py.File(d / "hdf5_dm_july2021_1.hdf5", "w") as f:
        for i in range(1000):
            f.create_dataset(
                f"frame_{i}_x", data=np.clip(frames.astype(int) + i % 30, 0, 255).astype(np.uint8), compression="gzip"
            )
            f.create_dataset(f"frame_{i}_y", data=np.zeros(ACTION_DIM, np.float32))
    return d


def run(ckpt, v2_dir, v1_dir, out, agent_cfg, **kw):
    kw = {"n": 6, "batch_size": 4, "seed": 0, "device": "cpu", "lpips_fn": fake_lpips, **kw}
    return uc.run_upsampler_check(ckpt, v2_dir, out_dir=out, v1_fullres_dir=v1_dir, agent_cfg=agent_cfg, **kw)


def test_end_to_end_writes_json_and_grid(ckpt, v2_dir, v1_dir, agent_cfg, tmp_path, capsys) -> None:
    report = run(ckpt, v2_dir, v1_dir, tmp_path, agent_cfg)
    on_disk = json.loads((tmp_path / "upsampler_check.json").read_text())
    assert on_disk["v2"]["n"] == 6 and on_disk["v1"]["n"] == 6
    for ds in ("v2", "v1"):
        for method in ("upsampler", "bicubic"):
            for metric in ("psnr", "ssim", "lpips"):
                assert set(on_disk[ds][method][metric]) == {"mean", "std"}
                assert math.isfinite(on_disk[ds][method][metric]["mean"])
    crit = on_disk["verdict"]["criteria"]
    assert crit["v2_vs_v1_lpips"]["ratio"] == pytest.approx(
        on_disk["v2"]["upsampler"]["lpips"]["mean"] / on_disk["v1"]["upsampler"]["lpips"]["mean"]
    )
    assert on_disk["verdict"]["passed"] in (True, False)
    assert on_disk["checkpoint"]["path"] == str(ckpt) and len(on_disk["checkpoint"]["sha256_16"]) == 16
    assert on_disk["seed"] == 0 and on_disk["device"] == "cpu" and on_disk["torch_version"] == torch.__version__
    assert "git_sha" in on_disk and report["v2"] == on_disk["v2"]
    grid = cv2.imread(str(tmp_path / "upsampler_check_grid.png"))
    # header + 2 x (banner + 4 rows) bands; four tiles per row
    assert grid.shape == (26 + 2 * (26 + 4 * H), 4 * W, 3)
    out = capsys.readouterr().out
    assert "upsampler" in out and "bicubic" in out and on_disk["verdict"]["summary"] in out


def test_end_to_end_is_deterministic_per_seed(ckpt, v2_dir, v1_dir, agent_cfg, tmp_path) -> None:
    a = run(ckpt, v2_dir, v1_dir, tmp_path / "a", agent_cfg)
    b = run(ckpt, v2_dir, v1_dir, tmp_path / "b", agent_cfg)
    c = run(ckpt, v2_dir, v1_dir, tmp_path / "c", agent_cfg, seed=1)
    assert a["v2"] == b["v2"] and a["v1"] == b["v1"]
    assert a["v2"]["upsampler"] != c["v2"]["upsampler"]
    assert (tmp_path / "a" / "upsampler_check_grid.png").read_bytes() == (
        tmp_path / "b" / "upsampler_check_grid.png"
    ).read_bytes()


def test_without_v1_only_v2_is_scored(ckpt, v2_dir, agent_cfg, tmp_path) -> None:
    report = run(ckpt, v2_dir, None, tmp_path, agent_cfg)
    assert "v1" not in report and "v2" in report
    assert report["verdict"]["criteria"]["v2_vs_v1_lpips"]["passed"] is None


def test_n_larger_than_dataset_uses_all_valid_frames(ckpt, v2_dir, agent_cfg, tmp_path) -> None:
    report = run(ckpt, v2_dir, None, tmp_path, agent_cfg, n=1000)
    assert report["v2"]["n"] == (5 - 1) + (4 - 1) + (6 - 1)


def test_upsampler_gets_prev_gt_and_bicubic_of_the_recipe_low_res(
    ckpt, v2_dir, agent_cfg, tmp_path, monkeypatch
) -> None:
    from data import CS2Hdf5Dataset
    from models.diffusion import DiffusionSampler

    calls = []
    real = DiffusionSampler.sample

    def spy(self, prev_obs, prev_act):
        calls.append((prev_obs.clone(), prev_act))
        return real(self, prev_obs, prev_act)

    monkeypatch.setattr(DiffusionSampler, "sample", spy)
    run(ckpt, v2_dir, None, tmp_path, agent_cfg, n=5, batch_size=2)
    assert [c[0].shape[0] for c in calls] == [2, 2, 1]

    ds = CS2Hdf5Dataset(v2_dir)
    items = uc.sample_frame_indices(ds.lengths, 5, seed=0)
    cond, act = calls[0]
    assert act is None and cond.shape == (2, 2, 3, H, W)
    for i, (e, t) in enumerate(items[:2]):
        pair = uc._load_pair(ds, e, t)
        assert torch.equal(cond[i, 0], pair[0])  # previous ground-truth full-res frame
        low = uc.frames_to_tensor(to_low_res(uc._to_bgr_uint8(pair[1:])))  # v1 recipe on GT frame t
        expected = torch.nn.functional.interpolate(low, scale_factor=5, mode="bicubic")[0]
        torch.testing.assert_close(cond[i, 1], expected)


def test_bgr_roundtrip_is_exact_for_uint8_frames() -> None:
    bgr = np.random.default_rng(0).integers(0, 256, (2, 8, 9, 3), dtype=np.uint8)
    np.testing.assert_array_equal(uc._to_bgr_uint8(uc.frames_to_tensor(bgr)), bgr)


def test_load_upsampler_agent_requires_an_upsampler(ckpt, agent_cfg) -> None:
    cfg = OmegaConf.merge(agent_cfg, {})
    cfg.upsampler = None
    with pytest.raises(ValueError, match="upsampler"):
        uc.load_upsampler_agent(ckpt, torch.device("cpu"), cfg)


# --- cli --------------------------------------------------------------------------------------------------------


def test_cli_passes_options_and_exits_nonzero_on_fail(monkeypatch, tmp_path) -> None:
    seen = {}

    def fake_run(ckpt, fullres, n, out, **kw):
        seen.update(ckpt=ckpt, fullres=fullres, n=n, out=out, **kw)
        return {"verdict": {"passed": False}}

    monkeypatch.setattr(uc, "run_upsampler_check", fake_run)
    argv = ["eval", "upsampler", "--ckpt", "c.pt", "--fullres", "f", "--n", "8", "--v1-fullres", "v1", "--seed", "3"]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv + ["--out", str(tmp_path), "--device", "cpu", "--batch-size", "2"])
    assert exc.value.code == 1
    assert (seen["n"], seen["seed"], seen["device"], seen["batch_size"]) == (8, 3, "cpu", 2)
    assert str(seen["v1_fullres_dir"]) == "v1" and seen["out"] == tmp_path
    monkeypatch.setattr(uc, "run_upsampler_check", lambda *a, **k: {"verdict": {"passed": True}})
    assert cli.main(argv) == 0
