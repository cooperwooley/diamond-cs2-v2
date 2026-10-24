"""Tests for the carried-over v1 evals (#34): loss curves, rollout grids and strips, FID, and the CLI end to end.

CPU only and offline: tiny random agents, synthetic HDF5 episodes, and a fake FID feature extractor (the real one
downloads the Inception weights).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from agent import Agent
from cs2wm import cli
from cs2wm.actions import ACTION_DIM
from cs2wm.config import load_config
from cs2wm.eval import carried, fid, loss_curve, rollouts
from cs2wm.pipeline.pack import write_episode
from cs2wm.play.agent import DEFAULT_AGENT_CONFIG_PATH
from cs2wm.play.world_model import WorldModel

LOW_H, LOW_W = 30, 56
N_COND = 4
HORIZON = 3
EP_LEN = 12


# --- loss curves ---------------------------------------------------------------------------------------------------


def write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def metric_line(epoch: int, train: float, test: float | None) -> str:
    return json.dumps(
        {"epoch": epoch, "train_loss": train, "test_loss": test, "lr": 1e-4, "duration_s": 1.0, "gpu": "cpu"}
    )


def test_v2_metrics_dedupe_by_epoch_last_line_wins(tmp_path) -> None:
    path = write_lines(
        tmp_path / "metrics.jsonl",
        [
            metric_line(1, 0.9, None),
            metric_line(2, 0.8, 0.7),
            metric_line(2, 0.75, 0.65),  # duplicate after a hard kill: this one wins
            metric_line(3, 0.7, None),
            "",
        ],
    )
    got = loss_curve.load_v2_metrics(path)
    assert got["epochs"] == [1, 2, 3]
    assert got["train_loss"] == [0.9, 0.75, 0.7]
    assert got["test_loss"] == [None, 0.65, None]
    assert (got["n_lines"], got["n_duplicates"]) == (4, 1)


def test_v2_metrics_sorts_out_of_order_epochs(tmp_path) -> None:
    path = write_lines(tmp_path / "m.jsonl", [metric_line(3, 0.3, 0.3), metric_line(1, 0.1, 0.1)])
    assert loss_curve.load_v2_metrics(path)["epochs"] == [1, 3]


def test_v2_metrics_tolerates_a_truncated_last_line_only(tmp_path) -> None:
    good = metric_line(1, 0.9, 0.8)
    got = loss_curve.load_v2_metrics(write_lines(tmp_path / "a.jsonl", [good, '{"epoch": 2, "train_lo']))
    assert got["epochs"] == [1] and got["n_lines"] == 1
    with pytest.raises(ValueError, match="b.jsonl:1: not valid JSON"):
        loss_curve.load_v2_metrics(write_lines(tmp_path / "b.jsonl", ['{"epoch": 2, "tr', good]))


@pytest.mark.parametrize("bad", ['{"train_loss": 1.0}', '{"epoch": "2"}', "[1, 2]", '{"epoch": true}'])
def test_v2_metrics_rejects_lines_without_an_integer_epoch(tmp_path, bad) -> None:
    with pytest.raises(ValueError, match="integer 'epoch'"):
        loss_curve.load_v2_metrics(write_lines(tmp_path / "m.jsonl", [metric_line(1, 1.0, 1.0), bad]))


def test_v2_metrics_empty_file_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="no metrics lines"):
        loss_curve.load_v2_metrics(write_lines(tmp_path / "m.jsonl", [""]))


def test_v1_losses_structure_from_the_committed_jsons(repo_root) -> None:
    v1 = loss_curve.load_v1_losses(repo_root / "results" / "v1")
    # Test loss: already per epoch, the cloud checkpoints v1 evaluated.
    assert v1["test"]["epochs"] == [95, 120, 149]
    assert v1["test"]["denoiser"][-1] == pytest.approx(0.058349620481650974)
    assert v1["test"]["upsampler"][0] == pytest.approx(2.822832097263039)
    # Train loss: 100 per-step values converted to epochs at 400 steps per epoch.
    early = v1["train_early"]
    assert early["n_steps"] == len(early["loss"]) == len(early["epochs"]) == 100
    assert early["epochs"][0] == pytest.approx(1 / 400) and early["epochs"][-1] == pytest.approx(0.25)
    assert v1["steps_per_epoch"] == 400


def test_v1_losses_rejects_other_structures(tmp_path) -> None:
    (tmp_path / "cloud_training_losses.json").write_text(json.dumps({"5": {"upsampler": 1.0}}))
    (tmp_path / "training_losses.json").write_text(json.dumps([1.0, 2.0]))
    with pytest.raises(ValueError, match="no 'denoiser'"):
        loss_curve.load_v1_losses(tmp_path)
    (tmp_path / "cloud_training_losses.json").write_text(json.dumps({"5": {"denoiser": 1.0}}))
    (tmp_path / "training_losses.json").write_text(json.dumps({"a": 1}))
    with pytest.raises(ValueError, match="list of per-step losses"):
        loss_curve.load_v1_losses(tmp_path)


def test_run_loss_curve_writes_figures_and_series(tmp_path, repo_root) -> None:
    metrics = write_lines(
        tmp_path / "metrics.jsonl", [metric_line(e, 1.0 / e, 0.5 / e if e % 2 else None) for e in (1, 2, 3)]
    )
    out = tmp_path / "out"
    res = loss_curve.run_loss_curve(metrics, repo_root / "results" / "v1", out)
    assert res["x_axis"] == "epoch" and res["v2"]["epochs"] == [1, 2, 3]
    assert res["v2"]["test_loss"] == [0.5, None, pytest.approx(0.5 / 3)]
    for name in res["figure"]:
        assert (out / name).stat().st_size > 0
    assert sorted(res["figure"]) == ["loss_curve.pdf", "loss_curve.png"]
    json.dumps(res, allow_nan=False)  # plain JSON types
    alone = loss_curve.run_loss_curve(None, repo_root / "results" / "v1", tmp_path / "alone")
    assert alone["v2"] is None and (tmp_path / "alone" / "loss_curve.png").exists()


# --- FID -----------------------------------------------------------------------------------------------------------


def fake_extractor(images: np.ndarray) -> np.ndarray:
    """Deterministic 24-dim features: per-channel mean, std and 5x2 block means (first 18)."""
    x = images.astype(np.float64) / 255
    n, h, w, _ = x.shape
    blocks = x.reshape(n, 5, h // 5, 2, w // 2, 3).mean(axis=(2, 4)).reshape(n, -1)
    return np.concatenate([x.mean((1, 2)), x.std((1, 2)), blocks[:, :18]], axis=1)


def random_images(n: int, seed: int, offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 200, size=(n, LOW_H, LOW_W, 3)).astype(np.float64) + offset
    return x.clip(0, 255).astype(np.uint8)


def test_fid_identical_sets_is_zero_and_shifted_sets_is_positive() -> None:
    a = random_images(64, 0)
    assert fid.compute_fid(a, a.copy(), fake_extractor) == pytest.approx(0.0, abs=1e-6)
    same_dist = fid.compute_fid(random_images(64, 1), random_images(64, 2), fake_extractor)
    shifted = fid.compute_fid(random_images(64, 1, offset=50), random_images(64, 2), fake_extractor)
    assert shifted > 0 and shifted > 10 * same_dist
    assert fid.compute_fid(a, random_images(64, 3), fake_extractor) == pytest.approx(
        fid.compute_fid(random_images(64, 3), a, fake_extractor), rel=1e-6
    )


def test_fid_matches_pytorch_fid_on_the_features() -> None:
    from pytorch_fid.fid_score import calculate_frechet_distance

    a, b = random_images(40, 4), random_images(40, 5, offset=20)
    fa, fb = fake_extractor(a), fake_extractor(b)
    want = calculate_frechet_distance(fa.mean(0), np.cov(fa, rowvar=False), fb.mean(0), np.cov(fb, rowvar=False))
    assert fid.compute_fid(a, b, fake_extractor) == pytest.approx(want)


def test_fid_input_validation() -> None:
    with pytest.raises(ValueError, match="at least 2 images"):
        fid.compute_fid(random_images(1, 0), random_images(5, 1), fake_extractor)
    with pytest.raises(ValueError, match="feature sizes differ"):
        fid.compute_fid(random_images(5, 0), random_images(6, 1), lambda x: np.random.rand(len(x), len(x)))


def test_to_uint8_round_trips_ground_truth_and_clamps() -> None:
    u = np.arange(256, dtype=np.uint8)
    x = torch.from_numpy(np.tile(u, (3, 1, 1))).float().div(255).mul(2).sub(1)  # [3, 1, 256]
    back = fid.to_uint8(x[None])
    assert back.shape == (1, 1, 256, 3) and (back[0, 0, :, 0] == u).all()
    out = fid.to_uint8(torch.tensor([[[[-3.0]], [[0.0]], [[3.0]]]]))
    assert out[0, 0, 0].tolist() == [0, 128, 255]


# --- test sets, windows and rollouts -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def write_synthetic(root: Path, cfg, name: str, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, size=(n, 150, 280, 3), dtype=np.uint8)
    actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
    actions[np.arange(n), rng.integers(0, 11, size=n)] = 1.0
    meta = {
        "episode_id": name,
        "demo_id": f"demo_{name}",
        "kind": "dm",
        "player": None,
        "start_tick": 0,
        "cs2_build": "1",
        "demoparser2_version": "0.42.0",
        "cs2wm_git_sha": "abc1234",
    }
    write_episode(frames, actions, meta, root, cfg)


@pytest.fixture(scope="module")
def v2_root(tmp_path_factory, cfg) -> Path:
    root = tmp_path_factory.mktemp("v2_test")
    write_synthetic(root, cfg, "ep0", EP_LEN, 0)
    write_synthetic(root, cfg, "ep1", EP_LEN + 3, 1)
    write_synthetic(root, cfg, "short", 5, 2)  # shorter than a window: never sampled
    return root


@pytest.fixture(scope="module")
def v1_root(tmp_path_factory) -> Path:
    """A v1-format test set: a DIAMOND ``Dataset`` directory (``info.pt`` plus ``.pt`` episodes)."""
    from data import Dataset
    from data.episode import Episode

    root = tmp_path_factory.mktemp("v1_test") / "test"
    ds = Dataset(root, None, "v1_test")
    rng = torch.Generator().manual_seed(7)
    for n in (EP_LEN, EP_LEN + 2):
        obs = (torch.rand(n, 3, LOW_H, LOW_W, generator=rng) * 255).byte().float().div(255).mul(2).sub(1)
        ep = Episode(obs, torch.zeros(n, ACTION_DIM), torch.zeros(n), torch.zeros(n, dtype=torch.uint8),
                     torch.zeros(n, dtype=torch.uint8), {})  # fmt: skip
        ds.add_episode(ep)
    ds.save_to_default_path()
    return root


@pytest.fixture(scope="module")
def agent_cfg(tiny_model_dotlist):
    return OmegaConf.merge(OmegaConf.load(DEFAULT_AGENT_CONFIG_PATH), OmegaConf.from_dotlist(tiny_model_dotlist()))


def make_agent(agent_cfg, seed: int) -> Agent:
    """A tiny agent with random weights. DIAMOND zero-initialises its output layers, which would make an untrained
    model ignore its inputs, so those are randomised too."""
    torch.manual_seed(seed)
    agent = Agent(instantiate(agent_cfg, num_actions=ACTION_DIM))
    for p in agent.parameters():
        if p.ndim > 1 and not p.abs().max() > 0:
            torch.nn.init.normal_(p, std=0.1)
    return agent.eval().requires_grad_(False)


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory, agent_cfg) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("ckpts")
    out = {}
    for i, name in enumerate(("v1", "v2")):
        torch.save(make_agent(agent_cfg, i + 1).state_dict(), root / f"{name}.pt")
        out[name] = root / f"{name}.pt"
    return out


def tiny_factory(agent_cfg):
    def factory(name: str, ckpt: Path) -> Agent:
        agent = Agent(instantiate(agent_cfg, num_actions=ACTION_DIM))
        agent.load(ckpt, load_denoiser=True, load_upsampler=False)
        return agent.eval().requires_grad_(False)

    return factory


@pytest.fixture(scope="module")
def worlds(ckpts, agent_cfg) -> dict[str, WorldModel]:
    factory = tiny_factory(agent_cfg)
    return {n: WorldModel(factory(n, p), 1, use_upsampler=False) for n, p in ckpts.items()}


def test_episode_source_reads_both_formats(v2_root, v1_root) -> None:
    v2 = rollouts.EpisodeSource(v2_root)
    assert v2.kind == "hdf5" and v2.names == ["ep0", "ep1", "short"] and v2.lengths == [EP_LEN, EP_LEN + 3, 5]
    frames, actions = v2.load(0)
    assert frames.shape == (EP_LEN, 3, LOW_H, LOW_W) and actions.shape == (EP_LEN, ACTION_DIM)
    assert -1 <= frames.min() and frames.max() <= 1
    v1 = rollouts.EpisodeSource(v1_root)
    assert v1.kind == "diamond" and len(v1) == 2 and v1.lengths == [EP_LEN, EP_LEN + 2]
    frames, actions = v1.load(1)
    assert frames.shape == (EP_LEN + 2, 3, LOW_H, LOW_W) and actions.shape == (EP_LEN + 2, ACTION_DIM)
    with pytest.raises(ValueError, match="split-json"):
        rollouts.EpisodeSource(v1_root, split_json=Path("split.json"))


def test_episode_source_missing_data_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        rollouts.EpisodeSource(tmp_path)


def test_sample_windows_are_deterministic_spread_and_long_enough(v2_root) -> None:
    src = rollouts.EpisodeSource(v2_root)
    length = N_COND + HORIZON
    a, b = rollouts.sample_windows(src, 5, length, seed=1), rollouts.sample_windows(src, 5, length, seed=1)
    assert [(w.episode, w.start) for w in a] == [(w.episode, w.start) for w in b]
    other = rollouts.sample_windows(src, 5, length, seed=2)
    assert [(w.episode, w.start) for w in a] != [(w.episode, w.start) for w in other]
    assert {w.episode for w in a[:2]} == {"ep0", "ep1"}  # first pass visits every eligible episode, never "short"
    assert all(w.episode != "short" for w in a)
    for w in a:
        assert w.frames.shape == (length, 3, LOW_H, LOW_W) and w.actions.shape == (length, ACTION_DIM)
        full, acts = src.load(src.names.index(w.episode))
        torch.testing.assert_close(w.frames, full[w.start : w.start + length])
        np.testing.assert_array_equal(w.actions, acts[w.start : w.start + length])
    with pytest.raises(ValueError, match="at least 1"):
        rollouts.sample_windows(src, 0, length, 0)
    with pytest.raises(ValueError, match="no episode has at least 99"):
        rollouts.sample_windows(src, 1, 99, 0)


def test_generate_rollouts_matches_world_rollout_with_ground_truth_actions(v2_root, worlds) -> None:
    src = rollouts.EpisodeSource(v2_root)
    windows = rollouts.sample_windows(src, 3, N_COND + HORIZON, seed=0)
    world = worlds["v2"]
    got = rollouts.generate_rollouts(world, windows, seed=5, batch_size=2)
    assert got.shape == (3, HORIZON, 3, LOW_H, LOW_W)
    w = windows[1]
    acts = torch.from_numpy(w.actions)[None]
    want, _ = world.rollout(
        w.frames[None, :N_COND], acts[:, :N_COND], acts[:, N_COND - 1 : N_COND - 1 + HORIZON],
        [rollouts.window_generator(5, w.index)],
    )  # fmt: skip
    torch.testing.assert_close(got[1], want[0])


def test_generate_rollouts_is_batch_invariant_seeded_and_model_specific(v2_root, worlds) -> None:
    src = rollouts.EpisodeSource(v2_root)
    windows = rollouts.sample_windows(src, 3, N_COND + HORIZON, seed=0)
    one = rollouts.generate_rollouts(worlds["v2"], windows, 5, batch_size=1)
    three = rollouts.generate_rollouts(worlds["v2"], windows, 5, batch_size=3)
    torch.testing.assert_close(one, three)
    assert not torch.equal(one, rollouts.generate_rollouts(worlds["v2"], windows, 6, batch_size=3))
    assert not torch.equal(one, rollouts.generate_rollouts(worlds["v1"], windows, 5, batch_size=3))
    zero = rollouts.generate_rollouts(worlds["v2"], windows, 5, batch_size=3, zero_actions=True)
    assert not torch.equal(one, zero)


def test_generate_rollouts_rejects_bad_windows(v2_root, worlds) -> None:
    windows = rollouts.sample_windows(rollouts.EpisodeSource(v2_root), 2, N_COND + 2, seed=0)
    ragged = [windows[0], rollouts.Window(1, "x", 0, windows[1].frames[:-1], windows[1].actions[:-1])]
    with pytest.raises(ValueError, match="windows must all have more than"):
        rollouts.generate_rollouts(worlds["v2"], ragged, 0)
    with pytest.raises(ValueError, match="more than n_cond"):
        rollouts.generate_rollouts(
            worlds["v2"], [rollouts.Window(0, "x", 0, windows[0].frames[:N_COND], windows[0].actions[:N_COND])], 0
        )


def test_step_metrics_known_values() -> None:
    gt = torch.zeros(2, 3, 3, 4, 4)  # [0, 1] grey 0.5
    pred = gt.clone()
    pred[:, 1] += 0.2  # step 2 off by 0.2 in [-1, 1], i.e. 0.1 in [0, 1]
    pred[:, 2] += 4.0  # step 3 far out of range: clamped to 1, i.e. 0.5 off in [0, 1]
    m = rollouts.step_metrics(pred, gt)
    assert m["mse"] == pytest.approx([0.0, 0.01, 0.25], abs=1e-7)
    assert m["psnr"][1] == pytest.approx(20.0, abs=1e-4) and m["psnr"][2] == pytest.approx(10 * math.log10(4), abs=1e-4)
    assert m["psnr"][0] == pytest.approx(100.0)  # the 1e-10 floor, not inf


def test_run_rollouts_writes_grid_strips_and_metrics(v2_root, worlds, tmp_path) -> None:
    windows = rollouts.sample_windows(rollouts.EpisodeSource(v2_root), 3, N_COND + HORIZON, seed=0)
    preds = {n: rollouts.generate_rollouts(w, windows, 0) for n, w in worlds.items()}
    res = rollouts.run_rollouts(preds, windows, N_COND, tmp_path, single_step_samples=2, strip_contexts=2)
    assert res["steps_shown"] == [1]  # STRIP_STEPS clipped to the tiny horizon
    assert res["files"] == ["single_step_grid.png", "rollout_strip_00.png", "rollout_strip_01.png",
                            "rollout_metrics.pdf", "rollout_metrics.png"]  # fmt: skip
    for f in res["files"]:
        assert (tmp_path / f).stat().st_size > 0
    cell_w, cell_h = LOW_W * rollouts.SCALE, LOW_H * rollouts.SCALE
    with Image.open(tmp_path / "single_step_grid.png") as grid:  # last context | ground truth | v1 | v2, 2 rows
        assert grid.size == (4 * cell_w, rollouts.LABEL_H + 2 * cell_h)
    with Image.open(tmp_path / "rollout_strip_00.png") as strip:  # row label + context + 1 step; GT + 2 models
        assert strip.size == (rollouts.ROW_LABEL_W + 2 * cell_w, rollouts.LABEL_H + 3 * cell_h)
    m = res["metrics"]
    assert set(m) == {"v1", "v2", "hold_last_context"}
    for v in m.values():
        assert len(v["mse"]) == len(v["psnr"]) == HORIZON
        assert v["single_step"] == {"mse": v["mse"][0], "psnr": v["psnr"][0]}
        assert all(x > 0 for x in v["mse"])
    gt = rollouts.ground_truth(windows, N_COND)
    hold = windows[0].frames[N_COND - 1].expand(1, HORIZON, -1, -1, -1)
    assert m["hold_last_context"]["mse"][0] == pytest.approx(
        np.mean(
            [
                rollouts.step_metrics(w.frames[N_COND - 1][None, None], g[None, :1])["mse"][0]
                for w, g in zip(windows, gt)
            ]
        )
    )
    assert hold.shape[1] == HORIZON
    assert res["windows"] == [{"index": w.index, "episode": w.episode, "start": w.start} for w in windows]
    with pytest.raises(ValueError, match="rollouts for 3 windows"):
        rollouts.run_rollouts({"v1": preds["v1"][:2]}, windows, N_COND, tmp_path)


# --- the whole run -------------------------------------------------------------------------------------------------


@pytest.fixture
def tiny_run(monkeypatch, agent_cfg, ckpts, v2_root, v1_root, tmp_path, repo_root):
    """Patch the real agent builder and the Inception extractor with tiny offline stand-ins."""
    factory = tiny_factory(agent_cfg)
    monkeypatch.setattr(carried, "build_denoiser_agent", lambda ckpt, device: factory("", ckpt))
    monkeypatch.setattr(fid, "make_inception_extractor", lambda device="cpu", batch_size=64: fake_extractor)
    metrics = write_lines(tmp_path / "metrics.jsonl", [metric_line(e, 1.0 / e, 0.5 / e) for e in (1, 2)])
    argv = ["eval", "carried", "--models", f"v1={ckpts['v1']}", f"v2={ckpts['v2']}", "--data", str(v2_root)]
    argv += ["--v1-data", str(v1_root), "--v2-metrics", str(metrics), "--v1-results", str(repo_root / "results/v1")]
    argv += ["--device", "cpu", "--n-fid", "6", "--frames", str(HORIZON), "--contexts", "2", "--batch-size", "2"]
    argv += ["--denoising-steps", "1"]
    return argv


def test_cli_carried_end_to_end(tiny_run, tmp_path) -> None:
    out = tmp_path / "carried"
    assert cli.main([*tiny_run, "--out", str(out), "--seed", "3"]) == 0
    text = (out / "summary.json").read_text()
    summary = json.loads(text, parse_constant=lambda x: pytest.fail(f"non-finite number {x} in summary.json"))
    assert summary["schema_version"] == 1 and summary["seed"] == 3
    assert summary["config"]["frames"] == HORIZON and summary["config"]["n_fid"] == 6
    assert set(summary["models"]) == {"v1", "v2"}
    for m in summary["models"].values():
        assert len(m["checkpoint"]["sha256_16"]) == 16
    assert set(summary["data"]) == {"v2_test", "v1_test"}
    assert summary["data"]["v2_test"]["n_windows"] == summary["data"]["v1_test"]["n_windows"] == 2
    assert summary["data"]["v1_test"]["format"] == "diamond" and summary["data"]["v2_test"]["format"] == "hdf5"
    assert summary["loss_curve"]["v2"]["epochs"] == [1, 2] and summary["loss_curve"]["v1"]["test"]["epochs"] == [
        95,
        120,
        149,
    ]
    # The issue's FID matrix: v1 on the v1 test set, v1 on the v2 test set, v2 on the v2 test set.
    rows = {(r["model"], r["dataset"]): r for r in summary["fid"]["rows"]}
    assert set(rows) == {("v1", "v1_test"), ("v1", "v2_test"), ("v2", "v2_test")}
    for r in rows.values():
        assert r["n_generated"] == r["n_real"] == 6 and r["fid"] >= 0
    assert summary["fid"]["protocol"]["resolution"] == "30x56" and "2048" in summary["fid"]["protocol"]["features"]
    r = summary["rollouts"]
    assert (
        r["dataset"] == "v2_test" and len(r["windows"]) == 2 and set(r["metrics"]) == {"v1", "v2", "hold_last_context"}
    )
    assert {"torch", "pytorch_fid", "git_sha"} <= set(summary["versions"])
    for name in summary["files"]:
        assert (out / name).exists(), name
    assert {"loss_curve.png", "single_step_grid.png", "rollout_metrics.png", "summary.json"} <= set(summary["files"])
    assert summary["warnings"] == []


def test_cli_carried_is_reproducible_and_seed_dependent(tiny_run, tmp_path) -> None:
    def run(name: str, seed: int) -> dict:
        assert cli.main([*tiny_run, "--out", str(tmp_path / name), "--seed", str(seed)]) == 0
        return json.loads((tmp_path / name / "summary.json").read_text())

    a, b, c = run("a", 0), run("b", 0), run("c", 1)
    assert a["fid"] == b["fid"] and a["rollouts"] == b["rollouts"]
    assert a["fid"] != c["fid"]
    assert (tmp_path / "a" / "single_step_grid.png").read_bytes() == (
        tmp_path / "b" / "single_step_grid.png"
    ).read_bytes()


def test_run_carried_without_optional_inputs_warns(
    monkeypatch, agent_cfg, ckpts, v2_root, tmp_path, repo_root, capsys
) -> None:
    summary = carried.run_carried(
        {"v1": ckpts["v1"]}, v2_root, tmp_path, v1_results=repo_root / "results/v1", n_frames=HORIZON, contexts=2,
        n_fid=4, device="cpu", denoising_steps=1, agent_factory=tiny_factory(agent_cfg), extractor=fake_extractor,
    )  # fmt: skip
    assert [(r["model"], r["dataset"]) for r in summary["fid"]["rows"]] == [("v1", "v2_test")]
    assert len(summary["warnings"]) == 2 and summary["loss_curve"]["v2"] is None
    assert "WARNING" in capsys.readouterr().err


def test_run_carried_validates_arguments(ckpts, v2_root, tmp_path) -> None:
    with pytest.raises(ValueError, match="no models"):
        carried.run_carried({}, v2_root, tmp_path)
    with pytest.raises(ValueError, match="at least 1"):
        carried.run_carried(ckpts, v2_root, tmp_path, contexts=0)


def test_cli_carried_requires_data_and_models() -> None:
    with pytest.raises(SystemExit):
        cli.main(["eval", "carried", "--models", "v2=a.pt"])
    with pytest.raises(SystemExit):
        cli.main(["eval", "carried", "--data", "d"])


def test_cli_carried_passes_options(monkeypatch, tmp_path) -> None:
    seen = {}
    monkeypatch.setattr(
        carried, "run_carried", lambda models, data, out, **kw: seen.update(models=models, data=data, out=out, **kw)
    )
    argv = [
        "eval",
        "carried",
        "--models",
        "v1=a.pt",
        "v2=b.pt",
        "--data",
        "d",
        "--v1-data",
        "e",
        "--split-json",
        "s.json",
    ]
    argv += ["--v2-metrics", "m.jsonl", "--v1-results", "r", "--out", str(tmp_path), "--seed", "3", "--device", "cpu"]
    argv += [
        "--batch-size",
        "4",
        "--n-fid",
        "100",
        "--frames",
        "8",
        "--contexts",
        "5",
        "--denoising-steps",
        "1",
        "--zero-actions",
    ]
    assert cli.main(argv) == 0
    assert seen["models"] == {"v1": Path("a.pt"), "v2": Path("b.pt")} and seen["data"] == Path("d")
    assert (seen["v1_data_dir"], seen["split_json"], seen["v2_metrics"], seen["v1_results"]) == (
        Path("e"), Path("s.json"), Path("m.jsonl"), Path("r"))  # fmt: skip
    assert (seen["seed"], seen["device"], seen["batch_size"], seen["n_fid"], seen["n_frames"], seen["contexts"]) == (
        3, "cpu", 4, 100, 8, 5)  # fmt: skip
    assert seen["denoising_steps"] == 1 and seen["zero_actions"] is True
    assert (seen["v1_actions"], seen["v1_name"], seen["s_cond"], seen["fid_seeds"]) == ("zeros", "v1", None, 1)
    argv += ["--v1-actions", "both", "--v1-name", "old", "--s-cond", "0", "--fid-seeds", "3"]
    cli.main(argv)
    assert (seen["v1_actions"], seen["v1_name"], seen["s_cond"], seen["fid_seeds"]) == ("both", "old", 0.0, 3)
    cli.main(["eval", "carried", "--models", "v2=b.pt", "--data", "d"])
    assert seen["out"] == Path("results/carried") and seen["n_fid"] == 2048 and seen["n_frames"] == 64
    assert seen["contexts"] == 16 and seen["denoising_steps"] == 3 and seen["zero_actions"] is False


# --- action source, sampler record, deviations, FID seeds ------------------------------------------------------------


def test_action_sources() -> None:
    assert carried.action_sources("v1", "v1", "zeros", False) == ["zeros"]
    assert carried.action_sources("v2", "v1", "zeros", False) == ["gt"]
    assert carried.action_sources("v1", "v1", "gt", False) == ["gt"]
    assert carried.action_sources("v1", "v1", "both", False) == ["zeros", "gt"]
    assert carried.action_sources("v2", "v1", "both", False) == ["gt"]
    assert carried.action_sources("v2", "v1", "gt", True) == ["zeros"]  # --zero-actions forces zeros for all
    assert carried.action_sources("old", "old", "gt", False) == ["gt"]  # --v1-name


def run_tiny(ckpts, agent_cfg, v2_root, v1_root, tmp_path, **kw) -> dict:
    kw = {"n_frames": HORIZON, "contexts": 2, "n_fid": 6, "device": "cpu", "denoising_steps": 1, **kw}
    return carried.run_carried(
        ckpts, v2_root, tmp_path, v1_results=Path(__file__).resolve().parents[1] / "results/v1", v1_data_dir=v1_root,
        agent_factory=tiny_factory(agent_cfg), extractor=fake_extractor, **kw,
    )  # fmt: skip


def test_default_v1_gets_zero_actions_and_v2_gt(ckpts, agent_cfg, v2_root, v1_root, tmp_path) -> None:
    s = run_tiny(ckpts, agent_cfg, v2_root, v1_root, tmp_path)
    rows = {(r["label"], r["dataset"]): r for r in s["fid"]["rows"]}
    assert {k: v["action_source"] for k, v in rows.items()} == {
        ("v1", "v1_test"): "zeros", ("v1", "v2_test"): "zeros", ("v2", "v2_test"): "gt"}  # fmt: skip
    m = s["rollouts"]["metrics"]
    assert (m["v1"]["action_source"], m["v2"]["action_source"]) == ("zeros", "gt")
    assert "action_source" not in m["hold_last_context"]
    assert "per row" in s["fid"]["protocol"]["generated"]


def test_v1_actions_both_gives_two_labelled_v1_rows(ckpts, agent_cfg, v2_root, v1_root, tmp_path) -> None:
    s = run_tiny(ckpts, agent_cfg, v2_root, v1_root, tmp_path, v1_actions="both")
    labels = {(r["label"], r["dataset"]): r["action_source"] for r in s["fid"]["rows"]}
    assert labels == {
        ("v1 (zero actions)", "v1_test"): "zeros", ("v1 (zero actions)", "v2_test"): "zeros",
        ("v1 (GT actions)", "v1_test"): "gt", ("v1 (GT actions)", "v2_test"): "gt", ("v2", "v2_test"): "gt"}  # fmt: skip
    assert set(s["rollouts"]["metrics"]) == {"v1 (zero actions)", "v1 (GT actions)", "v2", "hold_last_context"}
    assert (tmp_path / "rollout_metrics.png").exists()
    # The same model with different actions generates different frames.
    a, b = (s["rollouts"]["metrics"][k]["mse"] for k in ("v1 (zero actions)", "v1 (GT actions)"))
    assert a != b


def test_zero_actions_flag_forces_zeros_for_every_model(ckpts, agent_cfg, v2_root, v1_root, tmp_path) -> None:
    s = run_tiny(ckpts, agent_cfg, v2_root, v1_root, tmp_path, zero_actions=True)
    assert {r["action_source"] for r in s["fid"]["rows"]} == {"zeros"}


def test_v1_name_selects_the_zero_action_model(ckpts, agent_cfg, v2_root, v1_root, tmp_path) -> None:
    models = {"old": ckpts["v1"], "v2": ckpts["v2"]}
    s = run_tiny(models, agent_cfg, v2_root, v1_root, tmp_path, v1_name="old")
    assert {r["label"]: r["action_source"] for r in s["fid"]["rows"]} == {"old": "zeros", "v2": "gt"}


def test_sampler_record_s_cond_override_and_deviations(ckpts, agent_cfg, v2_root, v1_root, tmp_path) -> None:
    s = run_tiny(ckpts, agent_cfg, v2_root, v1_root, tmp_path / "a")
    sampler = s["config"]["sampler"]
    assert sampler["s_cond"] == 0.005 and sampler["num_steps_denoising"] == 1
    assert (sampler["sigma_min"], sampler["sigma_max"]) == (2e-3, 5.0)
    assert s["fid"]["protocol"]["sampler"] == sampler
    dev = " ".join(s["fid"]["protocol"]["deviations_from_v1"])
    for needle in ("256 frames", "rounded", "PNG round trip", "s_cond", "action source", "noisy"):
        assert needle in dev
    zero = run_tiny(ckpts, agent_cfg, v2_root, v1_root, tmp_path / "b", s_cond=0.0)
    assert zero["config"]["sampler"]["s_cond"] == 0.0 and "is 0.0" in " ".join(
        zero["fid"]["protocol"]["deviations_from_v1"]
    )
    assert zero["rollouts"]["metrics"]["v2"]["mse"] != s["rollouts"]["metrics"]["v2"]["mse"]


def test_fid_seeds_report_mean_and_std(ckpts, agent_cfg, v2_root, v1_root, tmp_path) -> None:
    one = run_tiny(ckpts, agent_cfg, v2_root, v1_root, tmp_path / "a", seed=4)
    assert all(r["fid_std"] is None and r["seeds"] == [4] for r in one["fid"]["rows"])
    three = run_tiny(ckpts, agent_cfg, v2_root, v1_root, tmp_path / "b", seed=4, fid_seeds=3)
    assert three["config"]["fid_seeds"] == [4, 5, 6]
    for r in three["fid"]["rows"]:
        assert r["seeds"] == [4, 5, 6] and len(r["fid_per_seed"]) == 3
        assert r["fid"] == pytest.approx(np.mean(r["fid_per_seed"])) and r["fid_std"] == pytest.approx(
            np.std(r["fid_per_seed"], ddof=1)
        )
    # The first seed is the same run as a single-seed one, and the rollout figures use it.
    by = {(r["label"], r["dataset"]): r for r in one["fid"]["rows"]}
    for r in three["fid"]["rows"]:
        assert r["fid_per_seed"][0] == pytest.approx(by[(r["label"], r["dataset"])]["fid"])
    assert one["rollouts"] == three["rollouts"]
    with pytest.raises(ValueError, match="fid_seeds"):
        run_tiny(ckpts, agent_cfg, v2_root, v1_root, tmp_path / "c", fid_seeds=0)
    with pytest.raises(ValueError, match="v1_actions"):
        run_tiny(ckpts, agent_cfg, v2_root, v1_root, tmp_path / "d", v1_actions="some")


def test_n_fid_beyond_the_episodes_cycles_windows(ckpts, agent_cfg, v2_root, v1_root, tmp_path) -> None:
    # 2 usable episodes, 7 windows of 3 frames: windows revisit episodes, every frame count is still exact.
    s = run_tiny(ckpts, agent_cfg, v2_root, v1_root, tmp_path, n_fid=20)
    assert all(r["n_generated"] == r["n_real"] == 20 and r["n_windows"] == 7 for r in s["fid"]["rows"])
    eps = [w["episode"] for w in s["data"]["v2_test"]["windows"]]
    assert len(eps) == 7 and set(eps) == {"ep0", "ep1"}


# --- readers, alignment, baselines, edge cases ---------------------------------------------------------------------


def test_fid_rejects_mismatched_shape_or_dtype() -> None:
    a = random_images(8, 0)
    with pytest.raises(ValueError, match="differ"):
        fid.compute_fid(a, a[:, :, :-1], fake_extractor)
    with pytest.raises(ValueError, match="differ"):
        fid.compute_fid(a, a.astype(np.float32), fake_extractor)


def test_both_test_set_formats_give_the_same_rgb_tensor(tmp_path) -> None:
    import h5py

    from data import Dataset

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(5, LOW_H, LOW_W, 3), dtype=np.uint8)
    rgb[:, 3, 7] = (255, 0, 0)  # a pure red pixel at row 3, column 7 in every frame
    # v2 pack format: BGR, raw.
    (tmp_path / "h5").mkdir()
    with h5py.File(tmp_path / "h5" / "e.hdf5", "w") as f:
        f.attrs["n_frames"], f.attrs["encoding"] = 5, "raw"
        for i in range(5):
            f[f"frame_{i}_x"] = rgb[i, ..., ::-1]
            f[f"frame_{i}_y"] = np.zeros(ACTION_DIM, dtype=np.float32)
    # v1 DIAMOND format: RGB uint8 [n, 3, h, w] as Episode.save writes it.
    ds = Dataset(tmp_path / "pt", None, "t", save_on_disk=False)
    ds.add_episode(
        __import__("data.episode", fromlist=["Episode"]).Episode(
            torch.zeros(5, 3, LOW_H, LOW_W), torch.zeros(5, ACTION_DIM), torch.zeros(5),
            torch.zeros(5, dtype=torch.uint8), torch.zeros(5, dtype=torch.uint8), {},
        )
    )  # fmt: skip
    path = ds._get_episode_path(0)
    path.parent.mkdir(parents=True)
    torch.save(
        {"obs": torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous(), "act": torch.zeros(5, ACTION_DIM),
         "rew": torch.zeros(5), "end": torch.zeros(5, dtype=torch.uint8), "trunc": torch.zeros(5, dtype=torch.uint8),
         "info": {}}, path,
    )  # fmt: skip
    ds.save_to_default_path()
    a, _ = rollouts.EpisodeSource(tmp_path / "h5").load(0)
    b, _ = rollouts.EpisodeSource(tmp_path / "pt").load(0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert a.shape == (5, 3, LOW_H, LOW_W)
    assert a[2, :, 3, 7].tolist() == [1.0, -1.0, -1.0]  # red in RGB order, not BGR


def test_action_alignment_reaches_the_sampler_as_expected(monkeypatch, worlds) -> None:
    """Window actions carry a one-hot frame marker; the sampler must see actions t..t+n_cond-1 at step t."""
    from cs2wm.play import world_model

    length = N_COND + HORIZON
    windows = []
    for k in range(2):
        acts = np.zeros((length, ACTION_DIM), dtype=np.float32)
        acts[np.arange(length), 10 * k + np.arange(length)] = 1.0  # one-hot marker: window and frame index
        windows.append(rollouts.Window(k, "x", 0, torch.zeros(length, 3, LOW_H, LOW_W), acts))
    seen = []
    real = world_model.sample_from_noise

    def spy(sampler, prev_obs, prev_act, randn):
        seen.append(prev_act.clone())
        return real(sampler, prev_obs, prev_act, randn)

    monkeypatch.setattr(world_model, "sample_from_noise", spy)
    rollouts.generate_rollouts(worlds["v2"], windows, 0, batch_size=2)
    assert len(seen) == HORIZON
    for t, got in enumerate(seen):
        for k in range(2):
            # Step t: the context actions of frames t..t+n_cond-2 and the action taken at frame t+n_cond-1.
            assert got[k].argmax(-1).tolist() == [10 * k + i for i in range(t, t + N_COND)]
            assert (got[k].sum(-1) == 1).all()
    seen.clear()
    rollouts.generate_rollouts(worlds["v2"], windows, 0, batch_size=2, zero_actions=True)
    assert all(not g.any() for g in seen)


def test_hold_last_context_metrics_match_a_hand_computed_value(tmp_path) -> None:
    length = N_COND + 2
    frames = torch.zeros(length, 3, LOW_H, LOW_W)  # 0.0 in [-1, 1] is 0.5 in [0, 1]
    frames[N_COND - 1] = -1.0  # last context: 0.0
    frames[N_COND] = 1.0  # first target: 1.0 -> held frame is off by 1.0
    frames[N_COND + 1] = 0.0  # second target: 0.5 -> off by 0.5
    windows = [rollouts.Window(0, "x", 0, frames, np.zeros((length, ACTION_DIM), dtype=np.float32))]
    res = rollouts.run_rollouts({"m": torch.zeros(1, 2, 3, LOW_H, LOW_W)}, windows, N_COND, tmp_path)
    hold = res["metrics"]["hold_last_context"]
    assert hold["mse"] == pytest.approx([1.0, 0.25])
    assert hold["psnr"] == pytest.approx([0.0, 10 * math.log10(4)], abs=1e-5)
    assert hold["single_step"] == {"mse": 1.0, "psnr": pytest.approx(0.0, abs=1e-5)}
    assert res["metrics"]["m"]["mse"] == pytest.approx([0.25, 0.0])  # predicts 0.5 everywhere


def test_loss_curve_single_truncated_line_and_non_finite_test_loss(tmp_path, repo_root) -> None:
    with pytest.raises(ValueError, match="no metrics lines"):
        loss_curve.load_v2_metrics(write_lines(tmp_path / "a.jsonl", ['{"epoch": 1, "train_lo']))
    lines = [
        '{"epoch": 1, "train_loss": 0.9, "test_loss": NaN}',
        '{"epoch": 2, "train_loss": NaN, "test_loss": null}',
        '{"epoch": 3, "train_loss": 0.5, "test_loss": Infinity}',
        metric_line(4, 0.4, 0.3),
    ]
    got = loss_curve.load_v2_metrics(write_lines(tmp_path / "b.jsonl", lines))
    assert got["test_loss"] == [None, None, None, 0.3] and got["train_loss"] == [0.9, None, 0.5, 0.4]
    res = loss_curve.run_loss_curve(tmp_path / "b.jsonl", repo_root / "results/v1", tmp_path / "o")
    assert (tmp_path / "o" / "loss_curve.png").stat().st_size > 0
    json.dumps(res, allow_nan=False)
    allnull = write_lines(tmp_path / "c.jsonl", ['{"epoch": 1, "train_loss": null, "test_loss": null}'])
    assert loss_curve.run_loss_curve(allnull, repo_root / "results/v1", tmp_path / "p")["v2"]["epochs"] == [1]


def test_loss_curve_train_ranges_are_disjoint_and_noted(repo_root, tmp_path) -> None:
    v1 = loss_curve.load_v1_losses(repo_root / "results" / "v1")
    early = v1["train_early"]
    assert early["mean_epoch"] == pytest.approx(np.mean(early["epochs"])) and early["mean_epoch"] < 0.25
    assert early["mean_loss"] == pytest.approx(np.mean(early["loss"]))
    notes = " ".join(loss_curve.run_loss_curve(None, repo_root / "results" / "v1", tmp_path)["notes"])
    assert "disjoint" in notes
