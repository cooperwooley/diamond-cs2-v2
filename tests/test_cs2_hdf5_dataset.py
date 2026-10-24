"""Tests for src/data CS2Hdf5Dataset: v2 variable-length and JPEG episodes, and v1 fixed-length files."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from cs2wm.config import load_config
from cs2wm.pipeline.pack import read_episode, write_episode
from data import CS2Hdf5Dataset, Dataset, DatasetTraverser, Episode, SegmentId

H, W = 150, 280


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides=["pack.fullres_mod=1"])


def frames_for(n: int, seed: int) -> np.ndarray:
    # Smooth content keeps JPEG error small.
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, size=(n, 1, 1, 3))
    ramp = np.linspace(0, 40, W, dtype=np.int64)[None, None, :, None]
    return np.clip(base + ramp, 0, 255).astype(np.uint8) * np.ones((n, H, W, 3), dtype=np.uint8)


def actions_for(n: int, seed: int) -> np.ndarray:
    return (np.random.default_rng(seed).uniform(size=(n, 51)) < 0.1).astype(np.float32)


def write(cfg, episode_id: str, n: int, low_dir: Path, full_dir: Path | None, seed: int = 0):
    meta = {
        "episode_id": episode_id,
        "demo_id": "demo0",
        "kind": "dm",
        "player": None,
        "start_tick": 0,
        "cs2_build": "1",
        "demoparser2_version": "0.42.0",
        "cs2wm_git_sha": "abc1234",
    }
    return write_episode(frames_for(n, seed), actions_for(n, seed), meta, low_dir, cfg, fullres_dir=full_dir)


def expected_obs(frames_bgr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames_bgr).flip(3).permute(0, 3, 1, 2).float().div(255).mul(2).sub(1)


@pytest.fixture
def v2_dirs(tmp_path, cfg):
    low, full = tmp_path / "low", tmp_path / "full"
    for i, n in enumerate([7, 12, 5]):
        write(cfg, f"ep{i}", n, low, full, seed=i)
    return low, full


def test_variable_lengths_from_n_frames(v2_dirs) -> None:
    ds = CS2Hdf5Dataset(v2_dirs[0])
    assert ds.lengths.tolist() == [7, 12, 5]
    assert ds.num_episodes == 3 and len(ds) == ds.num_steps == 24


@pytest.mark.parametrize("which", [0, 1], ids=["raw", "jpeg"])
def test_segment_matches_pack_reader(v2_dirs, which: int) -> None:
    directory = v2_dirs[which]
    ds = CS2Hdf5Dataset(directory)
    frames, actions, attrs = read_episode(directory / "ep1.hdf5")
    assert attrs["encoding"] == ("raw", "jpeg")[which]
    seg = ds[SegmentId(f"{directory.name}/ep1.hdf5", 2, 6)]
    assert seg.obs.shape == (4, 3, *frames.shape[1:3])
    torch.testing.assert_close(seg.obs, expected_obs(frames[2:6]))
    torch.testing.assert_close(seg.act, torch.from_numpy(actions[2:6]))
    assert seg.mask_padding.tolist() == [True] * 4


def test_padding_after_end_and_before_start(v2_dirs) -> None:
    ds = CS2Hdf5Dataset(v2_dirs[0])
    key = f"{v2_dirs[0].name}/ep2.hdf5"  # 5 frames
    right = ds[SegmentId(key, 3, 8)]
    assert right.obs.shape[0] == 5 and right.mask_padding.tolist() == [True, True, False, False, False]
    left = ds[SegmentId(key, -2, 2)]
    assert left.mask_padding.tolist() == [False, False, True, True]
    assert left.id == SegmentId(key, 0, 2)


def test_segment_start_beyond_episode_length_fails(v2_dirs) -> None:
    ds = CS2Hdf5Dataset(v2_dirs[0])
    with pytest.raises(AssertionError):
        ds[SegmentId(f"{v2_dirs[0].name}/ep2.hdf5", 5, 8)]  # ep2 has 5 frames


def test_integer_episode_id_and_load_episode(v2_dirs) -> None:
    ds = CS2Hdf5Dataset(v2_dirs[0])
    ep = ds.load_episode(1)
    frames, actions, _ = read_episode(v2_dirs[0] / "ep1.hdf5")
    assert len(ep) == 12
    torch.testing.assert_close(ep.obs, expected_obs(frames))
    torch.testing.assert_close(ep.act, torch.from_numpy(actions))


def test_natural_file_order(tmp_path, cfg) -> None:
    for name in ("ep10", "ep2", "ep1"):
        write(cfg, name, 3, tmp_path, None)
    ds = CS2Hdf5Dataset(tmp_path)
    assert list(ds._filenames) == [f"{tmp_path.name}/ep{i}.hdf5" for i in (1, 2, 10)]


def test_v1_fixed_length_files(tmp_path) -> None:
    d = tmp_path / "hdf5_dm_july2021_1"
    d.mkdir()
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(1000, 4, 6, 3), dtype=np.uint8)
    with h5py.File(d / "hdf5_dm_july2021_1.hdf5", "w") as f:  # no attrs, as in v1
        for i in range(1000):
            f.create_dataset(f"frame_{i}_x", data=frames[i])
            f.create_dataset(f"frame_{i}_y", data=np.zeros(51, dtype=np.float32))
    ds = CS2Hdf5Dataset(tmp_path)
    assert ds.lengths.tolist() == [1000] and ds.num_steps == 1000
    seg = ds[SegmentId(f"{d.name}/{d.name}.hdf5", 998, 1002)]
    assert seg.mask_padding.tolist() == [True, True, False, False]
    torch.testing.assert_close(seg.obs[:2], expected_obs(frames[998:1000]))


def test_dataset_traverser_without_full_res(tmp_path) -> None:
    """DatasetTraverser must not need a full-res dataset when none is configured (denoiser-only)."""
    ds = Dataset(tmp_path / "ds", None)
    for n in (6, 4):
        ds.add_episode(
            Episode(
                torch.zeros(n, 3, 4, 4),
                torch.zeros(n, 51),
                torch.zeros(n),
                torch.zeros(n, dtype=torch.uint8),
                torch.zeros(n, dtype=torch.uint8),
                {"original_file_id": "unused"},
            )
        )
    batches = list(DatasetTraverser(ds, batch_num_samples=2, chunk_size=3))
    assert sum(b.obs.shape[0] for b in batches) == 3  # the 1-frame tail chunk is dropped
    assert all("full_res" not in info for b in batches for info in b.info)
