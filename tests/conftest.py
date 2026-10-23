"""Shared pytest fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from cs2wm.config import load_config
from cs2wm.pipeline.pack import to_low_res, write_episode

REPO_ROOT = Path(__file__).resolve().parent.parent
# src/ is v1 code with flat imports (agent, data, models, ...); put it on the path as src/main.py does.
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import Dataset, Episode

N_EPISODES = {"train": 3, "test": 1}
N_FRAMES = 12  # longer than the denoiser (4 + 1 + 1) and upsampler (1 + 1 + 1) sequence lengths


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


def _tiny_model_dotlist(modules: tuple[str, ...] = ("denoiser", "upsampler")) -> list[str]:
    """OmegaConf dotlist (relative to an agent config) that shrinks the DIAMOND models to a few thousand parameters."""
    out = []
    for m in modules:
        base = f"{m}.inner_model"
        out += [f"{base}.cond_channels=16", f"{base}.depths=[1,1,1,1]", f"{base}.channels=[8,8,8,8]"]
        out.append(f"{base}.attn_depths=[0,0,0,0]")
    return out


@pytest.fixture(scope="session")
def tiny_model_dotlist():
    """Factory for the dotlist above."""
    return _tiny_model_dotlist


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """Low-res DIAMOND .pt datasets (train/test) plus the HDF5 full-res twins written by cs2wm.pipeline.pack."""
    root = tmp_path_factory.mktemp("data")
    cfg = load_config(overrides=["pack.fullres_mod=1"])
    rng = np.random.default_rng(0)
    full_dir = root / "full_res"
    for split, n_episodes in N_EPISODES.items():
        ds = Dataset(root / "low_res" / split, None)
        for i in range(n_episodes):
            episode_id = f"{split}{i}"
            frames = rng.integers(0, 256, size=(N_FRAMES, 150, 280, 3), dtype=np.uint8)
            acts = (rng.uniform(size=(N_FRAMES, 51)) < 0.1).astype(np.float32)
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
            result = write_episode(frames, acts, meta, root / "hdf5", cfg, fullres_dir=full_dir)
            obs = torch.from_numpy(to_low_res(frames)).flip(3).permute(0, 3, 1, 2).float().div(255).mul(2).sub(1)
            info = {"original_file_id": f"{result.fullres_path.parent.name}/{result.fullres_path.name}"}
            zeros = torch.zeros(N_FRAMES, dtype=torch.uint8)
            ds.add_episode(Episode(obs, torch.from_numpy(acts), torch.zeros(N_FRAMES), zeros, zeros.clone(), info))
        ds.save_to_default_path()
    return root
