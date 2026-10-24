"""CPU smoke test: 2 optimizer steps of the v1 Trainer on a tiny synthetic DIAMOND dataset, with a tiny model."""

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cs2wm.config import load_config
from cs2wm.pipeline.pack import to_low_res, write_episode
from data import Dataset, Episode
from trainer import Trainer

N_EPISODES = {"train": 3, "test": 1}
STEPS = {"batch_size": 2, "grad_acc_steps": 1, "steps_first_epoch": 2, "steps_per_epoch": 2, "lr_warmup_steps": 1}
N_FRAMES = 12  # longer than the denoiser (4 + 1 + 1) and upsampler (1 + 1 + 1) sequence lengths


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


def compose_cfg(repo_root: Path, data: Path, modes: str, tiny_model_dotlist) -> OmegaConf:
    denoiser_only = modes == "denoiser_only"
    overrides = [
        f"agent={'cs2_denoiser_only' if denoiser_only else 'cs2'}",
        f"env.path_data_low_res={data / 'low_res'}",
        f"env.path_data_full_res={'null' if denoiser_only else data / 'full_res'}",
        "common.devices=cpu",
        "training.num_workers_data_loaders=0",
        "training.num_final_epochs=1",
        "evaluation.every=1",
        "denoiser.training.num_autoregressive_steps=1",
        "upsampler.training.num_autoregressive_steps=1",
        # Two optimizer steps per epoch and component.
        *[f"{m}.training.{k}={v}" for m in ("denoiser", "upsampler") for k, v in STEPS.items()],
        *[f"agent.{x}" for x in tiny_model_dotlist(("denoiser",) if denoiser_only else ("denoiser", "upsampler"))],
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "config")):
        return compose(config_name="trainer", overrides=overrides)


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A working dir shaped like a Hydra run dir (the Trainer moves .hydra/config.yaml)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    (tmp_path / ".hydra").mkdir()
    (tmp_path / ".hydra" / "config.yaml").write_text("{}\n")
    return tmp_path


@pytest.mark.parametrize("mode", ["denoiser_only", "denoiser_upsampler"])
def test_two_optimizer_steps(mode, data, run_dir, repo_root, tiny_model_dotlist) -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)  # as in src/main.py
    cfg = compose_cfg(repo_root, data, mode, tiny_model_dotlist)
    trainer = Trainer(cfg, run_dir)

    names = ["denoiser"] if mode == "denoiser_only" else ["denoiser", "upsampler"]
    assert trainer._model_names == names
    assert (trainer.agent.upsampler is None) == (mode == "denoiser_only")
    assert (trainer.train_dataset._dataset_full_res is None) == (mode == "denoiser_only")
    before = {n: [p.detach().clone() for p in getattr(trainer.agent, n).parameters()] for n in names}

    trainer.run()

    for n in names:
        assert trainer.num_batch_train.get(n) == 2, n  # grad_acc_steps=1, so 2 forward/backward = 2 optimizer steps
        assert trainer.num_batch_test.get(n) > 0, n
        after = list(getattr(trainer.agent, n).parameters())
        assert any(not torch.equal(a, b) for a, b in zip(before[n], after)), f"{n} did not update"
        assert all(torch.isfinite(p).all() for p in after), n
    assert (run_dir / "checkpoints" / "state.pt").is_file()


def test_upsampler_without_full_res_path_is_rejected(data, run_dir, repo_root, tiny_model_dotlist) -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = compose_cfg(repo_root, data, "denoiser_upsampler", tiny_model_dotlist)
    cfg.env.path_data_full_res = None
    with pytest.raises(AssertionError, match="path_data_full_res"):
        Trainer(cfg, run_dir)
