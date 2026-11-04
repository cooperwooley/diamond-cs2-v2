"""CPU smoke test: 2 optimizer steps of the v1 Trainer on a tiny synthetic DIAMOND dataset, with a tiny model."""

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from agent import Agent
from cs2wm.actions import ACTION_DIM
from cs2wm.play.agent import DEFAULT_AGENT_CONFIG_PATH, DENOISER_ONLY_CONFIG_PATH, build_play_agent
from trainer import Trainer

STEPS = {"batch_size": 2, "grad_acc_steps": 1, "steps_first_epoch": 2, "steps_per_epoch": 2, "lr_warmup_steps": 1}


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
        *(["training.models=[upsampler]"] if modes == "upsampler_only" else []),
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


@pytest.mark.parametrize("mode", ["denoiser_only", "denoiser_upsampler", "upsampler_only"])
def test_two_optimizer_steps(mode, data, run_dir, repo_root, tiny_model_dotlist) -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)  # as in src/main.py
    cfg = compose_cfg(repo_root, data, mode, tiny_model_dotlist)
    trainer = Trainer(cfg, run_dir)

    names = {
        "denoiser_only": ["denoiser"],
        "denoiser_upsampler": ["denoiser", "upsampler"],
        "upsampler_only": ["upsampler"],
    }[mode]
    assert trainer._model_names == names
    assert (trainer.agent.upsampler is None) == (mode == "denoiser_only")
    assert (trainer.train_dataset._dataset_full_res is None) == (mode == "denoiser_only")
    all_names = ["denoiser"] if mode == "denoiser_only" else ["denoiser", "upsampler"]
    before = {n: [p.detach().clone() for p in getattr(trainer.agent, n).parameters()] for n in all_names}

    trainer.run()

    for n in names:
        assert trainer.num_batch_train.get(n) == 2, n  # grad_acc_steps=1, so 2 forward/backward = 2 optimizer steps
        assert trainer.num_batch_test.get(n) > 0, n
        after = list(getattr(trainer.agent, n).parameters())
        assert any(not torch.equal(a, b) for a, b in zip(before[n], after)), f"{n} did not update"
        assert all(torch.isfinite(p).all() for p in after), n
    assert (run_dir / "checkpoints" / "state.pt").is_file()

    if mode == "upsampler_only":
        # The denoiser is never stepped, evaluated or given an optimizer.
        assert trainer.num_batch_train.denoiser == 0 and trainer.num_batch_test.denoiser == 0
        assert trainer.opt.denoiser is None and trainer.lr_sched.denoiser is None
        assert all(torch.equal(a, b) for a, b in zip(before["denoiser"], trainer.agent.denoiser.parameters()))
        row = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[-1])
        assert row["train_loss"] is not None and row["test_loss"] is not None  # the upsampler's, not the denoiser's


def test_upsampler_without_full_res_path_is_rejected(data, run_dir, repo_root, tiny_model_dotlist) -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = compose_cfg(repo_root, data, "denoiser_upsampler", tiny_model_dotlist)
    cfg.env.path_data_full_res = None
    with pytest.raises(AssertionError, match="path_data_full_res"):
        Trainer(cfg, run_dir)


def test_upsampler_only_checkpoint_loads_into_play_agent(data, run_dir, repo_root, tiny_model_dotlist) -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = compose_cfg(repo_root, data, "upsampler_only", tiny_model_dotlist)
    trainer = Trainer(cfg, run_dir)
    trainer.run()
    (ckpt,) = (run_dir / "checkpoints" / "agent_versions").glob("agent_epoch_00001.pt")

    tiny = OmegaConf.from_dotlist(tiny_model_dotlist())
    denoiser_cfg = OmegaConf.merge(
        OmegaConf.load(DENOISER_ONLY_CONFIG_PATH), OmegaConf.from_dotlist(tiny_model_dotlist(("denoiser",)))
    )
    denoiser_ckpt = run_dir / "denoiser_only.pt"
    denoiser_agent = Agent(instantiate(denoiser_cfg, num_actions=ACTION_DIM))
    torch.save(denoiser_agent.state_dict(), denoiser_ckpt)

    agent = build_play_agent(
        denoiser_ckpt, ckpt, agent_cfg=OmegaConf.merge(OmegaConf.load(DEFAULT_AGENT_CONFIG_PATH), tiny)
    )
    for got, want in zip(agent.upsampler.state_dict().values(), trainer.agent.upsampler.state_dict().values()):
        torch.testing.assert_close(got, want)
    for got, want in zip(agent.denoiser.state_dict().values(), denoiser_agent.denoiser.state_dict().values()):
        torch.testing.assert_close(got, want)
    # The checkpoint's untrained denoiser is not what the play agent uses.
    assert not torch.equal(
        next(iter(agent.denoiser.state_dict().values())), next(iter(trainer.agent.denoiser.state_dict().values()))
    )


def test_duplicate_training_model_is_rejected(data, run_dir, repo_root, tiny_model_dotlist) -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = compose_cfg(repo_root, data, "upsampler_only", tiny_model_dotlist)
    cfg.training.models = ["upsampler", "upsampler"]
    with pytest.raises(AssertionError, match="more than once"):
        Trainer(cfg, run_dir)


def test_unknown_training_model_is_rejected(data, run_dir, repo_root, tiny_model_dotlist) -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = compose_cfg(repo_root, data, "denoiser_only", tiny_model_dotlist)
    cfg.training.models = ["upsampler"]  # the denoiser-only agent has none
    with pytest.raises(AssertionError, match="training.models"):
        Trainer(cfg, run_dir)
