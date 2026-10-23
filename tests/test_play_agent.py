"""Tests for cs2wm.play.agent.build_play_agent and Agent.load's selective loading."""

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from agent import Agent
from cs2wm.actions import ACTION_DIM
from cs2wm.play.agent import DEFAULT_AGENT_CONFIG_PATH, build_play_agent


@pytest.fixture
def agent_cfg(tiny_model_dotlist):
    cfg = OmegaConf.load(DEFAULT_AGENT_CONFIG_PATH)
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(tiny_model_dotlist()))


def make_agent(cfg, seed: int) -> Agent:
    torch.manual_seed(seed)
    return Agent(instantiate(cfg, num_actions=ACTION_DIM))


@pytest.fixture
def ckpts(tmp_path, agent_cfg):
    v2_cfg = OmegaConf.merge(agent_cfg, {})
    v2_cfg.upsampler = None
    v2 = make_agent(v2_cfg, seed=1)
    v1 = make_agent(agent_cfg, seed=2)
    torch.save(v2.state_dict(), tmp_path / "v2.pt")
    torch.save(v1.state_dict(), tmp_path / "v1.pt")
    return tmp_path / "v2.pt", tmp_path / "v1.pt", v2, v1


def test_build_play_agent_combines_checkpoints(ckpts, agent_cfg) -> None:
    v2_path, v1_path, v2, v1 = ckpts
    agent = build_play_agent(v2_path, v1_path, agent_cfg=agent_cfg)
    assert not agent.training
    assert all(not p.requires_grad for p in agent.parameters())
    for got, want in zip(agent.denoiser.state_dict().values(), v2.denoiser.state_dict().values()):
        torch.testing.assert_close(got, want)
    for got, want in zip(agent.upsampler.state_dict().values(), v1.upsampler.state_dict().values()):
        torch.testing.assert_close(got, want)
    # The v1 denoiser must not leak in.
    assert not torch.equal(
        next(iter(agent.denoiser.state_dict().values())), next(iter(v1.denoiser.state_dict().values()))
    )


def test_agent_load_upsampler_only_leaves_denoiser(ckpts, agent_cfg) -> None:
    _, v1_path, _, v1 = ckpts
    agent = make_agent(agent_cfg, seed=3)
    before = {k: v.clone() for k, v in agent.denoiser.state_dict().items()}
    agent.load(v1_path, load_denoiser=False, load_upsampler=True)
    for k, v in agent.denoiser.state_dict().items():
        torch.testing.assert_close(v, before[k])
    for got, want in zip(agent.upsampler.state_dict().values(), v1.upsampler.state_dict().values()):
        torch.testing.assert_close(got, want)


def test_default_config_has_upsampler() -> None:
    assert OmegaConf.load(DEFAULT_AGENT_CONFIG_PATH).upsampler is not None


def test_build_play_agent_rejects_denoiser_only_config(ckpts, agent_cfg) -> None:
    v2_path, v1_path, _, _ = ckpts
    cfg = OmegaConf.merge(agent_cfg, {})
    cfg.upsampler = None
    with pytest.raises(ValueError, match="upsampler"):
        build_play_agent(v2_path, v1_path, agent_cfg=cfg)


def test_build_play_agent_fails_when_v1_ckpt_has_no_upsampler(ckpts, agent_cfg) -> None:
    v2_path, _, _, _ = ckpts
    with pytest.raises(RuntimeError):
        build_play_agent(v2_path, v2_path, agent_cfg=agent_cfg)
