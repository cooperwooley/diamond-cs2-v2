"""Play/eval agent: the v2 denoiser plus the v1 upsampler (#5).

v2 trains only the denoiser, so its checkpoints hold no upsampler. For play and eval the v1 upsampler is loaded
from a v1 checkpoint into an agent that is built with both modules.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from cs2wm.actions import ACTION_DIM

if TYPE_CHECKING:
    from agent import Agent

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGENT_CONFIG_PATH = _REPO_ROOT / "config" / "agent" / "cs2.yaml"
DENOISER_ONLY_CONFIG_PATH = _REPO_ROOT / "config" / "agent" / "cs2_denoiser_only.yaml"


def _import_agent() -> type[Agent]:
    """Import ``agent.Agent``. src/ keeps v1's flat imports (agent, models, ...), so it has to be on sys.path."""
    src = str(_REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from agent import Agent

    return Agent


def build_play_agent(
    v2_ckpt: Path,
    v1_upsampler_ckpt: Path,
    device: str | torch.device = "cpu",
    agent_cfg: DictConfig | None = None,
) -> Agent:
    """Build an eval-mode agent with the denoiser from a v2 checkpoint and the upsampler from a v1 checkpoint.

    Args:
        v2_ckpt: Agent checkpoint holding the denoiser (``agent_versions/agent_epoch_*.pt``); any upsampler weights
            in it are ignored.
        v1_upsampler_ckpt: v1 agent checkpoint holding the upsampler; its denoiser weights are ignored.
        device: Device to put the agent on.
        agent_cfg: Agent config with both a denoiser and an upsampler; defaults to ``config/agent/cs2.yaml``.

    Returns:
        The agent in eval mode, with ``denoiser`` and ``upsampler`` both loaded.

    Raises:
        ValueError: If ``agent_cfg`` has no upsampler.
    """
    Agent = _import_agent()

    if agent_cfg is None:
        agent_cfg = OmegaConf.load(DEFAULT_AGENT_CONFIG_PATH)
    if agent_cfg.get("upsampler") is None:
        raise ValueError("agent_cfg must define an upsampler; use config/agent/cs2.yaml, not cs2_denoiser_only")
    agent = Agent(instantiate(agent_cfg, num_actions=ACTION_DIM))
    agent.load(v2_ckpt, load_denoiser=True, load_upsampler=False)
    agent.load(v1_upsampler_ckpt, load_denoiser=False, load_upsampler=True)
    return agent.to(device).eval().requires_grad_(False)


def build_denoiser_agent(v2_ckpt: Path, device: str | torch.device = "cpu") -> Agent:
    """Build an eval-mode denoiser-only agent (play with ``--no-upsampler``).

    Args:
        v2_ckpt: Agent checkpoint holding the denoiser.
        device: Device to put the agent on.

    Returns:
        The agent in eval mode, with ``upsampler`` None.
    """
    Agent = _import_agent()
    agent = Agent(instantiate(OmegaConf.load(DENOISER_ONLY_CONFIG_PATH), num_actions=ACTION_DIM))
    agent.load(v2_ckpt, load_denoiser=True, load_upsampler=False)
    return agent.to(device).eval().requires_grad_(False)
