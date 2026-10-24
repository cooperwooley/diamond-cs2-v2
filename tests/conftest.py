"""Shared pytest fixtures."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# src/ is v1 code with flat imports (agent, data, models, ...); put it on the path as src/main.py does.
sys.path.insert(0, str(REPO_ROOT / "src"))


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
