"""H1 controllability eval (#32): scripted inputs -> ego-rotation estimate -> correlation with the commanded input.

Compares v2, the v1 zero-action model and the shuffled-action control. Writes ``results/h1/summary.json``, per-model
CSVs and plots (commanded vs estimated scatter, per-bin curves).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig


@dataclass(frozen=True)
class ScriptedSuite:
    """Scripted input suite.

    Attributes:
        yaw_units: Yaw bins (+-{2, 4, 10, 20, 30, 60}).
        pitch_units: Pitch bins (+-{2, 4, 10, 20}).
        keys: Movement keys to test (W, S, A, D).
        rollout_frames: Frames per rollout (32).
        contexts: Held-out contexts (4 GT frames + actions), 50 by default.
    """

    yaw_units: tuple[int, ...] = (2, 4, 10, 20, 30, 60)
    pitch_units: tuple[int, ...] = (2, 4, 10, 20)
    keys: tuple[str, ...] = ("w", "s", "a", "d")
    rollout_frames: int = 32
    contexts: int = 50


def run_h1(models: dict[str, Path], cfg: DictConfig, out_dir: Path, contexts: int = 50, seed: int = 0) -> dict:
    """Roll out every scripted input for every model and compute the controllability metrics.

    Metrics per model: Pearson r and slope of estimated vs commanded d-yaw/d-pitch, sign accuracy, per-bin mean with
    95% bootstrap CI, and drift under no-op.

    Args:
        models: Mapping from name (``v2``, ``v1``, ``shuffled``) to checkpoint path.
        cfg: A validated cs2wm config.
        out_dir: Output directory (``results/h1``).
        contexts: Number of held-out contexts.
        seed: RNG seed.

    Returns:
        The summary dict that is written to ``summary.json``.
    """
    # TODO(#32): rollout via the shared agent loading (#28), estimate with cs2wm.eval.rotation, aggregate.
    raise NotImplementedError("TODO(#32): run_h1")
