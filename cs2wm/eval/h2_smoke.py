"""H2 volumetric-smoke eval (#33): LPIPS (AlexNet) and SSIM inside smoke masks vs static geometry.

Computed on held-out smoke episodes for v2 and v1, for teacher-forced single-step predictions and 16-step
autoregressive rollouts, with bootstrap CIs. Writes ``results/h2/summary.json`` and example triptychs.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig


def run_h2(models: dict[str, Path], cfg: DictConfig, out_dir: Path, seed: int = 0) -> dict:
    """Run the H2 comparison.

    Args:
        models: Mapping from name (``v2``, ``v1``) to checkpoint path.
        cfg: A validated cs2wm config.
        out_dir: Output directory (``results/h2``).
        seed: RNG seed.

    Returns:
        The summary dict that is written to ``summary.json``.
    """
    # TODO(#33): masks via smoke_masks, static mask = complement minus viewmodel box on enemy-free frames.
    raise NotImplementedError("TODO(#33): run_h2")
