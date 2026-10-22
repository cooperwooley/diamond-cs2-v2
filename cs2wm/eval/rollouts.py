"""Single-step grids and autoregressive rollouts (#34). Ports v1 ``scripts/eval_single_step.py`` (diamond-cs2 @
a0b9e24); see ATTRIBUTION.md. Rollouts are 64 frames from the same held-out contexts for v1 and v2.
"""

from __future__ import annotations

from pathlib import Path


def run_rollouts(ckpt: Path, data: Path, out_dir: Path, n_frames: int = 64, seed: int = 0) -> Path:
    """Generate single-step prediction grids and autoregressive rollouts.

    Args:
        ckpt: Agent checkpoint.
        data: Test split directory.
        out_dir: Output directory for the figures.
        n_frames: Rollout length.
        seed: RNG seed.

    Returns:
        The output directory.
    """
    # TODO(#34): port v1 scripts/eval_single_step.py on top of the shared agent loading.
    raise NotImplementedError("TODO(#34): run_rollouts")
