"""Loss curves, v1 vs v2 (#34). Ports v1 ``scripts/evaluate.py`` (diamond-cs2 @ a0b9e24); see ATTRIBUTION.md.

Reads v2 ``metrics.jsonl`` and the v1 JSONs under ``results/v1/`` and plots test loss against epoch on one axis.
"""

from __future__ import annotations

from pathlib import Path


def run_loss_curve(v2_metrics: Path, v1_results: Path, out_dir: Path) -> Path:
    """Plot v1 and v2 test loss on the same axes.

    Args:
        v2_metrics: v2 run's ``metrics.jsonl``.
        v1_results: ``results/v1`` directory.
        out_dir: Output directory.

    Returns:
        Path of the figure.
    """
    # TODO(#34): port v1 scripts/evaluate.py.
    raise NotImplementedError("TODO(#34): run_loss_curve")
