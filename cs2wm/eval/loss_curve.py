"""Loss curves, v1 vs v2 (#34). Ports v1 ``scripts/evaluate.py`` (cooperwooley/diamond-cs2@a0b9e24), which is written
against DIAMOND's (Alonso et al., MIT) agent and trainer; see ATTRIBUTION.md.

v1 plotted its denoiser test loss against epoch (``plot_loss_curve``: "Test Loss (MSE)" vs "Epoch") and, in the combined
figure, its early per-step training loss next to it. This module plots the same two quantities for v1 and v2 on the same
axes, in epochs.

Inputs
------
* v2 ``metrics.jsonl`` (trainer, #6): one JSON line per epoch with ``epoch``, ``train_loss``, ``test_loss`` (null on
  epochs without evaluation), ``lr``, ``duration_s``, ``samples_per_s`` and ``gpu``. A hard kill can duplicate a line,
  so lines are deduplicated by epoch, the last one winning. A truncated final line (a kill mid-write) is ignored.
* ``results/v1/cloud_training_losses.json``: ``{"<epoch>": {"denoiser": test loss, "upsampler": test loss}}`` for the
  checkpoints v1 evaluated (epochs 95, 120, 149 of the 150-epoch cloud run). Already per epoch. v1's epoch labels are
  used as they are.
* ``results/v1/training_losses.json``: a list with the denoiser training loss of the first 100 steps of an early local
  run (per step, not per epoch). It is converted to epochs with v1's steps per epoch, 400 (``STEPS_PER_EPOCH``, the
  decision of #6): step ``i`` (1-based) is plotted at ``i / 400``. v1's own figure placed these steps on a different
  scale (50 steps per epoch); the 400 here is the one v2 trains with.

Comparability: test losses are the denoiser's denoising MSE on the held-out split, but on *different* test splits and
data (v1: YouTube frames with a HUD, v2: HLAE captures), so the curves show convergence behaviour, not a like-for-like
score. v2's train loss is an epoch mean; v1's is a single early run read per step, so it is only drawn as a reference
and the train panel is on a log epoch axis to fit both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS_PER_EPOCH = 400  # v1 and v2 steps per epoch (#6)
V1_TEST_FILE = "cloud_training_losses.json"
V1_TRAIN_FILE = "training_losses.json"
V1_COLOR = "#D55E00"
V2_COLOR = "#0072B2"
FIGURE_STEM = "loss_curve"


def load_v2_metrics(path: Path) -> dict[str, Any]:
    """Read v2's ``metrics.jsonl``, deduplicated by epoch (last line wins).

    Args:
        path: ``metrics.jsonl`` written by the trainer.

    Returns:
        ``{"epochs", "train_loss", "test_loss", "n_lines", "n_duplicates"}`` with the series sorted by epoch;
        ``train_loss`` / ``test_loss`` hold None where a line has no value.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line other than the last is not valid JSON or has no integer ``epoch``, or the file has no
            usable line.
    """
    path = Path(path)
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    by_epoch: dict[int, dict[str, Any]] = {}
    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            if i == len(lines) - 1:  # a hard kill mid-write leaves a partial last line
                lines = lines[:-1]
                break
            raise ValueError(f"{path}:{i + 1}: not valid JSON") from exc
        epoch = rec.get("epoch") if isinstance(rec, dict) else None
        if type(epoch) is not int:
            raise ValueError(f"{path}:{i + 1}: expected a JSON object with an integer 'epoch'")
        by_epoch[epoch] = rec
    if not by_epoch:
        raise ValueError(f"{path}: no metrics lines")
    epochs = sorted(by_epoch)
    return {
        "epochs": epochs,
        "train_loss": [_opt_float(by_epoch[e].get("train_loss")) for e in epochs],
        "test_loss": [_opt_float(by_epoch[e].get("test_loss")) for e in epochs],
        "n_lines": len(lines),
        "n_duplicates": len(lines) - len(epochs),
    }


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def load_v1_losses(v1_results: Path, steps_per_epoch: int = STEPS_PER_EPOCH) -> dict[str, Any]:
    """Read v1's loss JSONs from ``results/v1``.

    Args:
        v1_results: Directory holding ``cloud_training_losses.json`` and ``training_losses.json``.
        steps_per_epoch: Steps per epoch used to convert the per-step training losses to epochs.

    Returns:
        ``{"steps_per_epoch", "test": {"epochs", "denoiser", "upsampler"}, "train_early": {"epochs", "loss",
        "n_steps"}}``; the test series is sorted by epoch.

    Raises:
        FileNotFoundError: If either file is missing.
        ValueError: If a file does not have the v1 structure.
    """
    v1_results = Path(v1_results)
    test = json.loads((v1_results / V1_TEST_FILE).read_text())
    if not isinstance(test, dict) or not test:
        raise ValueError(f"{V1_TEST_FILE}: expected a non-empty object keyed by epoch")
    epochs = sorted(int(k) for k in test)
    for k, v in test.items():
        if "denoiser" not in v:
            raise ValueError(f"{V1_TEST_FILE}: epoch {k} has no 'denoiser' loss")
    steps = json.loads((v1_results / V1_TRAIN_FILE).read_text())
    if not isinstance(steps, list) or not all(isinstance(x, (int, float)) for x in steps):
        raise ValueError(f"{V1_TRAIN_FILE}: expected a list of per-step losses")
    return {
        "steps_per_epoch": steps_per_epoch,
        "test": {
            "epochs": epochs,
            "denoiser": [float(test[str(e)]["denoiser"]) for e in epochs],
            "upsampler": [_opt_float(test[str(e)].get("upsampler")) for e in epochs],
        },
        "train_early": {
            "epochs": [(i + 1) / steps_per_epoch for i in range(len(steps))],
            "loss": [float(x) for x in steps],
            "n_steps": len(steps),
        },
    }


def plot_loss_curves(v1: dict[str, Any], v2: dict[str, Any] | None, out_dir: Path) -> list[str]:
    """Plot v1 and v2 test loss (left) and train loss (right) against epoch.

    Args:
        v1: ``load_v1_losses`` output.
        v2: ``load_v2_metrics`` output, or None to plot v1 alone.
        out_dir: Output directory.

    Returns:
        File names written (PDF and PNG).
    """
    fig, (ax_test, ax_train) = plt.subplots(1, 2, figsize=(11, 4))
    ax_test.plot(
        v1["test"]["epochs"], v1["test"]["denoiser"], "o-", color=V1_COLOR, label="v1 (cloud checkpoints)", lw=2
    )
    ax_train.plot(
        v1["train_early"]["epochs"],
        v1["train_early"]["loss"],
        "-",
        color=V1_COLOR,
        alpha=0.6,
        lw=1,
        label="v1 (first 100 steps, per step)",
    )
    if v2 is not None:
        pairs = [(e, x) for e, x in zip(v2["epochs"], v2["test_loss"]) if x is not None]
        if pairs:
            ax_test.plot(*zip(*pairs), "s-", color=V2_COLOR, label="v2", lw=2, ms=4)
        pairs = [(e, x) for e, x in zip(v2["epochs"], v2["train_loss"]) if x is not None]
        if pairs:
            ax_train.plot(*zip(*pairs), "s-", color=V2_COLOR, label="v2 (epoch mean)", lw=2, ms=4)
    ax_test.set(xlabel="Epoch", ylabel="Denoiser test loss (MSE)", title="Test loss")
    ax_train.set(xlabel="Epoch (log scale; v1 steps at step / 400)", ylabel="Denoiser train loss", title="Train loss")
    ax_train.set_xscale("log")
    for ax in (ax_test, ax_train):
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    out_dir = Path(out_dir)
    names = []
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{FIGURE_STEM}.{ext}", dpi=200, bbox_inches="tight")
        names.append(f"{FIGURE_STEM}.{ext}")
    plt.close(fig)
    return names


def run_loss_curve(v2_metrics: Path | None, v1_results: Path, out_dir: Path) -> dict[str, Any]:
    """Plot v1 and v2 loss against epoch on the same axes and return the plotted series.

    Args:
        v2_metrics: v2 run's ``metrics.jsonl``, or None to plot v1 alone.
        v1_results: ``results/v1`` directory.
        out_dir: Output directory (created).

    Returns:
        ``{"x_axis": "epoch", "v1": ..., "v2": ... or None, "figure": [file names], "notes": [...]}`` (the series of
        ``load_v1_losses`` and ``load_v2_metrics``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v1 = load_v1_losses(v1_results)
    v2 = load_v2_metrics(v2_metrics) if v2_metrics is not None else None
    return {
        "x_axis": "epoch",
        "v1": v1,
        "v2": v2,
        "figure": plot_loss_curves(v1, v2, out_dir),
        "notes": [
            f"v1 per-step training losses are converted to epochs at {STEPS_PER_EPOCH} steps per epoch (#6).",
            "v1 and v2 test losses are on different test splits (see loss_curve.py).",
        ],
    }
