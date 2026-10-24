"""Single-step grids and autoregressive rollouts (#34). Ports v1 ``scripts/eval_single_step.py`` and the frame
generation of ``scripts/evaluate.py`` (cooperwooley/diamond-cs2@a0b9e24), which are written against DIAMOND's
(Alonso et al., MIT) agent and sampler; see ATTRIBUTION.md.

What is carried over from v1:

* **Single step**: 4 ground-truth context frames and their actions -> the sampled next 30x56 frame, shown with the last
  context frame and the ground-truth next frame as a grid (v1: ``single_step_comparison.png``, columns "Last Context
  Frame | Predicted | Ground Truth", frames upscaled 4x with nearest neighbour). Here a column per model is added.
* **Multi-step**: 64-frame autoregressive rollouts compared with the ground-truth frames that follow the context
  (v1: ``generated_grid.png`` / ``temporal_strip_*.png`` / ``comparison.png``), shown as strips at a few timesteps.
* Low-res only: v1 sampled with the denoiser alone, so no upsampler is involved here.

What is different, on purpose:

* v1 used one context from episode 0 of its test set, zero actions (its labels were zeros) and drew its rollouts from
  its test episode 0. Here the contexts are drawn from all held-out episodes with a seeded RNG, the same ones for every
  model, and the rollouts are driven by the ground-truth actions of the held-out episode (``zero_actions=True``
  reproduces v1's zero-action driving). The rollouts use ``WorldModel.rollout``, whose noise is per window and shared by
  all models.
* v1 printed no image metrics. MSE and PSNR against the ground truth per rollout step, and the same for a
  "hold the last context frame" baseline, are added so the curves can be compared without eyeballing.
* A single-step prediction from a ground-truth context is exactly step 1 of the rollout from that context (same
  noise), so the single-step grid is built from step 1 of the rollouts instead of sampling it again.

Action alignment: the action at frame ``k`` is the one taken at ``k`` and leads to frame ``k + 1``. For a window of
``n_cond + T`` frames the context is frames ``0..n_cond-1`` with their actions, rollout step ``t`` (0-based) uses action
``n_cond - 1 + t`` and its ground truth is frame ``n_cond + t``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw

from cs2wm.actions import ACTION_DIM
from cs2wm.pipeline.pack import read_episode
from cs2wm.play.agent import _import_agent
from cs2wm.play.world_model import WorldModel, frames_to_tensor, tensor_to_rgb_uint8

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCALE = 4  # v1 upscaled the 30x56 frames 4x (nearest) for its grids
STRIP_STEPS = (1, 4, 8, 16, 32, 64)  # rollout steps (1-based) shown in the strips
LABEL_H = 16
ROW_LABEL_W = 96
BG = (30, 30, 30)
FG = (220, 220, 220)
MODEL_COLORS = ("#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00")


class EpisodeSource:
    """The episodes of a held-out test set, in either of the two on-disk formats.

    * v2 / pack format: a directory of ``.hdf5`` episodes (``pack.read_episode``), optionally restricted to test demos
      by a ``split.json`` as in ``h1_controllability.list_episodes``.
    * v1 format: a DIAMOND ``Dataset`` directory (``info.pt`` plus ``.pt`` episodes; ``<v1 data>/low_res/test``).

    Args:
        directory: The test set directory.
        split_json: Optional ``split.json`` for the HDF5 format.

    Raises:
        FileNotFoundError: If the directory holds neither format.
        ValueError: If ``split_json`` is given for a DIAMOND dataset.
    """

    def __init__(self, directory: Path, split_json: Path | None = None) -> None:
        self.directory = Path(directory)
        if (self.directory / "info.pt").is_file():
            if split_json is not None:
                raise ValueError("--split-json applies to HDF5 episode directories, not to a DIAMOND dataset")
            _import_agent()  # puts src/ on sys.path
            from data import Dataset

            self.kind = "diamond"
            self._dataset = Dataset(self.directory, None, "carried_eval", cache_in_ram=False)
            self._dataset.load_from_default_path()
            self.names = [str(i) for i in range(self._dataset.num_episodes)]
            self.lengths = [int(n) for n in self._dataset.lengths]
        else:
            from cs2wm.eval.h1_controllability import list_episodes

            self.kind = "hdf5"
            self._files = list_episodes(self.directory, split_json)
            self.names = [p.stem for p in self._files]
            self.lengths = []
            for p in self._files:
                with h5py.File(p, "r") as f:
                    self.lengths.append(int(f.attrs["n_frames"]))
        if not self.names:
            raise FileNotFoundError(f"{self.directory}: no episodes")

    def __len__(self) -> int:
        return len(self.names)

    def load(self, i: int) -> tuple[torch.Tensor, np.ndarray]:
        """Load episode ``i``.

        Returns:
            ``(frames, actions)``: ``float32[n, 3, h, w]`` RGB in [-1, 1] and ``float32[n, 51]``.

        Raises:
            ValueError: If the actions are not 51-dimensional.
        """
        if self.kind == "diamond":
            ep = self._dataset.load_episode(i)
            frames, actions = ep.obs.float(), ep.act.float().numpy()
        else:
            bgr, actions, _ = read_episode(self._files[i])
            frames = frames_to_tensor(bgr)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"{self.directory} episode {self.names[i]}: expected [n, {ACTION_DIM}] actions")
        return frames, actions


@dataclass(frozen=True, eq=False)
class Window:
    """A held-out window of ``n_cond + T`` consecutive ground-truth frames and actions.

    Attributes:
        index: Position in the sampled list.
        episode: Episode name.
        start: Index of the first frame in the episode.
        frames: ``float32[n_cond + T, 3, h, w]`` RGB in [-1, 1].
        actions: ``float32[n_cond + T, 51]``.
    """

    index: int
    episode: str
    start: int
    frames: torch.Tensor
    actions: np.ndarray


def sample_windows(source: EpisodeSource, n: int, length: int, seed: int) -> list[Window]:
    """Draw ``n`` windows deterministically, spread over the episodes.

    Episodes with at least ``length`` frames are visited in a seeded random order (cycling if ``n`` exceeds their
    number), each contributing a window that starts at a seeded random frame (the scheme of
    ``h1_controllability.sample_contexts``).

    Args:
        source: The test set.
        n: Windows wanted.
        length: Frames per window (``n_cond`` plus the rollout length).
        seed: RNG seed.

    Returns:
        ``n`` windows.

    Raises:
        ValueError: If ``n < 1`` or no episode has ``length`` frames.
    """
    if n < 1:
        raise ValueError(f"windows must be at least 1, got {n}")
    eligible = [i for i, m in enumerate(source.lengths) if m >= length]
    if not eligible:
        raise ValueError(f"{source.directory}: no episode has at least {length} frames")
    rng = np.random.default_rng(seed)
    order: list[int] = []
    while len(order) < n:
        order += [int(i) for i in rng.permutation(eligible)]
    picks = [(i, int(rng.integers(0, source.lengths[i] - length + 1))) for i in order[:n]]
    cache: dict[int, tuple[torch.Tensor, np.ndarray]] = {}
    windows = []
    for k, (i, start) in enumerate(picks):
        if i not in cache:
            cache = {i: source.load(i)}  # picks revisit an episode only after a full cycle: keep one
        frames, actions = cache[i]
        windows.append(
            Window(k, source.names[i], start, frames[start : start + length], actions[start : start + length])
        )
    return windows


def window_generator(seed: int, index: int) -> torch.Generator:
    """The noise generator of window ``index``: the same for every model, different per run seed."""
    state = np.random.SeedSequence([seed, index]).generate_state(2, dtype=np.uint32)
    return torch.Generator().manual_seed(int(state[0]) << 32 | int(state[1]))


def ground_truth(windows: Sequence[Window], n_cond: int) -> torch.Tensor:
    """The frames a rollout is compared with: ``float[N, T, 3, h, w]``, the frames after each window's context."""
    return torch.stack([w.frames[n_cond:] for w in windows])


def generate_rollouts(
    world: WorldModel,
    windows: Sequence[Window],
    seed: int,
    batch_size: int = 16,
    zero_actions: bool = False,
    progress: Callable[[int], None] | None = None,
) -> torch.Tensor:
    """Roll every window forward autoregressively from its ground-truth context.

    Args:
        world: The model (denoiser only is used; the upsampler is ignored).
        windows: Windows of ``n_cond + T`` frames; ``T`` is the rollout length.
        seed: Noise seed (per window, independent of the batch size and of the model).
        batch_size: Rollouts per batched forward pass.
        zero_actions: Drive the model with all-zero actions, as v1 did, instead of the episode's actions.
        progress: Called with the number of windows done after each batch.

    Returns:
        ``float[N, T, 3, h, w]`` generated low-res frames on the CPU (unclamped sampler output).

    Raises:
        ValueError: If a window is shorter than ``n_cond + 1`` frames or the windows differ in length.
    """
    n_cond = world.n_cond
    lengths = {len(w.frames) for w in windows}
    if len(lengths) != 1 or min(lengths) <= n_cond:
        raise ValueError(f"windows must all have more than n_cond={n_cond} frames, got lengths {sorted(lengths)}")
    out = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i : i + batch_size]
        frames = torch.stack([w.frames[:n_cond] for w in batch])
        acts = torch.from_numpy(np.stack([w.actions for w in batch])).float()
        if zero_actions:
            acts = torch.zeros_like(acts)
        low, _ = world.rollout(
            frames, acts[:, :n_cond], acts[:, n_cond - 1 : -1], [window_generator(seed, w.index) for w in batch]
        )
        out.append(low)
        if progress is not None:
            progress(len(batch))
    return torch.cat(out)


def step_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, list[float]]:
    """Per-step MSE and PSNR against the ground truth, averaged over the windows.

    Frames are compared on RGB in [0, 1] (predictions clamped to the valid range first); PSNR has peak 1 and is the mean
    of the per-frame PSNRs.

    Args:
        pred: ``float[N, T, 3, h, w]`` in about [-1, 1].
        gt: ``float[N, T, 3, h, w]`` in [-1, 1].

    Returns:
        ``{"mse": [T], "psnr": [T]}``.
    """
    p = pred.float().clamp(-1, 1).add(1).div(2)
    g = gt.float().add(1).div(2)
    mse = (p - g).pow(2).mean(dim=(2, 3, 4))  # [N, T]
    psnr = -10 * torch.log10(mse.clamp_min(1e-10))
    return {"mse": mse.mean(0).tolist(), "psnr": psnr.mean(0).tolist()}


def _grid(
    rows: Sequence[Sequence[np.ndarray]],
    col_labels: Sequence[str],
    row_labels: Sequence[str] | None = None,
    scale: int = SCALE,
) -> Image.Image:
    """Tile ``uint8[h, w, 3]`` cells into a labelled image, cells upscaled with nearest neighbour as v1 did."""
    h, w = rows[0][0].shape[:2]
    cw, ch = w * scale, h * scale
    left = ROW_LABEL_W if row_labels else 0
    img = Image.new("RGB", (left + len(col_labels) * cw, LABEL_H + len(rows) * ch), BG)
    draw = ImageDraw.Draw(img)
    for c, label in enumerate(col_labels):
        draw.text((left + c * cw + 2, 2), label, fill=FG)
    for r, row in enumerate(rows):
        y = LABEL_H + r * ch
        if row_labels:
            draw.text((2, y + 2), row_labels[r], fill=FG)
        for c, cell in enumerate(row):
            img.paste(Image.fromarray(cell).resize((cw, ch), Image.NEAREST), (left + c * cw, y))
    return img


def single_step_grid(
    windows: Sequence[Window], preds: dict[str, torch.Tensor], n_cond: int, n_samples: int, out_path: Path
) -> None:
    """Save the single-step comparison: ``last context | ground truth | one column per model``, a row per window.

    Args:
        windows: The windows (the first ``n_samples`` are shown).
        preds: Model name -> rollouts ``float[N, T, 3, h, w]``; step 1 is the single-step prediction.
        n_cond: Context length.
        n_samples: Rows.
        out_path: PNG path.
    """
    rows = []
    for k, w in enumerate(windows[:n_samples]):
        row = [tensor_to_rgb_uint8(w.frames[n_cond - 1]), tensor_to_rgb_uint8(w.frames[n_cond])]
        rows.append(row + [tensor_to_rgb_uint8(p[k, 0]) for p in preds.values()])
    _grid(rows, ["Last context", "Ground truth", *[f"{m} (1-step)" for m in preds]]).save(out_path)


def rollout_strip(
    windows: Sequence[Window],
    preds: dict[str, torch.Tensor],
    n_cond: int,
    index: int,
    steps: Sequence[int],
    out_path: Path,
) -> None:
    """Save the rollout strip of one window: a ground-truth row and a row per model at the given steps.

    Args:
        windows: The windows.
        preds: Model name -> rollouts ``float[N, T, 3, h, w]``.
        n_cond: Context length.
        index: Window to show.
        steps: 1-based rollout steps (columns); the first column is the last context frame.
        out_path: PNG path.
    """
    w = windows[index]
    rows = [
        [tensor_to_rgb_uint8(w.frames[n_cond - 1])] + [tensor_to_rgb_uint8(w.frames[n_cond + s - 1]) for s in steps]
    ]
    for p in preds.values():
        rows.append([tensor_to_rgb_uint8(w.frames[n_cond - 1])] + [tensor_to_rgb_uint8(p[index, s - 1]) for s in steps])
    _grid(rows, ["context", *[f"t+{s}" for s in steps]], ["ground truth", *preds]).save(out_path)


def plot_step_metrics(metrics: dict[str, Any], out_dir: Path) -> list[str]:
    """Plot MSE and PSNR against the rollout step for each model and the hold-last-frame baseline.

    Args:
        metrics: ``run_rollouts`` ``metrics`` (model -> ``{"mse", "psnr"}`` plus ``hold_last_context``).
        out_dir: Output directory.

    Returns:
        File names written (PDF and PNG).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    models = [m for m in metrics if m != "hold_last_context"]
    for ax, key, label in zip(axes, ("mse", "psnr"), ("MSE (RGB in [0, 1])", "PSNR (dB)")):
        for color, m in zip(MODEL_COLORS, models):
            ys = metrics[m][key]
            ax.plot(range(1, len(ys) + 1), ys, color=color, label=m, lw=2)
        ys = metrics["hold_last_context"][key]
        ax.plot(range(1, len(ys) + 1), ys, "--", color="gray", label="hold last context frame", lw=1.5)
        ax.set(xlabel="Rollout step", ylabel=label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    names = []
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"rollout_metrics.{ext}", dpi=200, bbox_inches="tight")
        names.append(f"rollout_metrics.{ext}")
    plt.close(fig)
    return names


def run_rollouts(
    preds: dict[str, torch.Tensor],
    windows: Sequence[Window],
    n_cond: int,
    out_dir: Path,
    single_step_samples: int = 12,
    strip_contexts: int = 3,
    action_sources: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write the single-step grid, the rollout strips and the per-step metrics for already generated rollouts.

    Args:
        preds: Model name -> ``generate_rollouts`` output for ``windows`` (same windows for every model).
        windows: The held-out windows the rollouts started from.
        n_cond: Context length.
        out_dir: Output directory (created).
        single_step_samples: Rows of the single-step grid (v1: 12).
        strip_contexts: Windows shown as rollout strips.
        action_sources: Name -> ``"gt"`` or ``"zeros"``, recorded as each entry's ``action_source`` (default ``gt``).

    Returns:
        ``{"metrics": {name: {"mse", "psnr", "single_step": {"mse", "psnr"}, "action_source"},
        "hold_last_context": ...},
        "steps_shown", "files", "windows": [{index, episode, start}]}``.

    Raises:
        ValueError: If a prediction does not have a row per window.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, p in preds.items():
        if len(p) != len(windows):
            raise ValueError(f"{name}: {len(p)} rollouts for {len(windows)} windows")
    gt = ground_truth(windows, n_cond)
    horizon = gt.shape[1]
    metrics: dict[str, Any] = {}
    for name, p in preds.items():
        metrics[name] = step_metrics(p, gt)
        metrics[name]["single_step"] = {k: v[0] for k, v in metrics[name].items()}
        metrics[name]["action_source"] = (action_sources or {}).get(name, "gt")
    hold = torch.stack([w.frames[n_cond - 1 : n_cond] for w in windows]).expand(-1, horizon, -1, -1, -1)
    metrics["hold_last_context"] = step_metrics(hold, gt)
    metrics["hold_last_context"]["single_step"] = {k: v[0] for k, v in metrics["hold_last_context"].items()}

    files = ["single_step_grid.png"]
    single_step_grid(windows, preds, n_cond, single_step_samples, out_dir / files[0])
    steps = [s for s in STRIP_STEPS if s <= horizon] or [horizon]
    for k in range(min(strip_contexts, len(windows))):
        files.append(f"rollout_strip_{k:02d}.png")
        rollout_strip(windows, preds, n_cond, k, steps, out_dir / files[-1])
    files += plot_step_metrics(metrics, out_dir)
    return {
        "metrics": metrics,
        "steps_shown": steps,
        "files": files,
        "windows": [{"index": w.index, "episode": w.episode, "start": w.start} for w in windows],
    }
