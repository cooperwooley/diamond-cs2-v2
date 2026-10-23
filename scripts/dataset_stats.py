#!/usr/bin/env python3
"""Dataset statistics (#17, #24, #25, #26).

Two modes:

- ``dataset_stats.py <dataset_dir> [--json out.json]``: hours and frames per kind and split, episode-length
  histogram, per-slot label rates, mouse-bin histograms and the fraction of all-zero actions (#17, #25).
- ``dataset_stats.py --hdf5 <root> --summary``: one-line running total of hours on the Mill, used by the nightly
  paste-back in #24.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs2wm.actions import ActionSpec
from cs2wm.config import load_config

SPLITS = ("train", "test")
_LENGTH_BINS = 10


class _Acc:
    """Running label statistics over episodes."""

    def __init__(self, spec: ActionSpec) -> None:
        self.spec = spec
        self.episodes = 0
        self.frames = 0
        self.lengths: list[int] = []
        self.button_sum = np.zeros(len(spec.keys) + len(spec.clicks), dtype=np.float64)
        self.mouse_x = np.zeros(len(spec.mouse_x_units), dtype=np.int64)
        self.mouse_y = np.zeros(len(spec.mouse_y_units), dtype=np.int64)
        self.all_zero = 0
        self.idle = 0

    def add(self, act: np.ndarray) -> None:
        spec = self.spec
        n_buttons = len(spec.keys) + len(spec.clicks)
        self.episodes += 1
        self.frames += len(act)
        self.lengths.append(len(act))
        self.button_sum += act[:, :n_buttons].sum(axis=0)
        x_block, y_block = act[:, spec.mouse_x_slice], act[:, spec.mouse_y_slice]
        x_bin, y_bin = x_block.argmax(axis=1), y_block.argmax(axis=1)
        # Rows without a mouse label (all-zero block) are left out of the histograms rather than counted as bin 0.
        x_ok, y_ok = x_block.any(axis=1), y_block.any(axis=1)
        self.mouse_x += np.bincount(x_bin[x_ok], minlength=len(self.mouse_x))
        self.mouse_y += np.bincount(y_bin[y_ok], minlength=len(self.mouse_y))
        self.all_zero += int((~act.any(axis=1)).sum())
        # Idle: nothing pressed or clicked and no mouse movement (both mouse one-hots on the zero bin).
        still = (x_ok & (x_bin == spec.mouse_x_units.index(0.0))) & (y_ok & (y_bin == spec.mouse_y_units.index(0.0)))
        self.idle += int((~act[:, :n_buttons].any(axis=1) & still).sum())

    def summary(self, fps: int, max_frames: int) -> dict[str, Any]:
        spec, frames = self.spec, max(self.frames, 1)
        counts, edges = np.histogram(self.lengths, bins=_LENGTH_BINS, range=(0, max_frames))
        slots = (*spec.keys, *spec.clicks)
        return {
            "episodes": self.episodes,
            "frames": self.frames,
            "hours": self.frames / fps / 3600,
            "length_hist": {"edges": edges.tolist(), "counts": counts.tolist()},
            "label_rates": {slot: float(v / frames) for slot, v in zip(slots, self.button_sum)},
            "mouse_x_hist": {f"{deg:+.3f}": int(c) for deg, c in zip(spec.mouse_x_centers_deg, self.mouse_x)},
            "mouse_y_hist": {f"{deg:+.3f}": int(c) for deg, c in zip(spec.mouse_y_centers_deg, self.mouse_y)},
            "all_zero_frac": self.all_zero / frames,
            "idle_frac": self.idle / frames,
        }


def compute_stats(dataset_dir: Path) -> dict[str, Any]:
    """Compute the statistics of an assembled dataset.

    Args:
        dataset_dir: ``$S/datasets/<name>`` holding ``low_res/{train,test}``.

    Returns:
        Nested dict: ``fps``, ``total`` and per split (``train``, ``test``) summaries, plus ``by_kind``
        (kind -> split -> episodes, frames, hours). Each summary has episodes, frames, hours, ``length_hist``,
        ``label_rates`` (per key/click slot), ``mouse_x_hist``/``mouse_y_hist`` (bin centre in degrees -> frames),
        ``all_zero_frac`` (strictly zero 51-vectors) and ``idle_frac`` (no button and zero mouse bins).

    Raises:
        FileNotFoundError: If there is no ``low_res`` directory.
    """
    cfg = load_config()
    spec = ActionSpec.from_config(cfg)
    fps, max_frames = int(cfg.timing.fps), int(cfg.episodes.max_frames)
    root = Path(dataset_dir) / "low_res"
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found; is this an assembled dataset directory?")
    total = _Acc(spec)
    accs = {s: _Acc(spec) for s in SPLITS}
    kind_frames: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(SPLITS, 0))
    kind_eps: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(SPLITS, 0))
    for split in SPLITS:
        for path in sorted((root / split).rglob("*.pt")):
            if path.name == "info.pt":
                continue
            ep = torch.load(path, map_location="cpu", weights_only=True)
            act = ep["act"].numpy()
            kind = ep["info"]["kind"]
            for acc in (total, accs[split]):
                acc.add(act)
            kind_frames[kind][split] += len(act)
            kind_eps[kind][split] += 1
    return {
        "fps": fps,
        "total": total.summary(fps, max_frames),
        **{s: accs[s].summary(fps, max_frames) for s in SPLITS},
        "by_kind": {
            kind: {
                s: {
                    "episodes": kind_eps[kind][s],
                    "frames": kind_frames[kind][s],
                    "hours": kind_frames[kind][s] / fps / 3600,
                }
                for s in SPLITS
            }
            for kind in sorted(kind_frames)
        },
    }


def format_stats(stats: dict[str, Any]) -> str:
    """Render ``compute_stats`` output as text."""
    lines = []
    t = stats["total"]
    lines.append(f"total: {t['episodes']} episodes, {t['frames']} frames, {t['hours']:.3f} h at {stats['fps']} fps")
    lines.append("\nper kind and split:")
    for kind, per in stats["by_kind"].items():
        for s, v in per.items():
            lines.append(f"  {kind:<6} {s:<5} {v['episodes']:>5} episodes {v['frames']:>9} frames {v['hours']:>8.3f} h")
    for name in ("train", "test", "total"):
        s = stats[name]
        lines.append(f"\n[{name}] {s['episodes']} episodes, {s['frames']} frames, {s['hours']:.3f} h")
        h = s["length_hist"]
        lines.append("  episode length histogram (frames):")
        for lo, hi, c in zip(h["edges"], h["edges"][1:], h["counts"]):
            lines.append(f"    [{lo:>6.0f}, {hi:>6.0f}) {c}")
        lines.append("  label rates: " + ", ".join(f"{k}={v:.3f}" for k, v in s["label_rates"].items()))
        for axis in ("x", "y"):
            lines.append(
                f"  mouse {axis} bins (deg: frames): "
                + ", ".join(f"{k}: {v}" for k, v in s[f"mouse_{axis}_hist"].items())
            )
        lines.append(f"  all-zero action fraction: {s['all_zero_frac']:.4f}; idle fraction: {s['idle_frac']:.4f}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", help="assembled dataset directory ($S/datasets/<name>)")
    p.add_argument("--hdf5", help="HDF5 root for --summary ($S/hdf5)")
    p.add_argument("--summary", action="store_true", help="print only the hours total of --hdf5")
    p.add_argument("--json", help="also write the stats to this file")
    p.add_argument("--figure", help="write the stats figure to this file (#26)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Print the stats.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    args = build_parser().parse_args(argv)
    if args.summary and not args.hdf5:
        raise SystemExit("--summary needs --hdf5")
    if not args.summary and not args.dataset:
        raise SystemExit("give a dataset directory, or --hdf5 with --summary")
    if args.summary:
        # TODO(#24): the --hdf5 --summary running total.
        raise NotImplementedError("TODO(#24): dataset_stats --summary")
    stats = compute_stats(Path(args.dataset))
    print(format_stats(stats))
    if args.json:
        Path(args.json).write_text(json.dumps(stats, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
