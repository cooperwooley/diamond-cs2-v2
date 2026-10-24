"""Label accuracy hand-check (#18): overlay clip, review sheet and scoring.

``render_overlay_clip`` writes an 8 fps video with the label overlay (keys, LMB/RMB, mouse bins, frame index);
``make_review_sheet`` writes ``clip_labels.csv`` (one row per label transition plus 200 random frames; columns
``frame, slot, label, ok``); ``score_review_sheet`` reports accuracy per slot group.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig


def render_overlay_clip(episode: Path, frames_dir: Path, start: int, n_frames: int, out: Path, cfg: DictConfig) -> Path:
    """Render the half-speed label-overlay video.

    Args:
        episode: Packed episode HDF5 (labels).
        frames_dir: Full-res rendered frames.
        start: First frame.
        n_frames: Number of frames (1920 = 2 minutes).
        out: Output ``clip.mp4``.
        cfg: A validated cs2wm config.

    Returns:
        ``out``.
    """
    # TODO(#18): cv2.VideoWriter at 8 fps with the overlay drawn from cs2wm.actions.decode.
    raise NotImplementedError("TODO(#18): render_overlay_clip")


def make_review_sheet(episode: Path, start: int, n_frames: int, out_csv: Path, seed: int = 0) -> Path:
    """Write the review sheet: one row per label transition (key down/up, fire, reload, switch, jump) plus 200
    random frames for mouse direction and magnitude.

    Args:
        episode: Packed episode HDF5 (labels).
        start: First frame.
        n_frames: Number of frames.
        out_csv: Output ``clip_labels.csv`` with columns ``frame, slot, label, ok``.
        seed: Seed for the random frames.

    Returns:
        ``out_csv``.
    """
    raise NotImplementedError("TODO(#18): make_review_sheet")


def score_review_sheet(sheet: Path, out_json: Path) -> dict[str, float]:
    """Score a filled-in sheet per slot group.

    Groups: WASD, jump, crouch, fire, reload, switch, mouse-x sign, mouse-x bin +-1, mouse-y.

    Args:
        sheet: Review sheet with the ``ok`` column filled in.
        out_json: Output, e.g. ``results/label_accuracy.json``.

    Returns:
        Accuracy per slot group.
    """
    raise NotImplementedError("TODO(#18): score_review_sheet")
