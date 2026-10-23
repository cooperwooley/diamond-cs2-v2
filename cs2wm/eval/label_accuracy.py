"""Label accuracy hand-check (#18): overlay clip, review sheet and scoring.

``render_overlay_clip`` writes an 8 fps (half speed) video with the label overlay (keys, LMB/RMB, mouse bins,
frame index); ``make_review_sheet`` writes ``clip_labels.csv`` (one row per label transition plus 200 random frames
for the mouse; columns ``frame, slot, label, ok, note``); ``score_review_sheet`` reports accuracy per slot group.

Filling in the sheet: ``ok`` is empty on every row. Look at the frame and its neighbours and enter

* ``1`` if the label matches what the player visibly did, ``0`` if not (every slot);
* on ``mouse_x`` rows only, ``m`` if the turn direction is right but the magnitude is off by more than one bin.

Rows left blank are ignored by the scoring (and counted in its report). A ``note`` can be added to wrong rows.

Slots and how they are judged (``slot`` column -> score group):

* ``wasd``, ``jump``, ``crouch``, ``fire``, ``reload``, ``switch``: one group each, ``ok == 1`` is correct. Rows are
  label transitions: key down/up for W/A/S/D, ctrl and LMB (fire); rising edge only for space (jump), R (reload)
  and slot keys 1/2/3 (switch).
* ``mouse_y``: ``ok == 1`` is correct (direction and magnitude within the bin tolerance).
* ``mouse_x``: one row per random frame feeds two groups. ``mouse-x sign`` counts the row correct when the
  direction is right (``ok`` is ``1`` or ``m``); ``mouse-x bin +-1`` counts it correct only when direction and
  magnitude are right (``ok == 1``). ``0`` fails both, so ``n`` is the same for both groups.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from cs2wm.actions import ActionSpec, decode
from cs2wm.config import load_config
from cs2wm.pipeline.pack import read_episode

UPSCALE = 3  # 280x150 -> 840x450, enough for the overlay text to be legible in VLC
N_RANDOM_FRAMES = 200
SHEET_COLUMNS = ("frame", "slot", "label", "ok", "note")
IMAGE_SUFFIXES = (".png", ".tga", ".jpg", ".jpeg", ".bmp")

# Success criteria from #18: every group >= 0.90, mouse-x sign >= 0.98.
DEFAULT_MIN_ACCURACY = 0.90
MIN_ACCURACY = {"mouse_x_sign": 0.98}

# Sheet slot -> score groups it feeds, in report order.
GROUPS = ("wasd", "jump", "crouch", "fire", "reload", "switch", "mouse_x_sign", "mouse_x_bin", "mouse_y")
_SLOT_ORDER = ("wasd", "jump", "crouch", "fire", "reload", "switch", "mouse_x", "mouse_y")

# Key (or click) name -> (slot, emit both down and up rows). Shift and RMB are not scored.
_TRANSITION_KEYS: dict[str, tuple[str, bool]] = {
    "w": ("wasd", True),
    "a": ("wasd", True),
    "s": ("wasd", True),
    "d": ("wasd", True),
    "space": ("jump", False),
    "ctrl": ("crouch", True),
    "lmb": ("fire", True),
    "r": ("reload", False),
    "1": ("switch", False),
    "2": ("switch", False),
    "3": ("switch", False),
}

_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_ON = (60, 200, 60)
_OFF = (70, 70, 70)
_ARROW = (0, 200, 255)


@dataclass(frozen=True)
class Row:
    """One review-sheet row.

    Attributes:
        frame: Episode frame index.
        slot: Score slot (see ``_SLOT_ORDER``).
        label: Human-readable claim to check.
    """

    frame: int
    slot: str
    label: str


def _window(n_total: int, start: int, n_frames: int) -> int:
    """Validate the clip window and return its (possibly clamped) length."""
    if start < 0 or start >= n_total:
        raise ValueError(f"--start {start} is outside the episode ({n_total} frames)")
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    n = min(n_frames, n_total - start)
    if n < n_frames:
        print(f"warning: episode has {n_total} frames, clip clamped to {n} (start={start})", file=sys.stderr)
    return n


def _mouse_bins(vec: np.ndarray, spec: ActionSpec) -> tuple[int, int]:
    """Active (mouse-x, mouse-y) bin indices of one action vector."""
    return int(np.argmax(vec[spec.mouse_x_slice])), int(np.argmax(vec[spec.mouse_y_slice]))


def _direction(deg: float, horizontal: bool) -> str:
    if deg == 0:
        return "none"
    if horizontal:
        return "right" if deg > 0 else "left"
    return "down" if deg > 0 else "up"


def extract_transitions(actions: np.ndarray, start: int, n_frames: int, spec: ActionSpec) -> list[Row]:
    """List the label transitions inside ``[start, start + n_frames)``.

    A transition at frame ``t`` compares frame ``t`` with ``t - 1``. At episode frame 0 there is no previous
    frame, so nothing is emitted for it.

    Args:
        actions: ``float[n, 51]`` labels of the whole episode.
        start: First frame of the clip.
        n_frames: Number of frames in the clip.
        spec: Action layout.

    Returns:
        Rows sorted by ``(frame, slot, label)``.
    """
    pressed = (
        np.concatenate(
            [actions[:, : len(spec.keys)], actions[:, [spec.idx_lmb]]],
            axis=1,
        )
        >= 0.5
    )
    names = (*spec.keys, "lmb")
    rows: list[Row] = []
    for name, (slot, both_edges) in _TRANSITION_KEYS.items():
        col = pressed[:, names.index(name)]
        for t in range(max(start, 1), start + n_frames):
            if col[t] and not col[t - 1]:
                if slot == "switch":
                    label = f"switch to slot {name}"
                elif both_edges:
                    label = f"{name} down"
                else:
                    label = f"{slot} ({name})"
                rows.append(Row(t, slot, label))
            elif both_edges and col[t - 1] and not col[t]:
                rows.append(Row(t, slot, f"{name} up"))
    return sorted(rows, key=lambda r: (r.frame, _SLOT_ORDER.index(r.slot), r.label))


def random_mouse_rows(actions: np.ndarray, start: int, n_frames: int, spec: ActionSpec, seed: int) -> list[Row]:
    """Mouse-x and mouse-y rows for seeded random frames of the clip.

    Args:
        actions: ``float[n, 51]`` labels of the whole episode.
        start: First frame of the clip.
        n_frames: Number of frames in the clip.
        spec: Action layout.
        seed: RNG seed.

    Returns:
        Two rows per sampled frame (``N_RANDOM_FRAMES`` frames, or every frame of a shorter clip).
    """
    rng = np.random.default_rng(seed)
    frames = np.sort(rng.choice(np.arange(start, start + n_frames), size=min(N_RANDOM_FRAMES, n_frames), replace=False))
    rows: list[Row] = []
    for t in frames:
        t = int(t)
        dec = decode(actions[t], spec)
        bx, by = _mouse_bins(actions[t], spec)
        rows.append(Row(t, "mouse_x", f"x bin {bx} = {dec.mouse_x_deg:+.2f} deg ({_direction(dec.mouse_x_deg, True)})"))
        rows.append(
            Row(t, "mouse_y", f"y bin {by} = {dec.mouse_y_deg:+.2f} deg ({_direction(dec.mouse_y_deg, False)})")
        )
    return rows


def make_review_sheet(
    episode: Path, start: int, n_frames: int, out_csv: Path, seed: int = 0, cfg: DictConfig | None = None
) -> Path:
    """Write the review sheet: one row per label transition (key down/up, fire, reload, switch, jump) plus 200
    random frames for mouse direction and magnitude.

    Rows are sorted by frame (then slot, label), so the same episode, window and seed give the same file.

    Args:
        episode: Packed episode HDF5 (labels).
        start: First frame.
        n_frames: Number of frames (clamped to the end of the episode).
        out_csv: Output ``clip_labels.csv`` with columns ``frame, slot, label, ok, note``; ``ok`` and ``note`` are
            empty.
        seed: Seed for the random frames.
        cfg: A validated cs2wm config; defaults to ``load_config()``.

    Returns:
        ``out_csv``.
    """
    spec = ActionSpec.from_config(cfg if cfg is not None else load_config())
    _, actions, _ = read_episode(Path(episode))
    n = _window(len(actions), start, n_frames)
    rows = extract_transitions(actions, start, n, spec) + random_mouse_rows(actions, start, n, spec, seed)
    rows.sort(key=lambda r: (r.frame, _SLOT_ORDER.index(r.slot), r.label))
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SHEET_COLUMNS)
        for r in rows:
            writer.writerow([r.frame, r.slot, r.label, "", ""])
    return out_csv


def _frame_number(path: Path) -> int:
    """The last integer in a file name, which orders the frames."""
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        raise ValueError(f"{path}: file name has no frame number")
    return int(nums[-1])


def load_fullres_frames(source: Path, start: int, n_frames: int, expected: int) -> np.ndarray:
    """Load the clip's full-res frames from a directory of images or from a full-res episode HDF5.

    Args:
        source: A directory of png/tga/jpg/bmp files (sorted by the integer in the name, file ``i`` is episode
            frame ``i``), or a full-res episode HDF5 written by ``pack``.
        start: First frame.
        n_frames: Number of frames.
        expected: The episode's frame count; the source must hold exactly this many frames.

    Returns:
        BGR ``uint8[n_frames, H, W, 3]``.

    Raises:
        ValueError: On a frame-count mismatch, duplicate frame numbers, or an unreadable image.
        FileNotFoundError: If ``source`` does not exist or the directory has no images.
    """
    source = Path(source)
    if source.is_dir():
        files = sorted((p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=_frame_number)
        if not files:
            raise FileNotFoundError(f"{source}: no {'/'.join(IMAGE_SUFFIXES)} files")
        nums = [_frame_number(p) for p in files]
        if len(set(nums)) != len(nums):
            raise ValueError(f"{source}: duplicate frame numbers in file names")
        if len(files) != expected:
            raise ValueError(f"{source}: {len(files)} frames but the episode has {expected}")
        out = []
        for p in files[start : start + n_frames]:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"{p}: cannot read image")
            out.append(img)
        return np.stack(out)
    if not source.is_file():
        raise FileNotFoundError(source)
    frames, _, attrs = read_episode(source)
    if attrs["encoding"] != "jpeg":
        raise ValueError(f"{source}: expected a full-res (jpeg) episode, got encoding {attrs['encoding']!r}")
    if len(frames) != expected:
        raise ValueError(f"{source}: {len(frames)} frames but the episode has {expected}")
    return frames[start : start + n_frames]


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.6, color=_WHITE) -> None:
    """Draw outlined text so it stays readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _arrow(
    img: np.ndarray,
    centre: tuple[int, int],
    offset_bins: int,
    step: int,
    horizontal: bool,
    label: str,
    at: tuple[int, int],
) -> None:
    """Draw a mouse arrow whose length is the bin's distance from the zero bin, plus its text."""
    cx, cy = centre
    d = offset_bins * step
    tip = (cx + d, cy) if horizontal else (cx, cy + d)
    if offset_bins == 0:
        cv2.circle(img, centre, 4, _ARROW, -1, cv2.LINE_AA)
    else:
        cv2.arrowedLine(img, centre, tip, _BLACK, 5, cv2.LINE_AA, tipLength=0.3)
        cv2.arrowedLine(img, centre, tip, _ARROW, 2, cv2.LINE_AA, tipLength=0.3)
    _text(img, label, at, 0.5)


def draw_overlay(frame_bgr: np.ndarray, vec: np.ndarray, index: int, spec: ActionSpec) -> np.ndarray:
    """Upscale a frame and draw its label overlay.

    Args:
        frame_bgr: ``uint8[H, W, 3]`` full-res frame.
        vec: ``float[51]`` action of this frame.
        index: Episode frame index to print.
        spec: Action layout.

    Returns:
        The annotated ``uint8[H * UPSCALE, W * UPSCALE, 3]`` image.
    """
    img = cv2.resize(frame_bgr, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]
    dec = decode(vec, spec)
    bx, by = _mouse_bins(vec, spec)
    zx = int(np.argmin(np.abs(spec.mouse_x_centers_deg)))
    zy = int(np.argmin(np.abs(spec.mouse_y_centers_deg)))

    _text(img, f"frame {index}", (10, 24), 0.7)

    # Key and click boxes along the bottom edge.
    labels = [(k.upper() if len(k) == 1 and k.isalpha() else k, k in dec.keys) for k in spec.keys]
    labels += [(spec.clicks[0].upper(), dec.lmb), (spec.clicks[1].upper(), dec.rmb)]
    x, y0, bh = 8, h - 34, 26
    for text, on in labels:
        bw = 14 * len(text) + 14
        cv2.rectangle(img, (x, y0), (x + bw, y0 + bh), _ON if on else _OFF, -1)
        cv2.rectangle(img, (x, y0), (x + bw, y0 + bh), _WHITE if on else _BLACK, 1)
        _text(img, text, (x + 7, y0 + 19), 0.5)
        x += bw + 4

    # Mouse arrows on the right: x pointing right/left, y pointing down/up (+y is look down).
    xa = (w - 150, 50)
    ya = (w - 40, 90)
    max_x = max(zx, len(spec.mouse_x_units) - 1 - zx)
    max_y = max(zy, len(spec.mouse_y_units) - 1 - zy)
    _arrow(img, xa, bx - zx, 60 // max_x, True, f"x bin {bx}: {dec.mouse_x_deg:+.2f} deg", (w - 230, 20))
    _arrow(img, ya, by - zy, 50 // max_y, False, f"y bin {by}: {dec.mouse_y_deg:+.2f} deg", (w - 230, 125))
    return img


def render_overlay_clip(episode: Path, frames_dir: Path, start: int, n_frames: int, out: Path, cfg: DictConfig) -> Path:
    """Render the half-speed label-overlay video.

    Args:
        episode: Packed episode HDF5 (labels).
        frames_dir: Full-res rendered frames: a directory of image files (file ``i`` is episode frame ``i``, ordered
            by the integer in the name) or a full-res episode HDF5 written by ``pack``.
        start: First frame.
        n_frames: Number of frames (1920 = 2 minutes); clamped to the end of the episode.
        out: Output ``clip.mp4`` (``mp4v``, ``timing.fps / 2`` fps).
        cfg: A validated cs2wm config.

    Returns:
        ``out``.

    Raises:
        ValueError: If the frame source does not hold exactly the episode's frames, or the writer cannot open.
    """
    spec = ActionSpec.from_config(cfg)
    _, actions, _ = read_episode(Path(episode))
    n = _window(len(actions), start, n_frames)
    frames = load_fullres_frames(Path(frames_dir), start, n, expected=len(actions))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(cfg.timing.fps) / 2, (w * UPSCALE, h * UPSCALE)
    )
    if not writer.isOpened():
        raise ValueError(f"cannot open a VideoWriter for {out}")
    try:
        for i, frame in enumerate(frames):
            writer.write(draw_overlay(frame, actions[start + i], start + i, spec))
    finally:
        writer.release()
    return out


def _judge(slot: str, ok: str, line: int) -> dict[str, bool]:
    """Map one filled ``ok`` to ``{group: correct}`` for the groups the slot feeds."""
    ok = ok.strip().lower()
    if ok in ("1", "1.0", "true"):
        verdict = "right"
    elif ok in ("0", "0.0", "false"):
        verdict = "wrong"
    elif ok == "m" and slot == "mouse_x":
        verdict = "magnitude"
    else:
        raise ValueError(f"row {line}: invalid ok {ok!r} for slot {slot!r} (use 1 or 0, or m on mouse_x rows)")
    if slot == "mouse_x":
        return {"mouse_x_sign": verdict != "wrong", "mouse_x_bin": verdict == "right"}
    return {slot: verdict == "right"}


def score_review_sheet(sheet: Path, out_json: Path) -> dict[str, float]:
    """Score a filled-in sheet per slot group.

    Groups: WASD, jump, crouch, fire, reload, switch, mouse-x sign, mouse-x bin +-1, mouse-y (see the module
    docstring for how rows are judged). Rows with an empty ``ok`` are skipped and counted. The result passes when
    every group has at least one judged row and meets its minimum (0.90, or 0.98 for mouse-x sign).

    Args:
        sheet: Review sheet with the ``ok`` column filled in.
        out_json: Output, e.g. ``results/label_accuracy.json``. Holds ``groups`` (``n, correct, accuracy,
            min_accuracy, pass`` each), ``blank_rows``, ``unscored`` groups and the overall ``pass``.

    Returns:
        Accuracy per scored slot group (groups without judged rows are omitted).

    Raises:
        ValueError: If a column is missing, a slot is unknown or an ``ok`` value is invalid.
    """
    counts = {g: [0, 0] for g in GROUPS}  # [n, correct]
    blank = 0
    with Path(sheet).open(newline="") as f:
        reader = csv.DictReader(f)
        missing = {"frame", "slot", "label", "ok"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{sheet}: missing columns {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            slot = (row["slot"] or "").strip()
            if slot not in _SLOT_ORDER:
                raise ValueError(f"row {line}: unknown slot {slot!r}")
            if not (row["ok"] or "").strip():
                blank += 1
                continue
            for group, correct in _judge(slot, row["ok"], line).items():
                counts[group][0] += 1
                counts[group][1] += int(correct)

    groups: dict[str, dict] = {}
    for g in GROUPS:
        n, correct = counts[g]
        threshold = MIN_ACCURACY.get(g, DEFAULT_MIN_ACCURACY)
        accuracy = correct / n if n else None
        groups[g] = {
            "n": n,
            "correct": correct,
            "accuracy": accuracy,
            "min_accuracy": threshold,
            "pass": accuracy is not None and accuracy >= threshold,
        }
    unscored = [g for g in GROUPS if groups[g]["n"] == 0]
    result = {
        "groups": groups,
        "blank_rows": blank,
        "unscored": unscored,
        "pass": all(g["pass"] for g in groups.values()),
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(format_table(result))
    return {g: v["accuracy"] for g, v in groups.items() if v["accuracy"] is not None}


def format_table(result: dict) -> str:
    """Render the scoring result as a readable table.

    Args:
        result: The dict written by ``score_review_sheet``.

    Returns:
        A multi-line string.
    """
    lines = [f"{'group':<14}{'n':>6}{'correct':>9}{'accuracy':>10}{'min':>7}  pass"]
    for g, v in result["groups"].items():
        acc = f"{v['accuracy']:.3f}" if v["accuracy"] is not None else "n/a"
        lines.append(
            f"{g:<14}{v['n']:>6}{v['correct']:>9}{acc:>10}{v['min_accuracy']:>7.2f}  {'yes' if v['pass'] else 'NO'}"
        )
    lines.append(f"blank rows ignored: {result['blank_rows']}")
    if result["unscored"]:
        lines.append(f"no judged rows for: {', '.join(result['unscored'])}")
    lines.append(f"overall: {'PASS' if result['pass'] else 'FAIL'}")
    return "\n".join(lines)
