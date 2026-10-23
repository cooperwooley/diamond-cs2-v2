"""Tests for cs2wm.eval.label_accuracy."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cs2wm.actions import ActionSpec, encode
from cs2wm.config import load_config
from cs2wm.eval import label_accuracy as la
from cs2wm.pipeline.pack import is_fullres_selected, write_episode

H, W = 150, 280
N = 12


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def spec(cfg):
    return ActionSpec.from_config(cfg)


def synthetic_actions(spec: ActionSpec, n: int = N) -> np.ndarray:
    """w held on frames 2..5, space on 3, lmb on 4..6, slot 2 from frame 7, r on 9; mouse sweeps."""
    rows = []
    for t in range(n):
        keys = []
        if 2 <= t <= 5:
            keys.append("w")
        if t == 3:
            keys.append("space")
        keys.append("1" if t < 7 else "2")
        if t == 9:
            keys.append("r")
        rows.append(encode(keys, 4 <= t <= 6, False, (t - 6) * 0.5, -t * 0.1, spec))
    return np.stack(rows)


def meta(episode_id: str) -> dict:
    return {
        "episode_id": episode_id,
        "demo_id": "demo0",
        "kind": "dm",
        "player": None,
        "start_tick": 0,
        "cs2_build": "1",
        "demoparser2_version": "0",
        "cs2wm_git_sha": "abc",
    }


def synthetic_frames(n: int = N) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(n, H, W, 3), dtype=np.uint8)


@pytest.fixture()
def episode(tmp_path, cfg, spec):
    """(low-res episode path, full-res episode path, frames) with the full-res twin guaranteed."""
    mod = int(cfg.pack.fullres_mod)
    eid = next(f"ep{i}" for i in range(1000) if is_fullres_selected(f"ep{i}", mod))
    frames = synthetic_frames()
    res = write_episode(frames, synthetic_actions(spec), meta(eid), tmp_path / "low", cfg, tmp_path / "full")
    assert res.fullres_path is not None
    return res.low_res_path, res.fullres_path, frames


def test_transitions(spec):
    rows = la.extract_transitions(synthetic_actions(spec), 0, N, spec)
    got = [(r.frame, r.slot, r.label) for r in rows]
    assert got == [
        (2, "wasd", "w down"),
        (3, "jump", "jump (space)"),
        (4, "fire", "lmb down"),
        (6, "wasd", "w up"),
        (7, "fire", "lmb up"),
        (7, "switch", "switch to slot 2"),
        (9, "reload", "reload (r)"),
    ]


def test_transitions_window_uses_previous_frame(spec):
    # Frames 3..6: the w press at frame 2 is outside; the lmb press at 4 and the w release at 6 are inside.
    rows = la.extract_transitions(synthetic_actions(spec), 3, 4, spec)
    assert [(r.frame, r.label) for r in rows] == [(3, "jump (space)"), (4, "lmb down"), (6, "w up")]


def test_sheet_deterministic_and_columns(tmp_path, episode):
    low = episode[0]
    a = la.make_review_sheet(low, 0, N, tmp_path / "a.csv", seed=3)
    b = la.make_review_sheet(low, 0, N, tmp_path / "b.csv", seed=3)
    assert a.read_bytes() == b.read_bytes()
    with a.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert tuple(reader.fieldnames) == ("frame", "slot", "label", "ok", "note")
    assert all(r["ok"] == "" for r in rows)
    frames = [int(r["frame"]) for r in rows]
    assert frames == sorted(frames)
    # 7 transitions + 2 mouse rows for each of the 12 frames (clip shorter than 200).
    assert len(rows) == 7 + 2 * N
    assert {r["slot"] for r in rows} >= {"wasd", "jump", "fire", "switch", "reload", "mouse_x", "mouse_y"}


def test_sheet_random_frames_seeded(tmp_path, cfg, spec):
    n = 400
    rng = np.random.default_rng(1)
    actions = np.stack([encode([], False, False, rng.normal() * 3, rng.normal(), spec) for _ in range(n)])
    # write_episode takes <= max_frames frames; labels only matter here.
    res = write_episode(np.zeros((n, H, W, 3), np.uint8), actions, meta("big"), tmp_path / "low", cfg)
    sheet = la.make_review_sheet(res.low_res_path, 0, n, tmp_path / "s.csv", seed=1)
    other = la.make_review_sheet(res.low_res_path, 0, n, tmp_path / "t.csv", seed=2)
    mouse = [r for r in csv.DictReader(sheet.open()) if r["slot"] == "mouse_x"]
    assert len(mouse) == la.N_RANDOM_FRAMES
    assert len({r["frame"] for r in mouse}) == la.N_RANDOM_FRAMES
    assert sheet.read_bytes() != other.read_bytes()


def write_sheet(path: Path, rows: list[tuple[str, str]]) -> Path:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(la.SHEET_COLUMNS)
        for i, (slot, ok) in enumerate(rows):
            w.writerow([i, slot, "x", ok, ""])
    return path


def test_scoring_with_blanks(tmp_path, capsys):
    rows = [("wasd", "1")] * 9 + [("wasd", "0")] + [("wasd", "")] * 3
    rows += [("jump", "1")]
    rows += [("mouse_x", "1")] * 48 + [("mouse_x", "m")] + [("mouse_x", "0")]
    rows += [("mouse_y", "1.0"), ("mouse_y", "0")]
    out = tmp_path / "res" / "acc.json"
    acc = la.score_review_sheet(write_sheet(tmp_path / "s.csv", rows), out)
    data = json.loads(out.read_text())
    g = data["groups"]
    assert g["wasd"] == {"n": 10, "correct": 9, "accuracy": 0.9, "min_accuracy": 0.9, "pass": True}
    assert data["blank_rows"] == 3
    # 50 judged mouse-x rows: sign wrong only on "0" (49/50), bin right only on "1" (48/50).
    assert (g["mouse_x_sign"]["n"], g["mouse_x_sign"]["correct"]) == (50, 49)
    assert g["mouse_x_sign"]["pass"] is True  # 0.98 meets the threshold exactly
    assert (g["mouse_x_bin"]["n"], g["mouse_x_bin"]["correct"]) == (50, 48)
    assert g["mouse_x_bin"]["pass"] is True
    assert g["mouse_y"]["accuracy"] == 0.5 and g["mouse_y"]["pass"] is False
    assert set(data["unscored"]) == {"crouch", "fire", "reload", "switch"}
    assert data["pass"] is False
    assert acc["wasd"] == 0.9 and "crouch" not in acc
    printed = capsys.readouterr().out
    assert "mouse_x_sign" in printed and "blank rows ignored: 3" in printed


def test_scoring_all_pass(tmp_path):
    rows = [(s, "1") for s in ("wasd", "jump", "crouch", "fire", "reload", "switch", "mouse_x", "mouse_y")]
    data = json.loads(
        (la.score_review_sheet(write_sheet(tmp_path / "s.csv", rows), tmp_path / "o.json"), tmp_path / "o.json")[
            1
        ].read_text()
    )
    assert data["pass"] is True and data["unscored"] == []


@pytest.mark.parametrize(
    "rows",
    [[("wasd", "yes")], [("jump", "m")], [("bogus", "1")]],
)
def test_scoring_rejects_bad_rows(tmp_path, rows):
    with pytest.raises(ValueError):
        la.score_review_sheet(write_sheet(tmp_path / "s.csv", rows), tmp_path / "o.json")


def read_back(path: Path) -> tuple[int, tuple[int, int], float]:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    count, size = 0, (0, 0)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        count += 1
        size = (img.shape[1], img.shape[0])
    cap.release()
    return count, size, fps


def test_render_from_hdf5(tmp_path, cfg, episode):
    low, full, _ = episode
    out = la.render_overlay_clip(low, full, 2, 8, tmp_path / "clip.mp4", cfg)
    assert out.exists()
    count, size, fps = read_back(out)
    assert (count, size) == (8, (W * la.UPSCALE, H * la.UPSCALE))
    assert fps == pytest.approx(cfg.timing.fps / 2)


def test_render_from_image_dir(tmp_path, cfg, episode):
    low, _, frames = episode
    d = tmp_path / "frames"
    d.mkdir()
    # Unpadded names: a plain string sort would put 10 before 2.
    for i, f in enumerate(frames):
        cv2.imwrite(str(d / f"seg_{i}.png"), f)
    out = la.render_overlay_clip(low, d, 0, N, tmp_path / "clip.mp4", cfg)
    assert read_back(out)[:2] == (N, (W * la.UPSCALE, H * la.UPSCALE))


def test_image_dir_sorted_by_integer(tmp_path):
    for i in (2, 10, 1):
        cv2.imwrite(str(tmp_path / f"f{i}.png"), np.full((4, 6, 3), i, np.uint8))
    got = la.load_fullres_frames(tmp_path, 0, 3, expected=3)
    assert [int(f[0, 0, 0]) for f in got] == [1, 2, 10]


def test_frame_count_mismatch(tmp_path, cfg, episode):
    low, _, frames = episode
    d = tmp_path / "frames"
    d.mkdir()
    for i, f in enumerate(frames[:-1]):
        cv2.imwrite(str(d / f"{i}.png"), f)
    with pytest.raises(ValueError, match=f"{N - 1} frames but the episode has {N}"):
        la.render_overlay_clip(low, d, 0, N, tmp_path / "clip.mp4", cfg)
    # A low-res (raw) episode is not a full-res source.
    with pytest.raises(ValueError, match="full-res"):
        la.render_overlay_clip(low, low, 0, N, tmp_path / "clip.mp4", cfg)


def test_window_clamped_and_start_checked(tmp_path, cfg, episode, capsys):
    low, full, _ = episode
    out = la.render_overlay_clip(low, full, 10, 100, tmp_path / "clip.mp4", cfg)
    assert read_back(out)[0] == N - 10
    assert "clamped" in capsys.readouterr().err
    with pytest.raises(ValueError, match="outside the episode"):
        la.render_overlay_clip(low, full, N, 1, tmp_path / "clip.mp4", cfg)
