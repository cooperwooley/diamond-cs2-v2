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
    assert g["wasd"] == {"n": 10, "correct": 9, "accuracy": 0.9, "min_accuracy": 0.9, "low_n": False, "pass": True}
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


def write_png_dir(d: Path, frames: np.ndarray) -> Path:
    d.mkdir()
    for i, f in enumerate(frames):
        cv2.imwrite(str(d / f"seg_{i}.png"), f)
    return d


def test_image_dir_with_trailing_extra_frame(tmp_path, cfg, episode):
    low, _, frames = episode
    extra = np.concatenate([frames, frames[:1]])  # n + 1 images: the renderer's trailing T_n frame
    d = write_png_dir(tmp_path / "frames", extra)
    out = la.render_overlay_clip(low, d, 0, None, tmp_path / "clip.mp4", cfg)
    assert read_back(out)[0] == N
    d2 = write_png_dir(tmp_path / "frames2", np.concatenate([frames, frames[:2]]))
    with pytest.raises(ValueError, match=f"{N + 2} frames but the episode has {N}"):
        la.render_overlay_clip(low, d2, 0, None, tmp_path / "clip.mp4", cfg)


def test_default_length_does_not_warn(tmp_path, cfg, episode, capsys):
    low, full, _ = episode
    la.render_overlay_clip(low, full, 0, None, tmp_path / "clip.mp4", cfg)
    assert capsys.readouterr().err == ""


def test_scoring_accepts_excel_bom(tmp_path):
    sheet = write_sheet(tmp_path / "s.csv", [("wasd", "1"), ("wasd", "0")])
    sheet.write_bytes(b"\xef\xbb\xbf" + sheet.read_bytes())
    acc = la.score_review_sheet(sheet, tmp_path / "o.json")
    assert acc["wasd"] == 0.5


def test_scoring_pools_sheets_and_flags_low_n(tmp_path, capsys):
    a = write_sheet(tmp_path / "a.csv", [("wasd", "1")] * 6 + [("jump", "1")])
    b = write_sheet(tmp_path / "b.csv", [("wasd", "1")] * 5 + [("wasd", "0")] * 1)
    out = tmp_path / "o.json"
    la.score_review_sheet([a, b], out)
    data = json.loads(out.read_text())
    assert data["sheets"] == [str(a), str(b)]
    assert (data["groups"]["wasd"]["n"], data["groups"]["wasd"]["correct"]) == (12, 11)
    assert data["groups"]["wasd"]["low_n"] is False
    assert data["groups"]["jump"]["low_n"] is True
    assert "low n" in capsys.readouterr().out


def test_crouch_rows(spec):
    rows = [encode(["ctrl"] if 2 <= t < 4 else [], False, False, 0.0, 0.0, spec) for t in range(6)]
    got = [(r.frame, r.slot, r.label) for r in la.extract_transitions(np.stack(rows), 0, 6, spec)]
    assert got == [(2, "crouch", "ctrl down"), (4, "crouch", "ctrl up")]


def test_mouse_rows_signs(spec):
    cases = [(5.0, 2.0, "right", "down"), (-5.0, -2.0, "left", "up"), (0.0, 0.0, "none", "none")]
    actions = np.stack([encode([], False, False, x, y, spec) for x, y, _, _ in cases])
    rows = la.random_mouse_rows(actions, 0, len(cases), spec, seed=0)
    by_frame = {(r.frame, r.slot): r.label for r in rows}
    for t, (_, _, dx, dy) in enumerate(cases):
        assert by_frame[(t, "mouse_x")].endswith(f"({dx})")
        assert by_frame[(t, "mouse_y")].endswith(f"({dy})")
    assert "+" in by_frame[(0, "mouse_x")] and "-" in by_frame[(1, "mouse_x")]


def test_overlay_alignment(tmp_path, cfg, spec):
    """Video frame j shows frame index start + j and the keys of actions[start + j]."""
    n = 8
    key_cycle = ["w", "a", "s", "d", "space", "ctrl", "shift", "r"]
    actions = np.stack([encode([key_cycle[t]], t % 2 == 0, False, 0.0, 0.0, spec) for t in range(n)])
    gray = np.full((n, H, W, 3), 90, np.uint8)
    res = write_episode(gray, actions, meta("align"), tmp_path / "low", cfg)
    frames_dir = write_png_dir(tmp_path / "frames", gray)
    start, count = 2, 5
    out = la.render_overlay_clip(res.low_res_path, frames_dir, start, count, tmp_path / "clip.mp4", cfg)
    cap = cv2.VideoCapture(str(out))
    for j in range(count):
        ok, img = cap.read()
        assert ok
        expected = [la.draw_overlay(gray[0], actions[t], t, spec).astype(np.float32) for t in range(n)]
        errs = [np.abs(img.astype(np.float32) - e).mean() for e in expected]
        assert int(np.argmin(errs)) == start + j
    cap.release()


def test_overlay_key_boxes_colour(spec):
    vec = encode(["w", "r"], True, False, 0.0, 0.0, spec)
    img = la.draw_overlay(np.full((H, W, 3), 90, np.uint8), vec, 0, spec)
    y = img.shape[0] - 34 + 3  # inside the box row, above the label text
    pressed = {tuple(img[y, x]) for x in range(10, 400) if tuple(img[y, x]) == la._ON}
    assert pressed  # at least one green (pressed) box
    assert la._OFF in {tuple(img[y, x]) for x in range(10, 400)}


def test_cli_render_and_score(tmp_path, episode):
    from cs2wm import cli

    low, _, frames = episode
    d = write_png_dir(tmp_path / "frames", np.concatenate([frames, frames[:1]]))
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "cs2wm.yaml"
    clip = tmp_path / "out" / "clip.mp4"
    argv = ["--config", str(cfg_path), "label-check", "render", "--episode", str(low), "--full-res-frames", str(d)]
    assert cli.main([*argv, "--out", str(clip)]) == 0
    sheet = tmp_path / "out" / "clip_labels.csv"
    assert clip.exists() and sheet.exists()
    assert read_back(clip)[0] == N
    with sheet.open(newline="") as f:
        rows = list(csv.DictReader(f))
    with sheet.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=la.SHEET_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "ok": "1"})
    res = tmp_path / "res" / "acc.json"
    assert cli.main(["label-check", "score", str(sheet), str(sheet), "--out", str(res)]) == 0
    data = json.loads(res.read_text())
    assert data["sheets"] == [str(sheet), str(sheet)]
    assert data["groups"]["wasd"]["n"] == 2 * 2
