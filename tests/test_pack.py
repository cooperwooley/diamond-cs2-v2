"""Tests for cs2wm.pipeline.pack."""

import hashlib
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from cs2wm.config import load_config
from cs2wm.pipeline import pack
from cs2wm.pipeline.pack import (
    append_manifest,
    is_fullres_selected,
    read_episode,
    to_low_res,
    write_episode,
)

H, W = 150, 280


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def gradient_frames(n: int, seed: int = 0) -> np.ndarray:
    """Smooth random per-channel gradients, uint8 [n, H, W, 3]."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.empty((n, H, W, 3), dtype=np.uint8)
    for i in range(n):
        for c in range(3):
            a, b, d = rng.uniform(0, 1, 3)
            out[i, :, :, c] = (30 + 195 * (a * yy / H + b * xx / W + d * 0.0) / max(a + b, 1e-3) * 0.5).clip(0, 255)
    return out


def bar_frames(n: int) -> np.ndarray:
    """Hard black/white vertical bars with irregular widths (bicubic overshoots)."""
    cols = np.zeros(W, dtype=np.uint8)
    for x in range(W):
        cols[x] = 255 if (x // 7) % 2 else 0
    img = np.broadcast_to(cols[None, :, None], (H, W, 3))
    return np.repeat(img[None], n, axis=0).copy()


def make_actions(n: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, (n, 51)).astype(np.float32)


def meta(episode_id: str = "ep0") -> dict:
    return {
        "episode_id": episode_id,
        "demo_id": "demo0",
        "kind": "dm",
        "player": None,
        "start_tick": 128,
        "cs2_build": "13000",
        "demoparser2_version": "0.42.0",
        "cs2wm_git_sha": "abc1234",
    }


def v1_reference(frames_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v1 create_low_res_dataset: returns (wrapped uint8 BGR HWC, pre-cast float p in BGR HWC)."""
    x = torch.stack([torch.tensor(f).flip(2).permute(2, 0, 1).div(255).mul(2).sub(1) for f in frames_bgr])
    y = TF.resize(x, (30, 56), interpolation=InterpolationMode.BICUBIC)
    p = y.add(1).div(2).mul(255)
    return p.byte().flip(1).permute(0, 2, 3, 1).numpy(), p.flip(1).permute(0, 2, 3, 1).numpy()


# 1. to_low_res basics


def test_to_low_res_shape_dtype_and_bgr_order() -> None:
    frames = np.zeros((3, H, W, 3), dtype=np.uint8)
    frames[..., 0] = 255  # pure blue in BGR
    out = to_low_res(frames)
    assert out.shape == (3, 30, 56, 3) and out.dtype == np.uint8
    # v1's .byte() truncates, so 255 may come back as 254
    assert (out[..., 0] >= 254).all() and (out[..., 1:] == 0).all()


def test_to_low_res_chunking_matches_unchunked(monkeypatch) -> None:
    frames = gradient_frames(7)
    whole = to_low_res(frames)
    monkeypatch.setattr(pack, "_CHUNK", 3)
    assert np.array_equal(to_low_res(frames), whole)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, H, W, 3), dtype=np.float32),
        np.zeros((H, W, 3), dtype=np.uint8),
        np.zeros((2, H, W, 4), dtype=np.uint8),
    ],
)
def test_to_low_res_rejects_bad_input(bad) -> None:
    with pytest.raises(ValueError):
        to_low_res(bad)


# 2. v1 equivalence


def test_matches_v1_on_smooth_images() -> None:
    frames = gradient_frames(4)
    v1, p = v1_reference(frames)
    ours = to_low_res(frames)
    in_range = (p >= 0) & (p <= 255)
    assert in_range.all()
    assert np.array_equal(ours[in_range], v1[in_range])


def test_clamps_where_v1_wraps() -> None:
    frames = bar_frames(2)
    v1, p = v1_reference(frames)
    ours = to_low_res(frames)
    out_of_range = (p < 0) | (p > 255)
    assert out_of_range.any(), "bar pattern must make bicubic overshoot"
    print(f"hard-edge wrap fraction: {out_of_range.mean():.4f}")
    assert np.array_equal(ours[out_of_range], np.clip(p, 0, 255)[out_of_range].astype(np.uint8))
    assert not np.array_equal(ours[out_of_range], v1[out_of_range])
    # v1's resize call omits antialias, but torchvision's default is antialias=True, so every in-range pixel must
    # still be bit-identical: the clamp is the only intended difference.
    assert np.array_equal(ours[~out_of_range], v1[~out_of_range])


def test_v1_reader_path_works_on_v2_file(tmp_path, cfg) -> None:
    res = write_episode(gradient_frames(2), make_actions(2), meta(), tmp_path, cfg)
    with h5py.File(res.low_res_path, "r") as f:
        t = torch.tensor(f["frame_0_x"][:]).flip(2).permute(2, 0, 1)
    assert tuple(t.shape) == (3, 30, 56)


# 3. round trip


def test_round_trip(tmp_path, cfg) -> None:
    n = 5
    frames, actions = gradient_frames(n), make_actions(n)
    res = write_episode(frames, actions, meta(), tmp_path / "out", cfg)
    assert res.low_res_path == tmp_path / "out" / "ep0.hdf5" and res.fullres_path is None and res.n_frames == n
    got_frames, got_actions, attrs = read_episode(res.low_res_path)
    assert np.array_equal(got_frames, to_low_res(frames))
    assert got_actions.dtype == np.float32 and np.array_equal(got_actions, actions)
    with h5py.File(res.low_res_path, "r") as f:
        expected = {f"frame_{i}_{c}" for i in range(n) for c in "xy"}
        assert set(f.keys()) == expected
        assert f["frame_0_x"].dtype == np.uint8 and f["frame_0_x"].shape == (30, 56, 3)
        assert f["frame_0_y"].dtype == np.float32 and f["frame_0_y"].shape == (51,)
        assert f["frame_0_x"].compression is None
    for key in (
        "n_frames",
        "schema_version",
        "encoding",
        "fps",
        "tickrate",
        "config_sha256",
        "episode_id",
        "demo_id",
        "kind",
    ):
        assert key in attrs
    assert attrs["n_frames"] == n and attrs["encoding"] == "raw" and attrs["player"] == ""
    assert attrs["schema_version"] == pack.SCHEMA_VERSION and attrs["fps"] == 16 and attrs["tickrate"] == 64
    assert (attrs["height"], attrs["width"]) == (30, 56)
    assert type(attrs["n_frames"]) is int


# 4. validation


def test_validation_errors(tmp_path, cfg) -> None:
    small = load_config(overrides=["episodes.min_frames=1", "episodes.max_frames=8"])
    frames, actions = gradient_frames(9), make_actions(9)
    with pytest.raises(ValueError, match="1..8"):
        write_episode(frames, actions, meta(), tmp_path, small)
    with pytest.raises(ValueError):  # n = 0
        write_episode(frames[:0], actions[:0], meta(), tmp_path, cfg)
    f3, a3 = gradient_frames(3), make_actions(3)
    with pytest.raises(ValueError, match="actions"):
        write_episode(f3, make_actions(2), meta(), tmp_path, cfg)
    with pytest.raises(ValueError, match="actions"):
        write_episode(f3, make_actions(3)[:, :50], meta(), tmp_path, cfg)
    nan = a3.copy()
    nan[1, 4] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        write_episode(f3, nan, meta(), tmp_path, cfg)
    with pytest.raises(ValueError, match="frames"):
        write_episode(f3[:, :100], a3, meta(), tmp_path, cfg)
    bad = meta()
    del bad["kind"]
    with pytest.raises(ValueError, match="kind"):
        write_episode(f3, a3, bad, tmp_path, cfg)
    for key in ("cs2_build", "demoparser2_version", "cs2wm_git_sha", "player", "start_tick", "demo_id"):
        missing = meta()
        del missing[key]
        with pytest.raises(ValueError, match=key):
            write_episode(f3, a3, missing, tmp_path, cfg)
    for bad_id in ("../evil", "a/b", "..", "a..b", ".hidden", "-x", ""):
        with pytest.raises(ValueError, match="episode_id"):
            write_episode(f3, a3, meta(bad_id), tmp_path, cfg)
    with pytest.raises(ValueError, match="kind"):
        write_episode(f3, a3, {**meta(), "kind": "bot"}, tmp_path, cfg)
    with pytest.raises(ValueError, match="scalar"):
        write_episode(f3, a3, {**meta(), "tags": [1, 2]}, tmp_path, cfg)
    with pytest.raises(ValueError, match="reserved"):
        write_episode(f3, a3, {**meta(), "n_frames": 3}, tmp_path, cfg)
    assert list(tmp_path.iterdir()) == []


# 5. full-res subset


def find_id(selected: bool) -> str:
    return next(f"ep{i}" for i in range(10_000) if is_fullres_selected(f"ep{i}") == selected)


def test_selection_deterministic_and_balanced() -> None:
    ids = [f"demo_{i}_ep_{i % 7}" for i in range(10_000)]
    first = [is_fullres_selected(i) for i in ids]
    assert first == [is_fullres_selected(i) for i in ids]
    assert 0.08 <= sum(first) / len(ids) <= 0.12
    assert is_fullres_selected("x", mod=1)


def test_fullres_written_for_selected_id(tmp_path, cfg) -> None:
    eid = find_id(True)
    n = 3
    frames = gradient_frames(n)
    actions = make_actions(n)
    res = write_episode(frames, actions, meta(eid), tmp_path / "lo", cfg, fullres_dir=tmp_path / "hi")
    assert res.fullres_path == tmp_path / "hi" / f"{eid}.hdf5" and res.fullres_path.exists()
    got, got_actions, attrs = read_episode(res.fullres_path)
    assert attrs["encoding"] == "jpeg" and attrs["jpeg_quality"] == cfg.pack.jpeg_quality
    assert (attrs["height"], attrs["width"]) == (H, W)
    assert got.shape == frames.shape and np.array_equal(got_actions, actions)
    with h5py.File(res.fullres_path, "r") as f:
        assert f["frame_0_x"].ndim == 1 and f["frame_0_x"].dtype == np.uint8
    for a, b in zip(frames, got):
        assert cv2.PSNR(a, b) > 30


def test_no_fullres_for_unselected_or_no_dir(tmp_path, cfg) -> None:
    frames, actions = gradient_frames(2), make_actions(2)
    eid = find_id(False)
    res = write_episode(frames, actions, meta(eid), tmp_path / "lo", cfg, fullres_dir=tmp_path / "hi")
    assert res.fullres_path is None and not (tmp_path / "hi").exists()
    sel = find_id(True)
    res = write_episode(frames, actions, meta(sel), tmp_path / "lo2", cfg, fullres_dir=None)
    assert res.fullres_path is None and list((tmp_path / "lo2").iterdir()) == [res.low_res_path]


def test_read_episode_errors(tmp_path) -> None:
    p = tmp_path / "bad.hdf5"
    with h5py.File(p, "w") as f:
        f.attrs["n_frames"] = 1
        f.attrs["encoding"] = "png"
    with pytest.raises(ValueError, match="bad.hdf5"):
        read_episode(p)
    with h5py.File(p, "w") as f:
        f.attrs["n_frames"] = 1
        f.attrs["encoding"] = "raw"
    with pytest.raises(KeyError, match="bad.hdf5"):
        read_episode(p)


# 6. atomicity


@pytest.mark.parametrize("fullres", [False, True])
def test_failed_write_leaves_nothing(tmp_path, cfg, monkeypatch, fullres) -> None:
    eid = find_id(fullres)
    out, hi = tmp_path / "lo", tmp_path / "hi"
    frames, actions = gradient_frames(4), make_actions(4)
    good = write_episode(frames, actions, meta(eid), out, cfg, fullres_dir=hi)
    before = {p: p.read_bytes() for p in (out / f"{eid}.hdf5", *hi.glob("*.hdf5"))}

    real = h5py.Group.create_dataset
    calls = {"n": 0}

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("boom")
        return real(self, *a, **k)

    monkeypatch.setattr(h5py.Group, "create_dataset", flaky)
    with pytest.raises(RuntimeError, match="boom"):
        write_episode(make_frames_other(), actions, meta(eid), out, cfg, fullres_dir=hi)
    monkeypatch.undo()

    assert {p.name for p in out.iterdir()} == {f"{eid}.hdf5"}
    assert {p: p.read_bytes() for p in (out / f"{eid}.hdf5", *hi.glob("*.hdf5"))} == before
    assert not list(tmp_path.rglob("*.tmp"))
    assert good.sha256


@pytest.mark.parametrize("fullres", [False, True])
def test_low_res_rename_failure_keeps_old_low_res(tmp_path, cfg, monkeypatch, fullres) -> None:
    eid = find_id(fullres)
    out, hi = tmp_path / "lo", tmp_path / "hi"
    actions = make_actions(4)
    write_episode(gradient_frames(4), actions, meta(eid), out, cfg, fullres_dir=hi)
    low = out / f"{eid}.hdf5"
    before = low.read_bytes()

    real = os.replace
    order: list[str] = []

    def flaky(src, dst):
        order.append(Path(dst).parent.name)
        if Path(dst) == low:
            raise OSError("rename failed")
        real(src, dst)

    monkeypatch.setattr(pack.os, "replace", flaky)
    with pytest.raises(OSError, match="rename failed"):
        write_episode(gradient_frames(4, seed=99), actions, meta(eid), out, cfg, fullres_dir=hi)
    monkeypatch.undo()

    assert order == (["hi", "lo"] if fullres else ["lo"])  # low-res is renamed last
    assert low.read_bytes() == before
    assert {p.name for p in out.iterdir()} == {low.name}
    assert not list(tmp_path.rglob("*.tmp"))


def make_frames_other() -> np.ndarray:
    return gradient_frames(4, seed=99)


def test_failed_first_write_creates_no_files(tmp_path, cfg, monkeypatch) -> None:
    monkeypatch.setattr(h5py.Group, "create_dataset", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        write_episode(gradient_frames(2), make_actions(2), meta(), tmp_path / "o", cfg)
    assert not list((tmp_path / "o").rglob("*"))


# 7. checksums


def test_sha256_matches_hashlib(tmp_path, cfg) -> None:
    eid = find_id(True)
    res = write_episode(gradient_frames(2), make_actions(2), meta(eid), tmp_path / "lo", cfg, tmp_path / "hi")
    assert res.low_res_path != res.fullres_path
    assert set(res.sha256) == {res.low_res_path.name, f"fullres/{res.fullres_path.name}"}
    assert res.sha256[res.low_res_path.name] == hashlib.sha256(res.low_res_path.read_bytes()).hexdigest()
    assert res.sha256[f"fullres/{res.fullres_path.name}"] == hashlib.sha256(res.fullres_path.read_bytes()).hexdigest()


def test_sha256_file_multi_chunk(tmp_path, monkeypatch) -> None:
    p = tmp_path / "blob"
    p.write_bytes(bytes(range(256)) * 100)
    monkeypatch.setattr(pack, "_HASH_CHUNK", 1000)
    assert pack.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


# 8. manifest


def test_append_manifest(tmp_path) -> None:
    path = tmp_path / "a" / "b" / "manifest.jsonl"
    append_manifest(path, {"z": 1, "a": "x"})
    append_manifest(path, {"file": "e.hdf5", "frames": 3})
    lines = path.read_text().splitlines()
    assert lines[0] == '{"a": "x", "z": 1}'
    assert json.loads(lines[1]) == {"file": "e.hdf5", "frames": 3}
    assert len(lines) == 2
