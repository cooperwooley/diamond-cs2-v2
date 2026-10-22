"""pack: HDF5 episodes in v1's schema (30x56) plus a deterministic ~10% full-res JPEG subset (#12).

Each file holds ``frame_{i}_x`` (``uint8 (30, 56, 3)`` BGR) and ``frame_{i}_y`` (``float32 (51,)``) plus attrs
(``n_frames``, ``schema_version``, ``fps``, ``tickrate``, ``start_tick``, ``demo_id``, ``kind``, ``player``,
``cs2_build``, ``demoparser2_version``, ``cs2wm_git_sha``, ``config_sha256``); the provenance ones are required meta
keys. Writes are atomic (temp file, fsync, rename).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from cs2wm.actions import ACTION_DIM
from cs2wm.config import config_sha256

SCHEMA_VERSION = 1
_CHUNK = 256
_HASH_CHUNK = 1 << 20
_REQUIRED_META = (
    "episode_id",
    "demo_id",
    "kind",
    "player",
    "start_tick",
    "cs2_build",
    "demoparser2_version",
    "cs2wm_git_sha",
)
_EPISODE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_RESERVED_ATTRS = frozenset(
    {
        "n_frames",
        "schema_version",
        "encoding",
        "fps",
        "tickrate",
        "height",
        "width",
        "config_sha256",
        "jpeg_quality",
    }
)


@dataclass(frozen=True)
class PackResult:
    """Files written for one episode.

    Attributes:
        low_res_path: Low-res HDF5 file.
        fullres_path: Full-res JPEG HDF5 file, or None if the episode is not in the subset.
        n_frames: Frames written.
        sha256: Hex sha256 by file name; the full-res file is keyed ``fullres/<name>`` (same name as the low-res one).
    """

    low_res_path: Path
    fullres_path: Path | None
    n_frames: int
    sha256: dict[str, str]


def _check_frames(frames_bgr: np.ndarray) -> None:
    if not isinstance(frames_bgr, np.ndarray) or frames_bgr.dtype != np.uint8:
        raise ValueError(f"frames must be a uint8 ndarray, got {getattr(frames_bgr, 'dtype', type(frames_bgr))}")
    if frames_bgr.ndim != 4 or frames_bgr.shape[-1] != 3:
        raise ValueError(f"frames must have shape [n, H, W, 3], got {frames_bgr.shape}")


def to_low_res(frames_bgr: np.ndarray, size: tuple[int, int] = (30, 56)) -> np.ndarray:
    """Downscale like v1 ``build_dataset.py::create_low_res_dataset``, with the uint8 cast clamped.

    Steps: BGR to RGB, scale to [-1, 1], torchvision bicubic (antialiased) resize to ``size``,
    ``add(1).div(2).mul(255)``, clamp to [0, 255], uint8, back to BGR. The clamp is the only intentional difference
    from v1, whose ``.byte()`` wraps bicubic overshoot at hard edges (257.5 becomes 1).

    Args:
        frames_bgr: ``uint8[n, H, W, 3]`` frames.
        size: Target ``(H, W)``.

    Returns:
        ``uint8[n, *size, 3]`` BGR frames.

    Raises:
        ValueError: If the input is not a uint8 ``[n, H, W, 3]`` array.
    """
    _check_frames(frames_bgr)
    out = np.empty((len(frames_bgr), *size, 3), dtype=np.uint8)
    for start in range(0, len(frames_bgr), _CHUNK):
        chunk = torch.from_numpy(np.ascontiguousarray(frames_bgr[start : start + _CHUNK]))
        x = chunk.flip(3).permute(0, 3, 1, 2).float().div(255).mul(2).sub(1)
        y = TF.resize(x, list(size), interpolation=InterpolationMode.BICUBIC, antialias=True)
        y = y.add(1).div(2).mul(255).clamp(0, 255).byte()
        out[start : start + _CHUNK] = y.flip(1).permute(0, 2, 3, 1).numpy()
    return out


def is_fullres_selected(episode_id: str, mod: int = 10) -> bool:
    """Deterministic subset rule: ``int(sha1(episode_id)[:8], 16) % mod == 0``.

    Args:
        episode_id: Episode id.
        mod: Subset modulus (``pack.fullres_mod``).

    Returns:
        True if the episode also gets a full-res file.
    """
    return int(hashlib.sha1(episode_id.encode()).hexdigest()[:8], 16) % mod == 0


def _meta_attrs(meta: Mapping[str, Any], kinds: Sequence[str]) -> dict[str, Any]:
    missing = [k for k in _REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"meta is missing required keys: {missing}")
    episode_id = str(meta["episode_id"])
    # The id becomes a file name under out_dir, so it must not be able to escape it.
    if not _EPISODE_ID_RE.fullmatch(episode_id) or ".." in episode_id:
        raise ValueError(f"invalid episode_id {episode_id!r}: expected [A-Za-z0-9][A-Za-z0-9._-]* without '..'")
    if meta["kind"] not in kinds:
        raise ValueError(f"invalid kind {meta['kind']!r}: expected one of {list(kinds)}")
    attrs: dict[str, Any] = {}
    for key, value in meta.items():
        if key in _RESERVED_ATTRS:
            raise ValueError(f"meta key {key!r} collides with a reserved attr")
        if value is None:
            value = ""
        elif isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(  # noqa: TRY004
                f"meta[{key!r}] must be a scalar str/int/float/bool/None, got {type(value).__name__}"
            )
        attrs[key] = value
    return attrs


def _write_h5(
    tmp: Path,
    frames: Iterator[np.ndarray],
    actions: np.ndarray,
    attrs: Mapping[str, Any],
) -> None:
    with h5py.File(tmp, "w") as f:
        for i, frame in enumerate(frames):
            f.create_dataset(f"frame_{i}_x", data=frame)
            f.create_dataset(f"frame_{i}_y", data=actions[i])
        for key, value in attrs.items():
            f.attrs[key] = value


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_episode(
    frames_bgr: np.ndarray,
    actions: np.ndarray,
    meta: Mapping[str, Any],
    out_dir: Path,
    cfg: DictConfig,
    fullres_dir: Path | None = None,
) -> PackResult:
    """Write one episode to ``<out_dir>/<episode_id>.hdf5`` (and the full-res twin if selected).

    Files are written to ``<name>.tmp`` in the target directory and renamed, so a failed write leaves neither a
    partial file nor a clobbered earlier one. Each temp file is fsynced before any rename. The full-res file is
    renamed first and the low-res file last: the low-res file (plus the manifest line the caller appends afterwards)
    is the commit point downstream stages look for. A crash between the two renames can leave a fresh full-res file
    next to a stale low-res one; re-running ``write_episode`` for the episode repairs it.

    Args:
        frames_bgr: ``uint8[n, 150, 280, 3]`` rendered frames.
        actions: ``float32[n, 51]`` labels.
        meta: File attrs; must contain ``episode_id``, ``demo_id``, ``kind``, ``player``, ``start_tick``,
            ``cs2_build``, ``demoparser2_version`` and ``cs2wm_git_sha``; scalar values only. ``episode_id`` must
            match ``[A-Za-z0-9][A-Za-z0-9._-]*`` without ``..`` and ``kind`` must be in ``episodes.kinds``.
        out_dir: Low-res output directory.
        cfg: A validated cs2wm config.
        fullres_dir: Full-res output directory, or None to skip the subset.

    Returns:
        The written paths and checksums.

    Raises:
        ValueError: If shapes disagree, ``n`` is outside ``[1, episodes.max_frames]``, actions are not finite, or
            meta is invalid.
    """
    _check_frames(frames_bgr)
    n = len(frames_bgr)
    render = tuple(int(v) for v in cfg.resolution.render)
    low = tuple(int(v) for v in cfg.resolution.low_res)
    if tuple(frames_bgr.shape[1:3]) != render:
        raise ValueError(f"frames must be {render} (H, W), got {tuple(frames_bgr.shape[1:3])}")
    max_frames = int(cfg.episodes.max_frames)
    if not 1 <= n <= max_frames:
        raise ValueError(f"episode has {n} frames, expected 1..{max_frames}")
    actions = np.asarray(actions)
    if actions.shape != (n, ACTION_DIM):
        raise ValueError(f"actions must have shape ({n}, {ACTION_DIM}), got {actions.shape}")
    try:
        actions = actions.astype(np.float32, casting="same_kind" if actions.dtype.kind == "f" else "unsafe")
    except (TypeError, ValueError) as e:
        raise ValueError(f"actions are not castable to float32: {e}") from e
    if not np.isfinite(actions).all():
        raise ValueError("actions contain NaN or inf")
    base = _meta_attrs(meta, cfg.episodes.kinds)
    episode_id = str(base["episode_id"])

    common = {
        **base,
        "n_frames": n,
        "schema_version": SCHEMA_VERSION,
        "fps": int(cfg.timing.fps),
        "tickrate": int(cfg.timing.tickrate),
        "config_sha256": config_sha256(cfg),
    }
    targets: list[tuple[Path, Iterator[np.ndarray], dict[str, Any]]] = []
    low_path = Path(out_dir) / f"{episode_id}.hdf5"
    targets.append(
        (low_path, iter(to_low_res(frames_bgr, low)), {**common, "encoding": "raw", "height": low[0], "width": low[1]})
    )
    full_path: Path | None = None
    if fullres_dir is not None and is_fullres_selected(episode_id, int(cfg.pack.fullres_mod)):
        full_path = Path(fullres_dir) / f"{episode_id}.hdf5"
        quality = int(cfg.pack.jpeg_quality)
        targets.append(
            (
                full_path,
                (_encode_jpeg(frame, quality) for frame in frames_bgr),
                {
                    **common,
                    "encoding": "jpeg",
                    "jpeg_quality": quality,
                    "height": render[0],
                    "width": render[1],
                },
            )
        )

    tmps: list[Path] = []
    try:
        for path, frames, attrs in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmps.append(tmp)
            _write_h5(tmp, frames, actions, attrs)
            _fsync_path(tmp)
        # Low-res is the first target, so reversed() renames it last (the commit point).
        for (path, _, _), tmp in reversed(list(zip(targets, tmps))):
            os.replace(tmp, path)
    except BaseException:
        for tmp in tmps:
            tmp.unlink(missing_ok=True)
        raise

    # The two files share a name, so the full-res entry is keyed "fullres/<name>".
    sha = {low_path.name: sha256_file(low_path)}
    if full_path is not None:
        sha[f"fullres/{full_path.name}"] = sha256_file(full_path)
    return PackResult(low_path, full_path, n, sha)


def _encode_jpeg(frame_bgr: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("cv2.imencode failed")
    return buf.reshape(-1)


def _py(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_episode(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read an episode in either encoding (raw uint8 or JPEG).

    Args:
        path: HDF5 file.

    Returns:
        ``(frames, actions, attrs)``: BGR ``uint8[n, H, W, 3]``, ``float32[n, 51]`` and the file attrs as plain
        Python values.

    Raises:
        KeyError: If an attr or dataset is missing.
        ValueError: If the encoding is unknown or a JPEG fails to decode.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        attrs = {k: _py(v) for k, v in f.attrs.items()}
        for key in ("n_frames", "encoding"):
            if key not in attrs:
                raise KeyError(f"{path}: missing attr {key!r}")
        n, encoding = int(attrs["n_frames"]), attrs["encoding"]
        if encoding not in ("raw", "jpeg"):
            raise ValueError(f"{path}: unknown encoding {encoding!r}")
        frames = []
        for i in range(n):
            for name in (f"frame_{i}_x", f"frame_{i}_y"):
                if name not in f:
                    raise KeyError(f"{path}: missing dataset {name!r}")
            raw = f[f"frame_{i}_x"][:]
            if encoding == "jpeg":
                raw = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if raw is None:
                    raise ValueError(f"{path}: frame_{i}_x is not a valid JPEG")
            frames.append(raw)
        actions = np.stack([f[f"frame_{i}_y"][:] for i in range(n)]).astype(np.float32, copy=False)
    return np.stack(frames), actions, attrs


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file, read in 1 MiB chunks.

    Args:
        path: File to hash.

    Returns:
        Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def append_manifest(manifest_path: Path, record: Mapping[str, Any]) -> None:
    """Append one record (file, frames, sha256) to ``manifest.jsonl`` and fsync.

    Args:
        manifest_path: ``hdf5/<kind>/<demo_id>/manifest.jsonl``.
        record: JSON-serialisable record.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run(demo_id: str, cfg: DictConfig) -> list[PackResult]:
    """CLI stage: pack every QA-passing episode of a demo, then delete its rendered frames.

    Args:
        demo_id: Demo id.
        cfg: A validated cs2wm config.

    Returns:
        One result per packed episode.
    """
    # TODO(#12): read frames + labels, write_episode, append_manifest, delete frames after success.
    raise NotImplementedError("TODO(#12): pack.run")
