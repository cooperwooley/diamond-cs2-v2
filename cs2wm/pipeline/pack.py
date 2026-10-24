"""pack: HDF5 episodes in v1's schema (30x56) plus a deterministic ~10% full-res JPEG subset (#12).

Each file holds ``frame_{i}_x`` (``uint8 (30, 56, 3)`` BGR) and ``frame_{i}_y`` (``float32 (51,)``) plus attrs
(``n_frames``, ``schema_version``, ``fps``, ``tickrate``, ``start_tick``, ``demo_id``, ``kind``, ``player``,
``cs2_build``, ``demoparser2_version``, ``cs2wm_git_sha``, ``config_sha256``); the provenance ones are required meta
keys. Writes are atomic (temp file, fsync, rename).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
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

from cs2wm import demo
from cs2wm.actions import ACTION_DIM
from cs2wm.config import config_sha256
from cs2wm.pipeline.extract_actions import labels_dir, read_qa, sweep_deleting
from cs2wm.pipeline.generate import _git_sha
from cs2wm.pipeline.segment import episodes_from_plan, read_render_plan

log = logging.getLogger(__name__)

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


class PackError(RuntimeError):
    """A packed episode does not hold what was meant to be written."""


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


def write_manifest(manifest_path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Replace ``manifest.jsonl`` with ``records`` atomically (temp file, fsync, rename).

    ``pack.run`` rewrites a demo's manifest from the episodes of that run, so a re-run never leaves duplicates.

    Args:
        manifest_path: ``hdf5/<kind>/<demo_id>/manifest.jsonl``.
        records: JSON-serialisable records, one line each.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(record, sort_keys=True) + "\n" for record in records)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, manifest_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _read_frames(seg_dir: Path, first: int, n: int, size: tuple[int, int]) -> np.ndarray:
    """Read frames ``first .. first + n - 1`` of a segment; frame ``first + n`` must exist but is not read."""
    extra = seg_dir / f"{first + n:06d}.png"
    if not extra.is_file():
        raise FileNotFoundError(f"{extra} is missing (an episode of n labels needs n + 1 rendered frames)")
    frames = np.empty((n, *size, 3), dtype=np.uint8)
    for i in range(n):
        path = seg_dir / f"{first + i:06d}.png"
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"{path} is missing or not a readable image")
        if img.shape != (*size, 3):
            raise ValueError(f"{path} has shape {img.shape}, expected {(*size, 3)}")
        frames[i] = img
    return frames


def _sidecar(cfg: DictConfig, kind: str, demo_id: str) -> dict[str, Any]:
    try:
        with open(Path(cfg.paths.demos) / kind / f"{demo_id}.json", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _check_roundtrip(result: PackResult, labels: np.ndarray) -> None:
    """Re-read what was written and require the stored actions to equal the labels (a write/read round trip)."""
    for path in (result.low_res_path, result.fullres_path):
        if path is None:
            continue
        _, actions, attrs = read_episode(path)
        if int(attrs["n_frames"]) != len(labels) or not np.array_equal(actions, labels):
            raise PackError(f"{path.name}: stored actions differ from the labels (write/read round trip)")


def _check_frame_alignment(result: PackResult, seg_dir: Path, first: int, n: int, size: tuple[int, int]) -> None:
    """Spot check that the stored first and last frames are source frames ``first`` and ``first + n - 1``.

    The source PNGs are read again from disk by index, independently of the slicing in ``run``. Low-res must equal
    ``to_low_res`` of the source; full-res (JPEG) must equal the source encoded and decoded with the same quality.
    """
    ends = (0, n - 1)
    src = np.stack([_read_frames(seg_dir, first + i, 1, size)[0] for i in ends])
    for path, expected in ((result.low_res_path, None), (result.fullres_path, "jpeg")):
        if path is None:
            continue
        frames, _, attrs = read_episode(path)
        if expected is None:
            want = to_low_res(src, (frames.shape[1], frames.shape[2]))
        else:
            quality = int(attrs["jpeg_quality"])
            want = np.stack([cv2.imdecode(_encode_jpeg(f, quality), cv2.IMREAD_COLOR) for f in src])
        if not np.array_equal(frames[list(ends)], want):
            raise PackError(
                f"{path.name}: stored frames 0 and {n - 1} are not source frames {first} and {first + n - 1}"
            )


def sweep_leftovers(cfg: DictConfig, demo_id: str) -> None:
    """Remove ``*.staging`` / ``*.old`` output dirs of an interrupted pack (``*.deleting`` is handled by extract)."""
    for tree in (cfg.paths.hdf5, cfg.paths.hdf5_fullres):
        for suffix in (".staging", ".old"):
            for path in Path(tree).glob(f"*/{demo_id}{suffix}"):
                shutil.rmtree(path, ignore_errors=True)


def _swap_in(staging: Path, final: Path) -> None:
    """Replace ``final`` by ``staging``: rename the old aside, rename staging into place, delete the old."""
    old = final.with_name(final.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        os.replace(final, old)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)
    shutil.rmtree(old, ignore_errors=True)


def _done_results(cfg: DictConfig, demo_id: str, kind: str, expected: set[str]) -> list[PackResult] | None:
    """Results rebuilt from ``manifest.jsonl`` if it lists every expected episode and every file matches its sha256."""
    out, full = Path(cfg.paths.hdf5) / kind / demo_id, Path(cfg.paths.hdf5_fullres) / kind / demo_id
    try:
        with open(out / "manifest.jsonl", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        by_file = {r["file"]: r for r in records}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if {Path(name).name.removesuffix(".hdf5") for name in by_file} != expected:
        return None
    results = []
    for ep_id in sorted(expected):
        low = by_file.get(f"{ep_id}.hdf5")
        if low is None:
            return None
        sha = {}
        paths: dict[str, Path] = {}
        for name, path in ((f"{ep_id}.hdf5", out / f"{ep_id}.hdf5"), (f"fullres/{ep_id}.hdf5", full / f"{ep_id}.hdf5")):
            rec = by_file.get(name)
            if rec is None:
                continue
            if not path.is_file() or sha256_file(path) != rec["sha256"]:
                return None
            sha[name], paths[name] = rec["sha256"], path
        results.append(PackResult(paths[f"{ep_id}.hdf5"], paths.get(f"fullres/{ep_id}.hdf5"), int(low["frames"]), sha))
    return results


def run(demo_id: str, cfg: DictConfig) -> list[PackResult]:
    """CLI stage: pack every QA-passing episode of a demo, then delete its rendered frames and labels.

    Reads ``qa/<demo_id>.jsonl`` (written by extract), the render plan and ``frames/<demo_id>/render.json``. Per
    passing episode the first ``n`` of its ``n + 1`` rendered frames go into the file with the ``n`` labels.

    Safety:

    * Episodes are written into ``<demo_id>.staging`` directories next to the final ones and checked there: the
      stored actions must equal the labels, and stored frames 0 and ``n - 1`` must be source frames ``first`` and
      ``first + n - 1``. Only when every episode succeeded are the staging dirs swapped in (the old dir is renamed
      aside, staging renamed into place, the old removed; the manifest is part of the swapped low-res dir, so that
      rename is the commit point). On any failure only the staging dirs are deleted and an earlier complete output
      stays as it was.
    * Idempotent: if the demo's ``manifest.jsonl`` already lists exactly the passing episodes and every file matches
      its sha256, the work is done. The call then returns those results even when ``frames/`` or ``render.json``
      is gone (a kill between the manifest write and the cleanup) and finishes the cleanup.
    * Cleanup renames ``frames/<id>`` to ``frames/<id>.deleting`` before deleting it, so a partial delete is
      recognisable; render and extract sweep such dirs.

    Provenance meta: ``cs2_build`` is the Steam ``buildid`` from ``appmanifest_730.acf``. The sidecar value (the build
    that recorded the demo) wins over the plan's (the build installed when planning); a warning is logged when they
    differ or when neither is known (an empty string is stored then). ``demoparser2_version`` comes from the plan,
    ``cs2wm_git_sha`` from the sidecar (else the current commit).

    Args:
        demo_id: Demo id.
        cfg: A validated cs2wm config.

    Returns:
        One result per packed episode (empty when no episode passed QA).

    Raises:
        FileNotFoundError: If the plan, ``render.json``, the QA file, a frame or a label file is missing.
        ValueError: If ``render.json`` was made from a different plan, or an episode is missing from the QA file or
            from ``render.json``.
        PackError: If a stored action or frame does not match what should have been written.
    """
    plan_path = Path(cfg.paths.plans) / demo_id / "render_plan.json"
    plan_sha = sha256_file(plan_path)
    plan = read_render_plan(plan_path)
    frames_root = Path(cfg.paths.frames) / demo_id
    qa = read_qa(cfg, demo_id)
    episodes = episodes_from_plan(plan)
    sweep_leftovers(cfg, demo_id)
    sweep_deleting(cfg, demo_id)

    passing = [ep for ep in episodes if qa.get(ep.episode_id, {}).get("passed")]
    kinds = sorted({ep.kind for ep in episodes})
    done: list[PackResult] | None = []
    for kind in kinds:
        expected = {ep.episode_id for ep in passing if ep.kind == kind}
        if not expected:
            continue
        part = _done_results(cfg, demo_id, kind, expected)
        if part is None:
            done = None
            break
        done += part
    if done is not None and not passing and (frames_root / "render.json").is_file():
        done = None  # nothing passes QA, but the inputs are still there: run, so stale output is cleared
    if done is not None:
        _cleanup(cfg, demo_id)
        log.info("%s: already packed (%d episode(s)); cleaned up", demo_id, len(done))
        return done

    with open(frames_root / "render.json", encoding="utf-8") as f:
        rendered = json.load(f)
    if rendered.get("plan_sha256") != plan_sha:
        raise ValueError(f"{demo_id}: render.json was made from a different plan than {plan_path}; re-run render")
    where = {e["episode_id"]: (seg["dir"], e["first_frame"]) for seg in rendered["segments"] for e in seg["episodes"]}
    size = (int(cfg.resolution.render[0]), int(cfg.resolution.render[1]))
    sidecar = _sidecar(cfg, episodes[0].kind, demo_id) if episodes else {}
    build, planned = sidecar.get("cs2_build"), plan.get("cs2_build")
    if build is not None and planned is not None and build != planned:
        log.warning(
            "%s: demo was recorded on CS2 build %s but planned on build %s; storing %s", demo_id, build, planned, build
        )
    build = build if build is not None else planned
    if build is None:
        log.warning("%s: cs2_build is unknown (no sidecar value and no plan value); storing an empty string", demo_id)
    provenance = {
        "cs2_build": build,
        "demoparser2_version": plan.get("parser_version") or demo.parser_version(),
        "cs2wm_git_sha": sidecar.get("cs2wm_git_sha") or _git_sha(),
    }

    staging = {  # kind -> (low-res staging dir, full-res staging dir)
        k: (
            Path(cfg.paths.hdf5) / k / f"{demo_id}.staging",
            Path(cfg.paths.hdf5_fullres) / k / f"{demo_id}.staging",
        )
        for k in kinds
    }
    results: list[PackResult] = []
    records: dict[str, list[dict[str, Any]]] = {}  # kind -> manifest records
    try:
        for ep in episodes:
            if ep.episode_id not in qa:
                raise ValueError(f"{demo_id}: {ep.episode_id} is not in the QA file; re-run extract")
            if not qa[ep.episode_id]["passed"]:
                log.info("%s: %s failed QA, not packed", demo_id, ep.episode_id)
                continue
            if ep.episode_id not in where:
                raise ValueError(f"{demo_id}: {ep.episode_id} is not in render.json")
            seg_dir, first = where[ep.episode_id]
            labels = np.load(labels_dir(cfg, demo_id) / f"{ep.episode_id}.npy")
            frames = _read_frames(frames_root / seg_dir, int(first), ep.n_frames, size)
            meta = {
                "episode_id": ep.episode_id,
                "demo_id": demo_id,
                "kind": ep.kind,
                "player": ep.player_name,
                "user_id": ep.user_id,  # demo user_id of the spectated player (bots share steamid 0); H2 needs it
                "start_tick": ep.start_tick,
                **provenance,
            }
            out, full = staging[ep.kind]
            result = write_episode(frames, labels, meta, out, cfg, fullres_dir=full)
            _check_roundtrip(result, labels)
            _check_frame_alignment(result, frames_root / seg_dir, int(first), ep.n_frames, size)
            results.append(result)
            for name, sha in result.sha256.items():
                records.setdefault(ep.kind, []).append({"file": name, "frames": result.n_frames, "sha256": sha})
        for kind in kinds:
            if kind in records:
                write_manifest(staging[kind][0] / "manifest.jsonl", records[kind])
    except BaseException:
        for out, full in staging.values():
            shutil.rmtree(out, ignore_errors=True)
            shutil.rmtree(full, ignore_errors=True)
        raise

    # Commit: full-res first, then the low-res dir with its manifest (the rename that sync and verify look for).
    final: list[PackResult] = []
    for kind in kinds:
        out, full = staging[kind]
        final_out, final_full = Path(cfg.paths.hdf5) / kind / demo_id, Path(cfg.paths.hdf5_fullres) / kind / demo_id
        if kind not in records:  # nothing passes QA now; do not leave an earlier run's output behind
            shutil.rmtree(final_out, ignore_errors=True)
            shutil.rmtree(final_full, ignore_errors=True)
            continue
        if full.exists():
            _swap_in(full, final_full)
        else:
            shutil.rmtree(final_full, ignore_errors=True)
        _swap_in(out, final_out)
    for r in results:
        # The files were written under <demo_id>.staging; they now live in <demo_id>.
        def moved(path: Path) -> Path:
            return path.parent.with_name(path.parent.name.removesuffix(".staging")) / path.name

        final.append(
            PackResult(moved(r.low_res_path), moved(r.fullres_path) if r.fullres_path else None, r.n_frames, r.sha256)
        )
    _cleanup(cfg, demo_id)
    log.info("%s: packed %d episode(s), %d frames", demo_id, len(final), sum(r.n_frames for r in final))
    return final


def _cleanup(cfg: DictConfig, demo_id: str) -> None:
    """Delete ``frames/<id>`` and ``labels/<id>``; the frames are renamed to ``.deleting`` first."""
    frames_root = Path(cfg.paths.frames) / demo_id
    if frames_root.exists():
        trash = frames_root.with_name(frames_root.name + ".deleting")
        shutil.rmtree(trash, ignore_errors=True)
        os.replace(frames_root, trash)
        shutil.rmtree(trash)
    shutil.rmtree(labels_dir(cfg, demo_id), ignore_errors=True)
