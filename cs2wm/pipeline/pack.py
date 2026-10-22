"""pack: HDF5 episodes in v1's schema (30x56) plus a deterministic ~10% full-res JPEG subset (#12).

Each file holds ``frame_{i}_x`` (``uint8 (30, 56, 3)`` BGR) and ``frame_{i}_y`` (``float32 (51,)``) plus attrs
(``n_frames``, ``schema_version``, ``fps``, ``tickrate``, ``start_tick``, ``demo_id``, ``kind``, ``player``,
``cs2_build``, ``demoparser2_version``, ``cs2wm_git_sha``, ``config_sha256``). Writes are atomic (temp file + rename).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PackResult:
    """Files written for one episode.

    Attributes:
        low_res_path: Low-res HDF5 file.
        fullres_path: Full-res JPEG HDF5 file, or None if the episode is not in the subset.
        n_frames: Frames written.
        sha256: Hex sha256 by file name.
    """

    low_res_path: Path
    fullres_path: Path | None
    n_frames: int
    sha256: dict[str, str]


def to_low_res(frames_bgr: np.ndarray, size: tuple[int, int] = (30, 56)) -> np.ndarray:
    """Downscale exactly like v1 ``build_dataset.py::create_low_res_dataset``.

    Steps: BGR to RGB, scale to [-1, 1], torchvision bicubic resize to ``size``, ``add(1).div(2).mul(255).byte()``,
    back to BGR. A test asserts bit-equality with the v1 reference.

    Args:
        frames_bgr: ``uint8[n, H, W, 3]`` frames.
        size: Target ``(H, W)``.

    Returns:
        ``uint8[n, *size, 3]`` BGR frames.
    """
    # TODO(#12): torch CPU resize identical to v1 (the reused v1 upsampler depends on these statistics), except clamp
    # to [0, 255] before the uint8 cast: v1's .byte() wraps bicubic overshoot (see the #12 decision comment).
    raise NotImplementedError("TODO(#12): to_low_res")


def is_fullres_selected(episode_id: str, mod: int = 10) -> bool:
    """Deterministic subset rule: ``int(sha1(episode_id)[:8], 16) % mod == 0``.

    Args:
        episode_id: Episode id.
        mod: Subset modulus (``pack.fullres_mod``).

    Returns:
        True if the episode also gets a full-res file.
    """
    raise NotImplementedError("TODO(#12): is_fullres_selected")


def write_episode(
    frames_bgr: np.ndarray,
    actions: np.ndarray,
    meta: Mapping[str, Any],
    out_dir: Path,
    cfg: DictConfig,
    fullres_dir: Path | None = None,
) -> PackResult:
    """Write one episode to ``hdf5/<kind>/<demo_id>/<episode_id>.hdf5`` (and the full-res twin if selected).

    Args:
        frames_bgr: ``uint8[n, 150, 280, 3]`` rendered frames.
        actions: ``float32[n, 51]`` labels.
        meta: File attrs (see module docstring).
        out_dir: Low-res output directory.
        cfg: A validated cs2wm config.
        fullres_dir: Full-res output directory, or None to skip the subset.

    Returns:
        The written paths and checksums.

    Raises:
        ValueError: If shapes disagree or ``n`` exceeds ``episodes.max_frames``.
    """
    # TODO(#12): atomic write; JPEG quality pack.jpeg_quality with attr encoding="jpeg" for the full-res file.
    raise NotImplementedError("TODO(#12): write_episode")


def read_episode(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read an episode in either encoding (raw uint8 or JPEG).

    Args:
        path: HDF5 file.

    Returns:
        ``(frames, actions, attrs)``.
    """
    # TODO(#12): decode by the encoding attr; the v2 CS2Hdf5Dataset (#5) reads the same files.
    raise NotImplementedError("TODO(#12): read_episode")


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file, read in chunks.

    Args:
        path: File to hash.

    Returns:
        Hex digest.
    """
    raise NotImplementedError("TODO(#12): sha256_file")


def append_manifest(manifest_path: Path, record: Mapping[str, Any]) -> None:
    """Append one record (file, frames, sha256) to ``manifest.jsonl``.

    Args:
        manifest_path: ``hdf5/<kind>/<demo_id>/manifest.jsonl``.
        record: JSON-serialisable record.
    """
    raise NotImplementedError("TODO(#12): append_manifest")


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
