"""sync: push packed HDF5, demos and QA to the Mill, verify checksums, free local disk (#16).

Pushes ``hdf5/``, ``hdf5_fullres/``, ``demos/`` and ``qa/`` with ``rsync -a --partial`` (via ``wsl.exe -e rsync`` on
Windows, with ``wslpath`` translation; ``rsync`` directly on Linux) using ``ssh -o BatchMode=yes``. Idempotent and
resumable. Only HDF5 whose remote sha256 matches the local ``manifest.jsonl`` is ever deleted; demos stay local.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig


def push(cfg: DictConfig, demo_id: str | None = None) -> None:
    """Rsync the local tree to ``<sync.host>:<sync.remote_root>``.

    Args:
        cfg: A validated cs2wm config.
        demo_id: Restrict to one demo (all pending demos if None).

    Raises:
        RuntimeError: If key auth is not set up or rsync fails.
    """
    # TODO(#16): rsync -a --partial per directory, BatchMode=yes, fail fast with a clear message.
    raise NotImplementedError("TODO(#16): push")


def verify_remote(cfg: DictConfig, demo_id: str) -> list[Path]:
    """Check remote sha256s against the local manifest, one ``ssh ... sha256sum -c`` per demo.

    Args:
        cfg: A validated cs2wm config.
        demo_id: Demo to verify.

    Returns:
        Local files whose remote copy matches.
    """
    raise NotImplementedError("TODO(#16): verify_remote")


def delete_verified(files: list[Path]) -> int:
    """Delete local HDF5 files that ``verify_remote`` confirmed. Never deletes anything else.

    Args:
        files: Verified local files.

    Returns:
        Number of files deleted.
    """
    raise NotImplementedError("TODO(#16): delete_verified")


def run(cfg: DictConfig) -> None:
    """CLI stage: push, verify, then delete verified HDF5 (``sync.delete_local_hdf5``).

    Args:
        cfg: A validated cs2wm config.
    """
    # TODO(#16): push -> verify_remote -> delete_verified, per demo.
    raise NotImplementedError("TODO(#16): sync.run")
