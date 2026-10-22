"""Load and validate ``config/cs2wm.yaml``, the single source of truth for paths, timing and action layout (#9)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig

DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "config" / "cs2wm.yaml"


class ConfigError(ValueError):
    """Raised when ``cs2wm.yaml`` holds inconsistent values."""


def load_config(path: str | Path | None = None, overrides: Sequence[str] = ()) -> DictConfig:
    """Load, resolve, validate and freeze the cs2wm config.

    Args:
        path: YAML file to load; defaults to ``DEFAULT_CONFIG_PATH``.
        overrides: OmegaConf dotlist overrides such as ``"timing.fps=8"``.

    Returns:
        A resolved, read-only config.

    Raises:
        ConfigError: If ``validate_config`` rejects the values.
    """
    # TODO(#9): OmegaConf.load, merge dotlist overrides, resolve, validate_config, set read-only.
    raise NotImplementedError("TODO(#9): load_config")


def validate_config(cfg: DictConfig) -> None:
    """Check cross-field consistency.

    Rules: ``tickrate % fps == 0``; mouse bin units sorted, symmetric around 0 and containing 0; layout dim == 51.

    Args:
        cfg: The config to check.

    Raises:
        ConfigError: On the first violated rule.
    """
    # TODO(#9): implement the rules listed in the docstring.
    raise NotImplementedError("TODO(#9): validate_config")


def ticks_per_frame(cfg: DictConfig) -> int:
    """Return ``timing.tickrate // timing.fps`` (4 for the default config).

    Args:
        cfg: A validated config.

    Returns:
        Demo ticks covered by one rendered frame.
    """
    # TODO(#9): tickrate // fps.
    raise NotImplementedError("TODO(#9): ticks_per_frame")


def config_sha256(cfg: DictConfig) -> str:
    """Hash the resolved config, for HDF5 attrs and demo sidecars.

    Args:
        cfg: A resolved config.

    Returns:
        Hex sha256 of the canonical (sorted-key) JSON dump.
    """
    # TODO(#9): sha256 of json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True).
    raise NotImplementedError("TODO(#9): config_sha256")
