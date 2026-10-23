"""Load and validate ``config/cs2wm.yaml``, the single source of truth for paths, timing and action layout (#9)."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "config" / "cs2wm.yaml"


# Sections that change labels or stored data. ``paths`` is machine-specific and ``supervisor`` holds the secret ntfy topic.
_HASHED_SECTIONS = ("timing", "resolution", "episodes", "actions", "qa", "segment", "pack", "render")


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
    cfg = OmegaConf.load(path or DEFAULT_CONFIG_PATH)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    validate_config(cfg)
    OmegaConf.set_readonly(cfg, True)
    return cfg


def _check_units(name: str, units: list[float], problems: list[str]) -> None:
    """Append problems for one mouse-bin axis (sorted, containing 0, symmetric)."""
    if any(b <= a for a, b in itertools.pairwise(units)):
        problems.append(f"actions.mouse.{name} must be strictly increasing, got {units}")
    if 0 not in units:
        problems.append(f"actions.mouse.{name} must contain 0, got {units}")
    if units != [-u for u in reversed(units)]:
        problems.append(f"actions.mouse.{name} must be symmetric around 0 (u == -u[::-1]), got {units}")


def validate_config(cfg: DictConfig) -> None:
    """Check cross-field consistency.

    Rules: ``tickrate % fps == 0``; resolutions are 2 positive ints; ``min_frames <= max_frames``; 11 keys and
    2 clicks; mouse bin units strictly increasing, symmetric around 0 and containing 0; layout dim == 51;
    ``deg_per_unit > 0``; ``-insecure`` in the launch options; QA fractions in [0, 1]; a default weapon slot.

    Args:
        cfg: The config to check.

    Raises:
        ConfigError: Listing every violated rule.
    """
    problems: list[str] = []

    tickrate, fps = cfg.timing.tickrate, cfg.timing.fps
    if tickrate <= 0 or fps <= 0:
        problems.append(f"timing.tickrate and timing.fps must be > 0, got {tickrate} and {fps}")
    elif tickrate % fps != 0:
        problems.append(f"timing.tickrate ({tickrate}) must be a multiple of timing.fps ({fps})")

    for name in ("capture", "render", "low_res"):
        res = list(cfg.resolution[name])
        if len(res) != 2 or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in res):
            problems.append(f"resolution.{name} must be [H, W] with 2 positive ints, got {res}")

    min_frames, max_frames = cfg.episodes.min_frames, cfg.episodes.max_frames
    if min_frames <= 0 or max_frames <= 0:
        problems.append(f"episodes.min_frames and max_frames must be > 0, got {min_frames} and {max_frames}")
    if min_frames > max_frames:
        problems.append(f"episodes.min_frames ({min_frames}) must be <= episodes.max_frames ({max_frames})")

    # cfg.actions.keys is the DictConfig.keys method, so index it.
    n_keys, n_clicks = len(cfg.actions["keys"]), len(cfg.actions.clicks)
    if n_keys != 11:
        problems.append(f"actions.keys must have 11 entries, got {n_keys}")
    if n_clicks != 2:
        problems.append(f"actions.clicks must have 2 entries, got {n_clicks}")

    x_units, y_units = list(cfg.actions.mouse.x_units), list(cfg.actions.mouse.y_units)
    _check_units("x_units", x_units, problems)
    _check_units("y_units", y_units, problems)
    total = n_keys + n_clicks + len(x_units) + len(y_units)
    if total != 51:
        problems.append(f"action layout must total 51 dims, got {total}")
    if cfg.actions.mouse.deg_per_unit <= 0:
        problems.append(f"actions.mouse.deg_per_unit must be > 0, got {cfg.actions.mouse.deg_per_unit}")

    if "-insecure" not in list(cfg.generate.launch_options):
        problems.append("generate.launch_options must contain '-insecure' (CS2 must never run without it)")

    if cfg.generate.chunking not in ("gotv", "netcon"):
        problems.append(f"generate.chunking must be 'gotv' or 'netcon', got {cfg.generate.chunking!r}")
    for key in ("chunk_min", "hang_timeout_min", "poll_s", "close_stable_s", "close_timeout_s", "max_failures"):
        if cfg.generate[key] <= 0:
            problems.append(f"generate.{key} must be > 0, got {cfg.generate[key]}")

    for key, value in cfg.qa.items():
        if (key.endswith("_frac") or key == "min_fire_recall") and not 0 <= value <= 1:
            problems.append(f"qa.{key} must be in [0, 1], got {value}")

    if cfg.qa.max_fill_ticks < 0:
        problems.append(f"qa.max_fill_ticks must be >= 0, got {cfg.qa.max_fill_ticks}")

    if "default" not in cfg.actions.slots:
        problems.append("actions.slots.default is missing")

    if problems:
        raise ConfigError("invalid cs2wm config:\n  - " + "\n  - ".join(problems))


def ticks_per_frame(cfg: DictConfig) -> int:
    """Return ``timing.tickrate // timing.fps`` (4 for the default config).

    Args:
        cfg: A validated config.

    Returns:
        Demo ticks covered by one rendered frame.
    """
    return int(cfg.timing.tickrate // cfg.timing.fps)


def config_sha256(cfg: DictConfig) -> str:
    """Hash the label-affecting config sections, for HDF5 attrs and demo sidecars.

    ``paths`` (machine-specific) and ``supervisor`` (secret ntfy topic) are excluded on purpose.

    Args:
        cfg: A resolved config.

    Returns:
        Hex sha256 of the sorted-key YAML dump of ``_HASHED_SECTIONS``.
    """
    subset = OmegaConf.create({name: cfg[name] for name in _HASHED_SECTIONS})
    return hashlib.sha256(OmegaConf.to_yaml(subset, sort_keys=True).encode()).hexdigest()
