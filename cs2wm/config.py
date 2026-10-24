"""Load and validate ``config/cs2wm.yaml``, the single source of truth for paths, timing and action layout (#9)."""

from __future__ import annotations

import hashlib
import itertools
import re
import shlex
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "config" / "cs2wm.yaml"


# Sections that change labels or stored data. ``paths`` is machine-specific and ``supervisor`` holds the secret ntfy topic.
_HASHED_SECTIONS = ("timing", "resolution", "episodes", "actions", "qa", "segment", "pack")
# Only the render keys that change the pixels or the camera are hashed; poll_s, launch_options, paths, ... are not.
_HASHED_RENDER_KEYS = ("tick_offset", "output_format", "spec_mode", "spec_player_field")
# Names that end up in file names and console commands (demo ids, the mirv_streams stream name).
SAFE_NAME_RE = re.compile(r"^[\w.\-]+$")


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


_UNSAFE_OPTIONS = ("-secure", "connect")
_UNSAFE_PREFIXES = ("+connect", "-connect")


def split_command_line(text: str) -> list[str]:
    """Split a command line into tokens, keeping quoted arguments (and their quotes) as one token.

    ``shlex`` in non-POSIX mode leaves backslashes alone, so Windows paths survive and a quoted path that happens to
    contain ``-insecure`` stays a single token instead of producing a fake flag.

    Args:
        text: A command line such as ``-steam +playdemo "C:\\a b\\x.dem"``.

    Returns:
        The tokens.

    Raises:
        ValueError: On unbalanced quotes.
    """
    return shlex.split(text, posix=False)


def launch_option_problem(options: Sequence[str]) -> str | None:
    """Check CS2 launch options for the offline-only rules.

    Args:
        options: Launch tokens.

    Returns:
        A description of the first violation, or None. ``-insecure`` is required; ``-secure`` and anything that
        connects to a server (``+connect``, ``-connect``, ``connect``) is refused.
    """
    if "-insecure" not in options:
        return "must contain '-insecure' (CS2 must never run without it)"
    for opt in options:
        low = opt.lower()
        if low in _UNSAFE_OPTIONS or low.startswith(_UNSAFE_PREFIXES):
            return f"{opt!r} is not allowed (CS2 only ever runs offline)"
    return None


def _check_generate(gen: DictConfig, problems: list[str]) -> None:
    """Validate the numeric and enum keys of ``generate`` (None or a non-number is a problem, not a TypeError)."""
    if gen.chunking not in ("gotv", "netcon"):
        problems.append(f"generate.chunking must be 'gotv' or 'netcon', got {gen.chunking!r}")
    nums: dict[str, float] = {}
    for key in (
        "chunk_min",
        "hang_timeout_min",
        "poll_s",
        "close_stable_s",
        "close_timeout_s",
        "max_failures",
        "min_chunk_s",
        "map_load_wait_s",
        "netcon_connect_timeout_s",
        "netconport",
    ):
        value = gen.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"generate.{key} must be a number, got {value!r}")
        elif value <= 0:
            problems.append(f"generate.{key} must be > 0, got {value}")
        else:
            nums[key] = value
    if "netconport" in nums and not 1024 <= nums["netconport"] <= 65535:
        problems.append(f"generate.netconport must be in [1024, 65535], got {nums['netconport']}")
    if {"close_stable_s", "close_timeout_s"} <= nums.keys() and nums["close_stable_s"] >= nums["close_timeout_s"]:
        problems.append("generate.close_stable_s must be < generate.close_timeout_s")
    netcon_keys = {"netcon_connect_timeout_s", "map_load_wait_s", "hang_timeout_min"}
    if gen.chunking == "netcon" and netcon_keys <= nums.keys():
        startup = nums["netcon_connect_timeout_s"] + nums["map_load_wait_s"]
        if startup >= nums["hang_timeout_min"] * 60:
            problems.append(
                "generate.netcon_connect_timeout_s + map_load_wait_s must be < hang_timeout_min * 60 "
                f"(the watchdog grace period starts at launch), got {startup} s"
            )


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

    for section in ("generate", "render"):
        try:
            options = [t for o in cfg[section].launch_options for t in split_command_line(str(o))]
        except ValueError as exc:
            problems.append(f"{section}.launch_options: {exc}")
            continue
        problem = launch_option_problem(options)
        if problem:
            problems.append(f"{section}.launch_options: {problem}")
    _check_generate(cfg.generate, problems)

    for key, value in cfg.qa.items():
        if (key.endswith("_frac") or key == "min_fire_recall") and not 0 <= value <= 1:
            problems.append(f"qa.{key} must be in [0, 1], got {value}")

    if cfg.qa.max_fill_ticks < 0:
        problems.append(f"qa.max_fill_ticks must be >= 0, got {cfg.qa.max_fill_ticks}")

    _validate_render(cfg, problems)
    _validate_supervisor(cfg.supervisor, problems)

    if "default" not in cfg.actions.slots:
        problems.append("actions.slots.default is missing")

    if problems:
        raise ConfigError("invalid cs2wm config:\n  - " + "\n  - ".join(problems))


def _validate_supervisor(sup: DictConfig, problems: list[str]) -> None:
    """Append problems with the ``supervisor`` section."""
    for key in (
        "generate_block_min",
        "drain_min",
        "max_working_gb",
        "min_free_gb",
        "heartbeat_min",
        "smoke_s",
        "max_demo_wall_min",
        "stall_backoff_min",
    ):
        value = sup.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            problems.append(f"supervisor.{key} must be a number > 0, got {value!r}")
    value = sup.get("backoff_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        problems.append(f"supervisor.backoff_s must be a number >= 0, got {value!r}")
    for key in ("max_attempts", "alert_failed_in_row", "max_generate_failures"):
        value = sup.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            problems.append(f"supervisor.{key} must be an integer >= 1, got {value!r}")


def _validate_render(cfg: DictConfig, problems: list[str]) -> None:
    """Append problems with the ``render`` section."""
    r = cfg.render
    if not SAFE_NAME_RE.match(str(r.stream_name)):
        problems.append(f"render.stream_name may only hold letters, digits, '_', '.' and '-', got {r.stream_name!r}")
    if r.output_format not in ("tga", "ffmpeg"):
        problems.append(f"render.output_format must be 'tga' or 'ffmpeg', got {r.output_format!r}")
    if r.spec_player_field not in ("entity_id", "user_id", "slot", "name"):
        problems.append(
            f"render.spec_player_field must be entity_id, user_id, slot or name, got {r.spec_player_field!r}"
        )
    for key in ("timeout_factor", "poll_s", "quit_delay_ticks"):
        if r[key] <= 0:
            problems.append(f"render.{key} must be > 0, got {r[key]}")
    if r.startup_s < 0 or r.retries < 0:
        problems.append(f"render.startup_s and render.retries must be >= 0, got {r.startup_s} and {r.retries}")
    if not 0 <= r.png_compression <= 9:
        problems.append(f"render.png_compression must be in [0, 9], got {r.png_compression}")


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

    ``paths`` (machine-specific), ``supervisor`` (secret ntfy topic), ``generate`` and the machine-local ``render``
    keys (timeouts, launch options, HLAE paths, PNG level) are excluded on purpose.

    Args:
        cfg: A resolved config.

    Returns:
        Hex sha256 of the sorted-key YAML dump of ``_HASHED_SECTIONS``.
    """
    render = {key: cfg.render[key] for key in _HASHED_RENDER_KEYS}
    if cfg.render.output_format == "ffmpeg":
        render["ffmpeg_preset"] = cfg.render.ffmpeg_preset  # the codec changes the pixels; unused for tga
    subset = OmegaConf.create({**{name: cfg[name] for name in _HASHED_SECTIONS}, "render": render})
    return hashlib.sha256(OmegaConf.to_yaml(subset, sort_keys=True).encode()).hexdigest()
