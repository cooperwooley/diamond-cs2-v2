"""Thin wrapper around demoparser2 (#11). The only module that imports it, so other code stays unit-testable."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def parser_version() -> str:
    """Return the installed demoparser2 version (stored in HDF5 attrs)."""
    # TODO(#11): importlib.metadata.version("demoparser2").
    raise NotImplementedError("TODO(#11): parser_version")


def header(demo_path: str | Path) -> dict[str, str]:
    """Parse the demo header (map, client name, CS2 patch/build).

    Args:
        demo_path: Path to a ``.dem`` file.

    Returns:
        Header fields as strings.
    """
    # TODO(#11): DemoParser(...).parse_header().
    raise NotImplementedError("TODO(#11): header")


def load_ticks(
    demo_path: str | Path,
    props: Sequence[str],
    players: Sequence[int] | None = None,
    ticks: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Load per-tick player properties.

    Args:
        demo_path: Path to a ``.dem`` file.
        props: Property names, e.g. ``X``, ``yaw``, ``active_weapon_name``.
        players: Restrict to these ``user_id`` values (all players if None).
        ticks: Restrict to these ticks (all ticks if None).

    Returns:
        One row per (tick, player).
    """
    # TODO(#11): DemoParser.parse_ticks with the same filters.
    raise NotImplementedError("TODO(#11): load_ticks")


def load_events(
    demo_path: str | Path,
    names: Sequence[str],
    player_props: Sequence[str] = ("user_id",),
) -> dict[str, pd.DataFrame]:
    """Load game events by name.

    Args:
        demo_path: Path to a ``.dem`` file.
        names: Event names, e.g. ``weapon_fire``.
        player_props: Player properties attached to each event.

    Returns:
        Mapping from event name to its table (empty frame when the demo has none).
    """
    # TODO(#11): DemoParser.parse_events; match bots by user_id or entity, not steamid (demoparser2 #358).
    raise NotImplementedError("TODO(#11): load_events")


def grenade_throws(demo_path: str | Path) -> pd.DataFrame:
    """Fallback for a missing ``grenade_thrown`` event, built from ``parse_grenades()``.

    Args:
        demo_path: Path to a ``.dem`` file.

    Returns:
        One row per throw (tick, thrower id, grenade type).
    """
    # TODO(#11): derive throws from the first tick of each grenade trajectory.
    raise NotImplementedError("TODO(#11): grenade_throws")


def smoke_events(demo_path: str | Path) -> pd.DataFrame:
    """Export smoke detonate/expired ticks, position and thrower for the H2 masks (#33).

    Args:
        demo_path: Path to a ``.dem`` file.

    Returns:
        Table saved as ``events/<demo_id>.parquet``.
    """
    # TODO(#23): join smokegrenade_detonate and smokegrenade_expired by entity id.
    raise NotImplementedError("TODO(#23): smoke_events")


def cs2_build_id(cs2_dir: str | Path) -> int:
    """Read the installed CS2 ``buildid`` from ``steamapps/appmanifest_730.acf``.

    Args:
        cs2_dir: CS2 install directory (``paths.cs2_dir``).

    Returns:
        The Steam build id.

    Raises:
        FileNotFoundError: If the manifest is missing.
    """
    # TODO(#13): parse the "buildid" entry of the ACF file (used by the patch guard).
    raise NotImplementedError("TODO(#13): cs2_build_id")


def inspect(demo_path: str | Path) -> dict:
    """Summarise a demo for ``cs2wm inspect-demo``: header, players, tick range, event counts.

    Args:
        demo_path: Path to a ``.dem`` file.

    Returns:
        JSON-serialisable summary; also used to decide whether a truncated chunk is kept.
    """
    # TODO(#14): header + players seen in ticks + counts of spawn/death/fire events.
    raise NotImplementedError("TODO(#14): inspect")
