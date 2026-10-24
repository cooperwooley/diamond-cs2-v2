"""Thin wrapper around demoparser2 (#11). The only module that imports it, so other code stays unit-testable."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

# CONFIRM(#14): demoparser2 field names below are from memory of its docs; check them against a real parse.
SMOKE_DETONATE_EVENT = "smokegrenade_detonate"
SMOKE_EXPIRED_EVENT = "smokegrenade_expired"
SMOKE_ENTITY_COLUMN = "entityid"  # joins a detonate to its expired event
SMOKE_POSITION_COLUMNS = ("x", "y", "z")
SMOKE_THROWER_COLUMNS = ("user_user_id", "userid", "user_userid")  # first one present; the event's player is "user"
SMOKE_PLAYER_PROPS = ("user_id",)  # attached to the events as user_<prop>
SMOKE_DURATION_TICKS = 1152  # CONFIRM(#14): 18 s at 64 Hz; used when a smoke has no expired event (demo cut off)
SMOKE_COLUMNS = ("entity_id", "detonate_tick", "expire_tick", "x", "y", "z", "thrower_user_id")


def _parser(demo_path: str | Path):
    """Return a ``demoparser2.DemoParser`` for a demo. The only place the library is imported (tests replace it)."""
    from demoparser2 import DemoParser

    return DemoParser(str(demo_path))


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
    parser = _parser(demo_path)
    wanted = list(props)
    if players is not None and "user_id" not in wanted:
        wanted.append("user_id")
    # demoparser2's own `players` filter takes steamids, which are 0 for bots, so filter on user_id afterwards.
    df = parser.parse_ticks(wanted, ticks=None if ticks is None else [int(t) for t in ticks])
    if players is not None:
        df = df[df["user_id"].isin([int(p) for p in players])]
    return df.reset_index(drop=True)


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
        Table saved as ``events/<demo_id>.parquet``: one row per smoke with ``entity_id``, ``detonate_tick``,
        ``expire_tick`` (``detonate_tick + SMOKE_DURATION_TICKS`` if the demo has no expired event for it), ``x``,
        ``y``, ``z`` (Source units) and ``thrower_user_id`` (-1 if unknown), sorted by detonation.
    """
    parser = _parser(demo_path)
    tables = dict(
        parser.parse_events([SMOKE_DETONATE_EVENT, SMOKE_EXPIRED_EVENT], player=list(SMOKE_PLAYER_PROPS)) or []
    )
    det = tables.get(SMOKE_DETONATE_EVENT)
    if det is None or len(det) == 0:
        floats = SMOKE_POSITION_COLUMNS
        return pd.DataFrame({c: pd.Series(dtype=np.float64 if c in floats else np.int64) for c in SMOKE_COLUMNS})
    thrower = next((c for c in SMOKE_THROWER_COLUMNS if c in det.columns), None)
    start = pd.DataFrame(
        {
            "entity_id": det[SMOKE_ENTITY_COLUMN].astype(np.int64),
            "detonate_tick": det["tick"].astype(np.int64),
            **{c: det[c].astype(np.float64) for c in SMOKE_POSITION_COLUMNS},
            "thrower_user_id": det[thrower].astype(np.int64) if thrower else -1,
        }
    ).sort_values("detonate_tick")
    exp = tables.get(SMOKE_EXPIRED_EVENT)
    if exp is None or len(exp) == 0:
        end = pd.DataFrame({"entity_id": np.array([], dtype=np.int64), "expire_tick": np.array([], dtype=np.int64)})
    else:
        end = pd.DataFrame(
            {"entity_id": exp[SMOKE_ENTITY_COLUMN].astype(np.int64), "expire_tick": exp["tick"].astype(np.int64)}
        ).sort_values("expire_tick")
    # Entity ids are recycled within a demo, so pair each detonation with the first later expiry of the same entity.
    joined = pd.merge_asof(
        start, end, left_on="detonate_tick", right_on="expire_tick", by="entity_id", direction="forward"
    )
    joined["expire_tick"] = (
        joined["expire_tick"].fillna(joined["detonate_tick"] + SMOKE_DURATION_TICKS).astype(np.int64)
    )
    return joined[list(SMOKE_COLUMNS)].sort_values("detonate_tick").reset_index(drop=True)


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
