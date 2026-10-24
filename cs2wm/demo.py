"""Thin wrapper around demoparser2 (#11). The only module that imports it, so other code stays unit-testable.

Everything that touches the parser goes through ``_make_parser``; tests monkeypatch it with a fake that has the same
methods as ``demoparser2.DemoParser`` 0.42 (``parse_header``, ``parse_ticks``, ``list_game_events``,
``parse_events``, ``parse_grenades``). No real ``.dem`` file is needed to test the callers.

Names marked ``CONFIRM(#14)`` are demoparser2 column or field names that are assumed from the 0.42 API and the step 3
script of #14, not yet seen on a real CS2 demo. The #14 session must confirm or correct them.
"""

from __future__ import annotations

import importlib.metadata
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# CONFIRM(#14): per-tick table columns. ``user_id`` is a requested prop (step 3 of #14); ``name`` and ``steamid`` are
# always returned by parse_ticks.
TICK_USER_ID = "user_id"
# CONFIRM(#14): the entity id of the player pawn/controller; requested in step 3 of #14. It is what ``spec_player`` is
# expected to take (or the user_id; #14 step 4 decides).
TICK_ENTITY_ID = "entity_id"
# CONFIRM(#14): event-table columns. A player prop ``P`` requested for the ``user`` role of an event arrives as
# ``user_P`` (``user_name`` and ``user_steamid`` always). Candidates are tried in order; the first present wins.
EVENT_USER_ID_COLUMNS = ("user_user_id", "user_userid", "userid", "user_id")
EVENT_NAME_COLUMNS = ("user_name", "name")
EVENT_STEAMID_COLUMNS = ("user_steamid", "steamid", "networkid")
EVENT_ENTITY_ID_COLUMNS = ("user_entity_id",)
# CONFIRM(#14): parse_grenades() columns. ``grenade_entity_id`` identifies one trajectory; without it throws are
# split where a thrower's trajectory has a gap of more than ``_THROW_GAP_TICKS``.
GRENADE_ENTITY_COLUMN = "grenade_entity_id"
_THROW_GAP_TICKS = 64  # one second; a thrower cannot throw the same grenade type twice that fast

_BUILD_ID_RE = re.compile(r'"buildid"\s+"(\d+)"', re.IGNORECASE)
_APPMANIFEST = "appmanifest_730.acf"
_INSPECT_EVENTS = ("player_spawn", "player_death", "weapon_fire")


def _make_parser(demo_path: str | Path) -> Any:
    """Create a ``demoparser2.DemoParser``; tests replace this with a fake.

    Args:
        demo_path: Path to a ``.dem`` file.

    Returns:
        The parser object.
    """
    from demoparser2 import DemoParser

    return DemoParser(str(demo_path))


def parser_version() -> str:
    """Return the installed demoparser2 version (stored in HDF5 attrs)."""
    return importlib.metadata.version("demoparser2")


def header(demo_path: str | Path) -> dict[str, str]:
    """Parse the demo header (map, client name, CS2 patch/build).

    Args:
        demo_path: Path to a ``.dem`` file.

    Returns:
        Header fields as strings.
    """
    # CONFIRM(#14): which header keys carry the map and the CS2 version (map_name, demo_version_name, client_name?).
    return {str(k): str(v) for k, v in _make_parser(demo_path).parse_header().items()}


def load_ticks(
    demo_path: str | Path,
    props: Sequence[str],
    players: Sequence[int] | None = None,
    ticks: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Load per-tick player properties.

    Players are filtered here by ``user_id``, not passed to demoparser2: its ``players`` argument matches steamids,
    which are 0 or shared for bots (demoparser2 #358).

    Args:
        demo_path: Path to a ``.dem`` file.
        props: Property names, e.g. ``X``, ``yaw``, ``active_weapon_name``.
        players: Restrict to these ``user_id`` values (all players if None).
        ticks: Restrict to these ticks (all ticks if None).

    Returns:
        One row per (tick, player), with ``tick``, ``name``, ``steamid`` and ``user_id`` besides ``props``.
    """
    wanted = list(dict.fromkeys([*props, TICK_USER_ID]))
    kwargs: dict[str, Any] = {}
    if ticks is not None:
        kwargs["ticks"] = [int(t) for t in ticks]
    df = _make_parser(demo_path).parse_ticks(wanted, **kwargs)
    if players is not None:
        df = df[df[TICK_USER_ID].isin([int(p) for p in players])]
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
    parser = _make_parser(demo_path)
    # CONFIRM(#14): list_game_events() lists only the events present in the demo (asking for an absent one may raise).
    present = set(parser.list_game_events())
    out = {name: pd.DataFrame() for name in names}
    wanted = [n for n in dict.fromkeys(names) if n in present]
    if wanted:
        for name, df in parser.parse_events(wanted, player=list(player_props)):
            out[name] = df if df is not None else pd.DataFrame()
    return out


def player_event_table(events: pd.DataFrame) -> pd.DataFrame:
    """Normalise the ``user`` role of an event table to ``tick, user_id, name, steamid[, entity_id]``.

    Args:
        events: A table from ``load_events``.

    Returns:
        One row per event with canonical columns (empty with those columns when ``events`` is empty). Rows with a
        null ``user_id`` (e.g. warmup events with no player) are dropped and counted in a warning. Missing
        ``name``/``steamid`` become ``""``/0. ``entity_id`` is present only if the demo provides it.

    Raises:
        KeyError: If no ``user_id`` column is found (the event was loaded without the ``user_id`` player prop).
    """
    cols = ["tick", "user_id", "name", "steamid"]
    if events.empty:
        return pd.DataFrame({c: pd.Series(dtype="int64" if c != "name" else "object") for c in cols})

    def pick(candidates: Sequence[str]) -> str | None:
        return next((c for c in candidates if c in events.columns), None)

    uid = pick(EVENT_USER_ID_COLUMNS)
    if uid is None:
        raise KeyError(f"no user_id column in event table (columns: {list(events.columns)})")
    null_uid = events[uid].isna()
    if null_uid.any():
        log.warning("dropping %d event row(s) with a null %s", int(null_uid.sum()), uid)
        events = events[~null_uid]
    name, sid, eid = pick(EVENT_NAME_COLUMNS), pick(EVENT_STEAMID_COLUMNS), pick(EVENT_ENTITY_ID_COLUMNS)
    out = pd.DataFrame(
        {
            "tick": events["tick"].astype("int64"),
            "user_id": events[uid].astype("int64"),
            "name": events[name].fillna("").astype(str) if name else "",
            "steamid": pd.to_numeric(events[sid], errors="coerce").fillna(0).astype("int64") if sid else 0,
        }
    )
    if eid:
        out["entity_id"] = pd.to_numeric(events[eid], errors="coerce").astype("Int64")
    return out


def grenade_throws(demo_path: str | Path) -> pd.DataFrame:
    """Fallback for a missing ``grenade_thrown`` event, built from ``parse_grenades()``.

    Args:
        demo_path: Path to a ``.dem`` file.

    Returns:
        One row per throw (tick, thrower id, grenade type): the first row of each grenade trajectory, sorted by tick.
    """
    df = _make_parser(demo_path).parse_grenades()
    if df is None or df.empty:
        return pd.DataFrame(columns=["tick", "thrower_steamid", "grenade_type"])
    if GRENADE_ENTITY_COLUMN in df.columns:
        traj = df[GRENADE_ENTITY_COLUMN]
    else:
        # No trajectory id: a new throw starts at the first row of a thrower+type, or after a gap in its ticks.
        df = df.sort_values(["thrower_steamid", "grenade_type", "tick"], kind="stable")
        gap = df.groupby(["thrower_steamid", "grenade_type"], dropna=False)["tick"].diff()
        traj = (gap.isna() | (gap > _THROW_GAP_TICKS)).cumsum()
    return df.groupby(traj, sort=False).head(1).sort_values("tick", kind="stable").reset_index(drop=True)


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

    The manifest is looked up next to the game folder (``<library>/steamapps/common/<game>`` ->
    ``<library>/steamapps``), then under ``cs2_dir/steamapps`` and in ``cs2_dir`` itself.

    Args:
        cs2_dir: CS2 install directory (``paths.cs2_dir``).

    Returns:
        The Steam build id.

    Raises:
        FileNotFoundError: If the manifest is missing.
        ValueError: If the manifest has no ``buildid`` entry.
    """
    cs2_dir = Path(cs2_dir)
    candidates = [
        cs2_dir.parent.parent / _APPMANIFEST,
        cs2_dir / "steamapps" / _APPMANIFEST,
        cs2_dir / _APPMANIFEST,
    ]
    manifest = next((p for p in candidates if p.is_file()), None)
    if manifest is None:
        raise FileNotFoundError(f"{_APPMANIFEST} not found; looked in {[str(p.parent) for p in candidates]}")
    match = _BUILD_ID_RE.search(manifest.read_text(encoding="utf-8", errors="replace"))
    if match is None:
        raise ValueError(f"no buildid entry in {manifest}")
    return int(match.group(1))


def last_tick(demo_path: str | Path) -> int:
    """Return the last tick of the demo (the ``demo_end_tick`` for ``segment.find_lives``).

    Args:
        demo_path: Path to a ``.dem`` file.

    Returns:
        The largest tick present in the per-tick table.
    """
    return int(load_ticks(demo_path, [])["tick"].max())


def inspect(demo_path: str | Path) -> dict:
    """Summarise a demo for ``cs2wm inspect-demo``: header, players, tick range, event counts.

    Never raises on a bad demo: any parser failure (including pyo3's ``PanicException``, a ``BaseException``) on a
    truncated, empty or corrupt file yields ``ok=False`` and the message in ``error``; whatever was read before the
    failure is still returned. ``KeyboardInterrupt`` and ``SystemExit`` propagate.

    Args:
        demo_path: Path to a ``.dem`` file.

    Returns:
        JSON-serialisable summary with ``ok``, ``error``, ``demo``, ``parser_version``, ``header``, ``first_tick``,
        ``last_tick``, ``players`` and ``events``; also used to decide whether a truncated chunk is kept.
    """
    out: dict = {
        "demo": str(demo_path),
        "ok": False,
        "error": None,
        "parser_version": None,
        "header": None,
        "first_tick": None,
        "last_tick": None,
        "players": [],
        "events": {},
    }
    try:
        out["parser_version"] = parser_version()
        out["header"] = header(demo_path)
        ticks = load_ticks(demo_path, ["steamid"])
        if ticks.empty:
            raise ValueError("demo has no ticks")
        missing = [c for c in ("tick", "name", "steamid", TICK_USER_ID) if c not in ticks.columns]
        if missing:
            raise ValueError(f"tick table is missing columns {missing}")
        out["first_tick"], out["last_tick"] = int(ticks["tick"].min()), int(ticks["tick"].max())
        steam = pd.to_numeric(ticks["steamid"], errors="coerce").fillna(0)
        out["players"] = [
            {
                "user_id": int(uid),
                "name": str(g["name"].iloc[0]),
                "steamid": int(steam.loc[g.index[0]]),
                "n_ticks": len(g),
            }
            for uid, g in ticks.dropna(subset=[TICK_USER_ID]).groupby(TICK_USER_ID, sort=True)
        ]
        events = load_events(demo_path, _INSPECT_EVENTS)
        out["events"] = {name: len(events[name]) for name in _INSPECT_EVENTS}
        out["ok"] = True
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 - pyo3 PanicException derives from BaseException
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def is_parseable(demo_path: str | Path) -> bool:
    """Return whether the demo parses cleanly (header, ticks and the key events), for #13's keep-or-drop decision.

    Args:
        demo_path: Path to a ``.dem`` file.

    Returns:
        ``inspect(demo_path)["ok"]``.
    """
    return bool(inspect(demo_path)["ok"])
