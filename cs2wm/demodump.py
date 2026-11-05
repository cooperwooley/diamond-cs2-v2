"""Shim that serves the ``demoparser2.DemoParser`` methods ``cs2wm.demo`` uses from the Go exporter ``demodump`` (#72).

``tools/demodump`` parses a demo with demoinfocs-golang and writes tables (see its ``main.go``). This module runs it
once per demo, caches the tables, and answers ``parse_header``, ``parse_ticks``, ``list_game_events``,
``parse_events`` and ``parse_grenades`` with the demoparser2 column names the callers already use, so ``segment``,
``extract_actions``, ``h2_smoke`` and ``inspect`` are unchanged.

Conventions (checked against demoparser2 0.42.1 on a casual client demo, see docs/DATASET.md section 4.5):

- ticks are demo ticks (``IngameTick``, the tick ``mirv_cmd addAtTick`` uses), one row per tick and player;
- ``yaw`` is -180..180 degrees and ``pitch`` -90..90 with positive looking down, as in demoparser2;
- ``active_weapon_name`` is the display name (``AK-47``, ``M4A1-S``), which ``actions.slot_for_weapon`` matches;
- ``entity_id`` is the player *controller* entity index (``pawn_entity_id`` is the pawn);
- GOTV demos carry no button state, so the ``FORWARD``/``FIRE``/... columns are never returned.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# Output format of the exporter (``version`` in tools/demodump/main.go) and the pinned demoinfocs-golang release
# (tools/demodump/go.mod). tests/test_demodump.py checks both against the Go sources. The format version is part of
# the cache key, so a bump re-parses every cached demo.
DEMODUMP_FORMAT = "1"
DEMOINFOCS_VERSION = "v6.0.0-rc.1"

ENV_BINARY = "DCS2_DEMODUMP"
REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = "scripts/build_demodump.sh"
_TIMEOUT_S = 1800  # a 30 minute demo takes seconds; this only stops a hung process
_VERSION_RE = re.compile(r"^demodump (\S+) demoinfocs (\S+)$")

# Tick columns that exist in every dump; ``parse_ticks`` returns these plus the wanted ones the dump has.
_TICK_ALWAYS = ("tick", "name", "steamid", "user_id")
_PLAYER_ROLE_COLUMNS = {"user_id": "user_user_id", "name": "user_name", "steamid": "user_steamid"}


class DemodumpNotFound(FileNotFoundError):
    """No ``demodump`` binary could be found."""


def binary_name() -> str:
    """File name of the exporter on this OS."""
    return "demodump.exe" if os.name == "nt" else "demodump"


def find_binary(configured: str | os.PathLike | None = None) -> Path:
    """Locate the exporter: env ``DCS2_DEMODUMP``, then ``demo.demodump`` from the config, then ``bin/`` in the repo, then PATH.

    Args:
        configured: The ``demo.demodump`` config value, if any.

    Returns:
        Path of an existing file.

    Raises:
        DemodumpNotFound: With the places searched and the build script, if there is none.
    """
    tried: list[str] = []
    for source, value in ((ENV_BINARY, os.environ.get(ENV_BINARY)), ("demo.demodump", configured)):
        if value:
            path = Path(value)
            if path.is_file():
                return path
            tried.append(f"{source}={path}")
    repo_bin = REPO_ROOT / "bin" / binary_name()
    if repo_bin.is_file():
        return repo_bin
    tried.append(str(repo_bin))
    on_path = shutil.which("demodump")
    if on_path:
        return Path(on_path)
    tried.append("PATH")
    raise DemodumpNotFound(
        f"demodump not found (looked at {', '.join(tried)}). Build it with {BUILD_SCRIPT} "
        f"(needs Go), or point {ENV_BINARY} at the binary."
    )


def cache_key(demo_path: Path) -> str:
    """Cache directory name: the demo's stem plus a hash of its absolute path, size, mtime and the dump format."""
    st = demo_path.stat()
    raw = f"{demo_path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{DEMODUMP_FORMAT}"
    return f"{demo_path.stem}-{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def dump_demo(demo_path: str | Path, cache_dir: str | Path, configured: str | os.PathLike | None = None) -> Path:
    """Run ``demodump`` on a demo unless a valid dump is cached, and return the dump directory.

    The dump is written to a temp directory beside the cache entry and renamed into place, so an interrupted run
    leaves no half-written entry.

    Args:
        demo_path: ``.dem`` file.
        cache_dir: Directory holding the dumps.
        configured: The ``demo.demodump`` config value, passed to ``find_binary`` (only needed on a cache miss).

    Returns:
        ``<cache_dir>/<stem>-<hash>``.

    Raises:
        DemodumpNotFound: If there is no exporter.
        RuntimeError: If the exporter fails or has a different format version than ``DEMODUMP_FORMAT``.
    """
    demo_path = Path(demo_path)
    target = Path(cache_dir) / cache_key(demo_path)
    if (target / "header.json").is_file():
        return target
    exe = find_binary(configured)
    check_version(exe)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        proc = subprocess.run(
            [str(exe), str(demo_path), str(tmp)], capture_output=True, text=True, timeout=_TIMEOUT_S, check=False
        )
        if proc.returncode != 0:
            raise RuntimeError(f"demodump failed on {demo_path} (exit {proc.returncode}): {proc.stderr.strip()[-500:]}")
        try:
            os.replace(tmp, target)
        except OSError:
            if not (target / "header.json").is_file():  # a concurrent run that renamed first is fine
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return target


def check_version(exe: Path) -> tuple[str, str]:
    """Return ``(format, demoinfocs version)`` of an exporter binary.

    Raises:
        RuntimeError: If it cannot be run or its format is not ``DEMODUMP_FORMAT``.
    """
    try:
        out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60, check=False).stdout
    except OSError as exc:
        raise RuntimeError(f"cannot run {exe}: {exc}") from exc
    match = _VERSION_RE.match(out.strip())
    if match is None:
        raise RuntimeError(f"{exe} --version printed {out.strip()!r}, expected 'demodump <n> demoinfocs <version>'")
    if match.group(1) != DEMODUMP_FORMAT:
        raise RuntimeError(
            f"{exe} writes dump format {match.group(1)}, cs2wm needs {DEMODUMP_FORMAT}; rebuild with {BUILD_SCRIPT}"
        )
    return match.group(1), match.group(2)


class DemodumpParser:
    """Drop-in for the parts of ``demoparser2.DemoParser`` that ``cs2wm.demo`` uses, backed by a cached dump."""

    def __init__(
        self,
        demo_path: str | Path,
        cache_dir: str | Path,
        configured: str | os.PathLike | None = None,
        *,
        dump: Path | None = None,
    ):
        """Dump the demo (or reuse the cache) and remember where.

        Args:
            demo_path: ``.dem`` file.
            cache_dir: Directory holding the dumps.
            configured: The ``demo.demodump`` config value.
            dump: An existing dump directory to read instead of running the exporter (tests).
        """
        self.dir = dump if dump is not None else dump_demo(demo_path, cache_dir, configured)

    def parse_header(self) -> dict[str, Any]:
        """The demo header: ``map_name``, ``patch_version``, ``tick_rate``, ``first_tick``, ``last_tick``, ..."""
        return json.loads((self.dir / "header.json").read_text(encoding="utf-8"))

    def parse_ticks(
        self,
        wanted_props: Sequence[str],
        *,
        players: Sequence[int] | None = None,
        ticks: Sequence[int] | None = None,
        prop_states: object = None,
    ) -> pd.DataFrame:
        """Per-tick player table.

        Args:
            wanted_props: Columns to return besides ``tick``, ``name``, ``steamid`` and ``user_id``. Names the dump
                does not have (button columns on a GOTV demo) are left out, as the callers expect.
            players: Restrict to these ``user_id`` values.
            ticks: Restrict to these ticks.
            prop_states: Ignored (demoparser2 signature compatibility).

        Returns:
            One row per (tick, player).
        """
        path = self.dir / "ticks.csv.gz"
        available = pd.read_csv(path, nrows=0).columns
        keep = [c for c in dict.fromkeys([*_TICK_ALWAYS, *wanted_props]) if c in available]
        df = pd.read_csv(path, usecols=keep)[keep]
        if ticks is not None:
            df = df[df["tick"].isin([int(t) for t in ticks])]
        if players is not None:
            df = df[df["user_id"].isin([int(p) for p in players])]
        return df.reset_index(drop=True)

    def list_game_events(self) -> list[str]:
        """Names of the events in the dump, raw and derived."""
        return sorted(p.name.removesuffix(".csv.gz") for p in (self.dir / "events").glob("*.csv.gz"))

    def parse_events(
        self, names: Sequence[str], *, player: Sequence[str] | None = None, other: object = None
    ) -> list[tuple[str, pd.DataFrame]]:
        """Event tables, with the ``user`` role as ``user_user_id``, ``user_name``, ``user_steamid``.

        Args:
            names: Event names (those the dump lacks are skipped, like demoparser2 ``list_game_events`` filtering).
            player: Player properties to attach; ``user_id``, ``name`` and ``steamid`` are always attached, and
                ``entity_id`` when asked for (as ``user_entity_id``).
            other: Ignored (demoparser2 signature compatibility).

        Returns:
            ``(name, table)`` pairs sorted by tick within each table. Raw columns keep their dump names
            (``userid`` is ``user_id``); ``derived`` is true for events reconstructed from tick data.
        """
        roster = pd.read_csv(self.dir / "players.csv").set_index("user_id")
        out: list[tuple[str, pd.DataFrame]] = []
        for name in names:
            path = self.dir / "events" / f"{name}.csv.gz"
            if not path.is_file():
                continue
            df = pd.read_csv(path).sort_values("tick", kind="stable").reset_index(drop=True)
            uid = df["user_id"] if "user_id" in df.columns else pd.Series(float("nan"), index=df.index)
            for src, dst in _PLAYER_ROLE_COLUMNS.items():  # an id missing from the roster (or null) gives NaN
                df[dst] = uid if src == "user_id" else roster[src].reindex(uid).to_numpy()
            if player is not None and "entity_id" in player:
                df["user_entity_id"] = roster["entity_id"].reindex(uid).to_numpy()
            out.append((name, df))
        return out

    def parse_grenades(self, *, extra: object = None, grenades: bool = True) -> pd.DataFrame:
        """Grenade trajectories: one row per (``grenade_entity_id``, ``tick``) with the thrower and ``X``/``Y``/``Z``."""
        return pd.read_csv(self.dir / "grenades.csv.gz")
