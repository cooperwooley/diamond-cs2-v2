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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# Output format of the exporter (``version`` in tools/demodump/main.go) and the pinned demoinfocs-golang release
# (tools/demodump/go.mod). tests/test_demodump.py checks both against the Go sources. The format version is part of
# the cache key, so a bump re-parses every cached demo.
DEMODUMP_FORMAT = "1"
DEMOINFOCS_VERSION = "v6.0.0-rc.1"
# ``behaviorVersion`` in tools/demodump/main.go: bumped whenever the derived-event rules or the table logic
# change, so cached dumps made by older logic are not reused. Tested against the Go source.
DEMODUMP_BEHAVIOR = "1"

ENV_BINARY = "DCS2_DEMODUMP"
# Linux path of the exporter, run through ``wsl.exe`` on Windows. Smart App Control blocks a self-built
# ``demodump.exe`` (WinError 4551), so on that PC the Linux build runs inside WSL instead (#78).
ENV_WSL = "DCS2_DEMODUMP_WSL"
WSL_EXE = "wsl.exe"
REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = "scripts/build_demodump.sh"
_TIMEOUT_S = 1800  # a 30 minute demo takes seconds; this only stops a hung process
_VERSION_RE = re.compile(r"^demodump (\S+) demoinfocs (\S+) behavior (\S+)$")

# Tick columns that exist in every dump; ``parse_ticks`` returns these plus the wanted ones the dump has.
_TICK_ALWAYS = ("tick", "name", "steamid", "user_id")


class DemodumpNotFound(FileNotFoundError):
    """No ``demodump`` binary could be found."""


def binary_name() -> str:
    """File name of the exporter on this OS."""
    return "demodump.exe" if os.name == "nt" else "demodump"


def _is_windows() -> bool:
    """Whether the pipeline runs on Windows (a seam for tests)."""
    return os.name == "nt"


_WIN_ABS_RE = re.compile(r"^([A-Za-z]):[\\/]+(.*)$")


def wsl_path(path: str | os.PathLike) -> str:
    r"""Translate an absolute Windows path to the path WSL sees it at: ``C:\dcs2\x.dem`` -> ``/mnt/c/dcs2/x.dem``.

    Assumes the default automount root ``/mnt/``.

    Raises:
        ValueError: If the path is not ``<drive>:\...`` (a UNC path, a relative path, or one without a drive).
    """
    text = os.fspath(path)
    match = _WIN_ABS_RE.match(text)
    if match is None:
        raise ValueError(
            f"cannot translate {text!r} to a WSL path: it must be an absolute path on a drive, like C:\\dcs2\\x.dem"
        )
    parts = [part for part in re.split(r"[\\/]+", match.group(2)) if part]
    return "/".join([f"/mnt/{match.group(1).lower()}", *parts])


@dataclass(frozen=True)
class Exporter:
    """How to run the exporter: the command prefix, and whether file arguments need translating to WSL paths.

    Attributes:
        argv: The command prefix (the binary, or ``wsl.exe -e <linux path>``).
        wsl: Whether it runs inside WSL, so file arguments are translated with ``wsl_path``.
    """

    argv: tuple[str, ...]
    wsl: bool = False

    def command(self, *files: str | os.PathLike) -> list[str]:
        """The command line for file arguments (the demo, the output directory); translated if it runs in WSL."""
        return [*self.argv, *(wsl_path(f) if self.wsl else str(f) for f in files)]

    def __str__(self) -> str:
        return " ".join(self.argv)


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


def find_exporter(
    configured: str | os.PathLike | None = None, configured_wsl: str | os.PathLike | None = None
) -> Exporter:
    """Pick how to run the exporter.

    On Windows, env ``DCS2_DEMODUMP_WSL`` (then ``demo.demodump_wsl``) names the Linux build to run through
    ``wsl.exe``; it wins over every other source, because setting it means the Windows ``.exe`` is blocked here.
    Elsewhere it is ignored. Otherwise the order of ``find_binary`` applies.

    Args:
        configured: The ``demo.demodump`` config value, if any.
        configured_wsl: The ``demo.demodump_wsl`` config value, if any.

    Raises:
        DemodumpNotFound: If the WSL path is not in use and ``find_binary`` finds nothing.
        ValueError: If the WSL path is used but is not an absolute Linux path.
    """
    wsl_value = os.environ.get(ENV_WSL) or (str(configured_wsl) if configured_wsl else None)
    if wsl_value:
        if _is_windows():
            if not wsl_value.startswith("/"):  # `wsl.exe -e` runs it as is: no `~` expansion, no Windows paths
                raise ValueError(f"{ENV_WSL} / demo.demodump_wsl must be an absolute Linux path, got {wsl_value!r}")
            return Exporter((WSL_EXE, "-e", wsl_value), wsl=True)
        log.debug("%s / demo.demodump_wsl only applies on Windows; ignoring %s", ENV_WSL, wsl_value)
    return Exporter((str(find_binary(configured)),))


def preflight(configured: str | os.PathLike | None = None, configured_wsl: str | os.PathLike | None = None) -> None:
    """Check that the exporter can be found and run, so a night does not discover a blocked one hours in.

    Raises:
        DemodumpNotFound: If there is no exporter.
        RuntimeError: As ``check_version``.
    """
    check_version(find_exporter(configured, configured_wsl))


def cache_key(demo_path: Path) -> str:
    """Cache directory name: the demo's stem plus a hash of its path, size, mtime and the exporter's versions."""
    st = demo_path.stat()
    raw = f"{demo_path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{DEMODUMP_FORMAT}|{DEMOINFOCS_VERSION}|{DEMODUMP_BEHAVIOR}"
    return f"{demo_path.stem}-{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def dump_demo(
    demo_path: str | Path,
    cache_dir: str | Path,
    configured: str | os.PathLike | None = None,
    configured_wsl: str | os.PathLike | None = None,
) -> Path:
    """Run ``demodump`` on a demo unless a valid dump is cached, and return the dump directory.

    The dump is written to a temp directory beside the cache entry and renamed into place, so an interrupted run
    leaves no half-written entry.

    Args:
        demo_path: ``.dem`` file.
        cache_dir: Directory holding the dumps.
        configured: The ``demo.demodump`` config value, passed to ``find_exporter`` (only needed on a cache miss).
        configured_wsl: The ``demo.demodump_wsl`` config value, passed to ``find_exporter``.

    Returns:
        ``<cache_dir>/<stem>-<hash>``.

    Raises:
        DemodumpNotFound: If there is no exporter.
        ValueError: If the exporter runs in WSL and a path cannot be translated.
        RuntimeError: If the exporter fails or has a different format version than ``DEMODUMP_FORMAT``.
    """
    demo_path = Path(demo_path)
    target = Path(cache_dir) / cache_key(demo_path)
    if (target / "header.json").is_file():
        return target
    exe = find_exporter(configured, configured_wsl)
    check_version(exe)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        # abspath, not resolve(): on Windows resolve() turns a mapped or subst drive into a UNC path WSL cannot take
        code, _, err = _run(exe.command(os.path.abspath(demo_path), os.path.abspath(tmp)), _TIMEOUT_S)
        if code != 0:
            raise RuntimeError(f"demodump failed on {demo_path} (exit {code}): {err.strip()[-500:]}")
        try:
            os.replace(tmp, target)
        except OSError:
            if not (target / "header.json").is_file():  # a concurrent run that renamed first is fine
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return target


def _decode(data: bytes | None) -> str:
    """Decode process output: ``wsl.exe`` writes its own messages as UTF-16, the exporter writes UTF-8."""
    if not data:
        return ""
    if b"\x00" in data:
        return data.decode("utf-16-le", errors="replace").lstrip("\ufeff")
    return data.decode("utf-8", errors="replace")


def _run(argv: list[str], timeout: float) -> tuple[int, str, str]:
    """Run a command and return ``(exit code, stdout, stderr)`` decoded."""
    proc = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
    return proc.returncode, _decode(proc.stdout), _decode(proc.stderr)


def check_version(exe: Exporter | Path) -> tuple[str, str, str]:
    """Return ``(format, demoinfocs version, behavior version)`` of an exporter.

    Raises:
        RuntimeError: If it cannot be run or does not report the versions the cache key assumes.
    """
    if not isinstance(exe, Exporter):
        exe = Exporter((str(exe),))
    try:
        code, out, err = _run([*exe.argv, "--version"], 60)
    except OSError as exc:
        raise RuntimeError(f"cannot run {exe}: {exc}") from exc
    match = _VERSION_RE.match(out.strip())
    if match is None and exe.wsl and code != 0:
        raise RuntimeError(
            f"{exe} --version failed (exit {code}): {(err or out).strip()[-300:]!r}. "
            f"Is the exporter at that Linux path in the default WSL distro? Check {ENV_WSL}."
        )
    if match is None:
        raise RuntimeError(
            f"{exe} --version printed {out.strip()!r}, expected 'demodump <n> demoinfocs <version> behavior <n>'"
        )
    found = match.groups()
    if found != (DEMODUMP_FORMAT, DEMOINFOCS_VERSION, DEMODUMP_BEHAVIOR):
        raise RuntimeError(
            f"{exe} is format {found[0]}, demoinfocs {found[1]}, behavior {found[2]}; cs2wm needs "
            f"{DEMODUMP_FORMAT}, {DEMOINFOCS_VERSION}, {DEMODUMP_BEHAVIOR}. Rebuild with {BUILD_SCRIPT}"
        )
    return found


def _occupant_at(tick: pd.Series, user_id: pd.Series, roster: pd.DataFrame) -> pd.DataFrame:
    """Who held each event's ``user_id`` at its tick.

    A ``user_id`` can be reused by another player, so the roster has one row per occupancy (``first_tick``,
    ``last_tick``). Each event takes the latest occupancy that started at or before its tick, or the first one if it
    precedes them all. Events without a ``user_id``, or with one the roster lacks, get NaN.

    Returns:
        A frame aligned with ``tick`` with ``name``, ``steamid`` and ``entity_id``.
    """
    cols = ["name", "steamid", "entity_id"]
    left = pd.DataFrame({"tick": tick.to_numpy(dtype="int64"), "user_id": user_id.to_numpy(), "row": range(len(tick))})
    known = left[left["user_id"].notna()].astype({"user_id": "int64"}).sort_values("tick", kind="stable")
    right = roster.astype({"user_id": "int64", "first_tick": "int64"}).sort_values("first_tick")[
        ["user_id", "first_tick", *cols]
    ]
    kwargs = {"left_on": "tick", "right_on": "first_tick", "by": "user_id"}
    back = pd.merge_asof(known, right, direction="backward", **kwargs).set_index("row")
    front = pd.merge_asof(known, right, direction="forward", **kwargs).set_index("row")
    joined = back[cols].copy()
    before_all = back["first_tick"].isna()
    joined.loc[before_all, cols] = front.loc[before_all, cols]
    return joined.reindex(range(len(tick))).reset_index(drop=True)


class DemodumpParser:
    """Drop-in for the parts of ``demoparser2.DemoParser`` that ``cs2wm.demo`` uses, backed by a cached dump."""

    def __init__(
        self,
        demo_path: str | Path,
        cache_dir: str | Path,
        configured: str | os.PathLike | None = None,
        configured_wsl: str | os.PathLike | None = None,
        *,
        dump: Path | None = None,
    ):
        """Dump the demo (or reuse the cache) and remember where.

        Args:
            demo_path: ``.dem`` file.
            cache_dir: Directory holding the dumps.
            configured: The ``demo.demodump`` config value.
            configured_wsl: The ``demo.demodump_wsl`` config value.
            dump: An existing dump directory to read instead of running the exporter (tests).
        """
        self.dir = dump if dump is not None else dump_demo(demo_path, cache_dir, configured, configured_wsl)

    def parse_header(self) -> dict[str, Any]:
        """The demo header: ``map_name``, ``patch_version``, ``tick_rate``, ``first_tick``, ``last_tick``, ..."""
        header = json.loads((self.dir / "header.json").read_text(encoding="utf-8"))
        if header.get("truncated"):
            log.warning(
                "%s is truncated: the demo ends early, the dump holds ticks %s..%s only",
                header.get("demo_file"),
                header.get("first_tick"),
                header.get("last_tick"),
            )
        return header

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
        roster = pd.read_csv(self.dir / "players.csv")
        out: list[tuple[str, pd.DataFrame]] = []
        for name in names:
            path = self.dir / "events" / f"{name}.csv.gz"
            if not path.is_file():
                continue
            df = pd.read_csv(path).sort_values("tick", kind="stable").reset_index(drop=True)
            uid = df["user_id"] if "user_id" in df.columns else pd.Series(float("nan"), index=df.index)
            who = _occupant_at(df["tick"], uid, roster)
            df["user_user_id"] = uid
            df["user_name"] = who["name"].to_numpy()
            df["user_steamid"] = who["steamid"].to_numpy()
            if player is not None and "entity_id" in player:
                df["user_entity_id"] = who["entity_id"].to_numpy()
            out.append((name, df))
        return out

    def parse_grenades(self, *, extra: object = None, grenades: bool = True) -> pd.DataFrame:
        """Grenade trajectories: one row per (``grenade_entity_id``, ``tick``) with the thrower and ``X``/``Y``/``Z``."""
        return pd.read_csv(self.dir / "grenades.csv.gz")
