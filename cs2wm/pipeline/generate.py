"""generate: unattended offline CS2 bot-deathmatch demo generation, Windows only (#13).

Launches CS2 offline as a listen server, records demos in N-minute chunks to ``demos/<kind>/<demo_id>.dem`` with a
``<demo_id>.json`` sidecar (start time, profile, chunk length, CS2 build id, cfg sha256, cs2wm git sha), and restarts
on a crash or hang. The launcher REFUSES to launch CS2 without ``-insecure``: HLAE and CS2 are only ever run offline.

How it works:

* CS2 is started directly (``game/bin/win64/cs2.exe``), not through HLAE (HLAE is for render only) and not through
  ``steam -applaunch 730`` (steam.exe returns at once, so there would be no process to watch or kill).
* ``config/cs2/generate_<profile>.cfg`` is rendered (``exec generate_*`` includes inlined, chunking commands appended),
  copied to ``<cs2_dir>/game/csgo/cfg/generate_<profile>.cfg`` and run with ``+exec generate_<profile>``. Its sha256
  goes into every sidecar.
* Both chunking modes are driven over ``-netconport``. After the map has loaded the local player joins the spectators
  (``jointeam 1``, lost if sent from the cfg because the client is not connected yet), optionally followed by
  ``mp_restartgame 1`` (``generate.restart_after_join``, repeated before EVERY chunk, never while recording: the
  10-minute DM round would otherwise end inside a chunk and the level change ends the GOTV recording, #79); every chunk, ``gotv``
  (default) sends ``sv_cheats 1; tv_record <demo_id>`` / ``tv_stoprecord`` and ``netcon`` sends
  ``sv_cheats 1; record <demo_id>`` / ``stop``. ``tv_autorecord`` is not used: it writes ``replays/auto-*.dem`` with
  uncontrolled names. Both ``record`` and ``tv_record`` write ``game/csgo/<demo_id>.dem``.
* CS2 writes into its own ``game/csgo`` dir. A closed chunk is copied next to its destination and renamed into
  ``paths.demos/<kind>/`` atomically (sidecar first, so a ``.dem`` is never visible without it).
* A watchdog restarts CS2 on exit or when no demo file grew for ``generate.hang_timeout_min``. A truncated chunk is
  kept only if ``cs2wm.demo.inspect`` parses it, else it goes to ``demos/_rejected/`` with a reason file.
* The patch guard compares the installed CS2 buildid with ``generate.expected_build`` before each launch and chunk.

The CS2 facts behind this module (command order, timings, the netcon line limit) are in ``docs/HLAE_NOTES.md``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any

from omegaconf import DictConfig

from cs2wm import config as cs2wm_config
from cs2wm import demo
from cs2wm.pipeline import ntfy
from cs2wm.pipeline import proc as proc_utils
from cs2wm.pipeline.netcon import NetconClient

log = logging.getLogger(__name__)

REJECTED_DIR = "_rejected"
# Per chunking mode: (start, stop) console commands. `record` and `tv_record` are cheat-protected (offline only, with
# -insecure) and both write game/csgo/<name>.dem; `sv_cheats 1` goes on the same line because it resets on a map load.
# The third item is the text of a successful start reply (#14 log): `Recording SourceTV replay from HLTVServerAsync:` and
# `CDemoRecorder::StartRecording(...)`.
CHUNK_COMMANDS = {
    "gotv": ("sv_cheats 1; tv_record {name}", "tv_stoprecord", "Recording SourceTV"),
    "netcon": ("sv_cheats 1; record {name}", "stop", "StartRecording"),
}
# `jointeam 1` takes effect when the server logs this (#14: `ChangeTeam() CTMDBG , team 2, req team 1 willSwitch 0`).
_TEAM_OK_RE = re.compile(r"ChangeTeam\(\).*req team 1")
_HUMANS_RE = re.compile(r"(\d+)\s+humans?")
SPECTATE_EXEMPT_PROFILE = "human"  # a human plays, so the local player must stay on a team
_EXEC_RE = re.compile(r"^\s*exec\s+(generate_\w+)\s*(?://.*)?$")


class PatchGuardError(RuntimeError):
    """The installed CS2 build differs from ``generate.expected_build``; generation stopped."""


class GenerateError(RuntimeError):
    """Generation gave up (CS2 would not stay up)."""


class _RecordingFailed(Exception):
    """The start command did not start a recording (no success text in the reply, or no data in the demo)."""


# ---------------------------------------------------------------------------------------------------------------------
# Launch command


def _require_insecure(cmd: Sequence[str]) -> None:
    """Raise ``ValueError`` unless ``-insecure`` is in ``cmd``, or if it asks CS2 to connect to a server or go secure."""
    proc_utils.require_offline(cmd)


def build_launch_command(cfg: DictConfig, profile: str, netconport: int | None = None) -> list[str]:
    """Build the CS2 launch command line from ``generate.launch_options``.

    Args:
        cfg: A validated cs2wm config.
        profile: ``dm``, ``smoke`` or ``human``; selects ``config/cs2/generate_<profile>.cfg``.
        netconport: ``-netconport`` for the chunk commands.

    Returns:
        The argv list: ``cs2.exe``, the launch options, ``-netconport <port>`` if given and ``+exec generate_<profile>``.

    Raises:
        ValueError: If ``-insecure`` is not among the launch options, ``-secure``/``+connect`` is, or the profile is unknown.
    """
    options = [str(o) for o in cfg.generate.launch_options]
    _require_insecure(options)
    if profile not in cfg.episodes.kinds:
        raise ValueError(f"unknown profile {profile!r}; expected one of {list(cfg.episodes.kinds)}")
    cmd = [str(Path(cfg.paths.cs2_dir) / cfg.generate.cs2_exe), *options]
    if netconport is not None:
        cmd += ["-netconport", str(int(netconport))]
    cmd += ["+exec", f"generate_{profile}"]
    return cmd


# ---------------------------------------------------------------------------------------------------------------------
# Console cfg


def render_cfg(profile: str, cfg: DictConfig, cfg_dir: Path | None = None) -> str:
    """Render ``generate_<profile>.cfg`` into the single file that is installed into CS2.

    ``exec generate_*`` lines are replaced by the included file (``generate_smoke`` builds on ``generate_dm``).

    Args:
        profile: ``dm``, ``smoke`` or ``human``.
        cfg: A validated cs2wm config.
        cfg_dir: Directory with the source cfgs (default: ``config/cs2`` of the repo).

    Returns:
        The cfg text.

    Raises:
        FileNotFoundError: If a cfg is missing.
        ValueError: On an ``exec`` cycle.
    """
    return "\n".join(_expand_cfg(f"generate_{profile}", (), cfg_dir)) + "\n"


def _expand_cfg(name: str, stack: tuple[str, ...], cfg_dir: Path | None) -> list[str]:
    """Lines of ``<name>.cfg`` with its ``exec generate_*`` includes inlined."""
    cfg_dir = cfg_dir or cs2wm_config.DEFAULT_CONFIG_PATH.parent / "cs2"
    if name in stack:
        raise ValueError(f"exec cycle in cfgs: {' -> '.join((*stack, name))}")
    out = [f"// ---- {name}.cfg"]
    for line in (cfg_dir / f"{name}.cfg").read_text(encoding="utf-8").splitlines():
        m = _EXEC_RE.match(line)
        out += _expand_cfg(m.group(1), (*stack, name), cfg_dir) if m else [line]
    return out


def cvar_value(name: str, reply: str) -> str | None:
    """The value of cvar ``name`` in a console reply, or None if the reply has no line for it.

    CS2 answers a bare cvar query with ``bot_difficulty = 2`` (checked on the PC, Nov 5); the CS:GO-style
    ``"bot_difficulty" = "2" ( def. "1" )`` is accepted too. The reply may carry unrelated log lines.
    """
    pattern = rf'(?m)^[ \t]*"?{re.escape(name)}"?[ \t]*=[ \t]*"?([^"\r\n]*?)"?(?:[ \t]+\(.*)?[ \t]*\r?$'
    matches = re.findall(pattern, reply)
    return matches[-1].strip() if matches else None


def post_load_commands(profile: str, cfg_dir: Path | None = None) -> list[str]:
    """The commands of ``generate_<profile>_post.cfg``: cvars that the map load resets and generate sends over netcon.

    Args:
        profile: ``dm``, ``smoke`` or ``human``.
        cfg_dir: Directory with the source cfgs (default: ``config/cs2`` of the repo).

    Returns:
        One command per line, comments and blank lines dropped; empty if the profile has no such cfg.

    Raises:
        ValueError: On an ``exec`` cycle.
    """
    name = f"generate_{profile}_post"
    if not ((cfg_dir or cs2wm_config.DEFAULT_CONFIG_PATH.parent / "cs2") / f"{name}.cfg").is_file():
        return []
    lines = (ln.split("//", 1)[0].strip() for ln in _expand_cfg(name, (), cfg_dir))
    return [ln for ln in lines if ln]


def install_cfg(text: str, profile: str, cs2_dir: Path) -> tuple[Path, str]:
    """Write a rendered cfg to ``<cs2_dir>/game/csgo/cfg/generate_<profile>.cfg``.

    Args:
        text: Output of ``render_cfg``.
        profile: ``dm``, ``smoke`` or ``human``.
        cs2_dir: CS2 install dir (``paths.cs2_dir``).

    Returns:
        The installed path and the sha256 of its content.
    """
    dest = Path(cs2_dir) / "game" / "csgo" / "cfg" / f"generate_{profile}.cfg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    if not dest.exists() or dest.read_bytes() != data:
        dest.write_bytes(data)
    return dest, hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# Chunk schedule


def plan_chunks(hours: float, chunk_min: float) -> list[float]:
    """Split ``hours`` of demo time into chunk lengths.

    Args:
        hours: Total demo time.
        chunk_min: Chunk length in minutes.

    Returns:
        Chunk lengths in minutes; all equal to ``chunk_min`` except a shorter last one.
    """
    total = hours * 60.0
    if total <= 0 or chunk_min <= 0:
        return []
    n = math.ceil(total / chunk_min - 1e-9)
    lengths = [chunk_min] * n
    lengths[-1] = total - chunk_min * (n - 1)
    return lengths


def make_demo_id(kind: str, started: datetime, index: int) -> str:
    """Build ``<kind>_<YYYYmmdd-HHMMSS>_<nnn>``: unique per run and chunk, sortable and filesystem-safe.

    Args:
        kind: Episode kind (the profile).
        started: Start time of the generate run.
        index: Chunk number within the run.

    Returns:
        The demo id; also the console ``record`` name and the file stem.
    """
    return f"{kind}_{started:%Y%m%d-%H%M%S}_{index:03d}"


# ---------------------------------------------------------------------------------------------------------------------
# Watchdog


class Watchdog:
    """Pure hang/exit state machine: inputs are liveness, demo bytes and a clock reading.

    Args:
        timeout_s: Declare a hang after this long without growth.
        now: Current clock reading; the grace period starts here.
    """

    def __init__(self, timeout_s: float, now: float) -> None:
        self.timeout_s = timeout_s
        self._last_progress = now
        self._last_size: int | None = None

    @property
    def last_progress(self) -> float:
        """Clock reading of the last growth (or the last rearm)."""
        return self._last_progress

    def rearm(self, now: float) -> None:
        """Restart the grace period (after a launch or at a chunk boundary)."""
        self._last_progress = now

    def observe(self, now: float, alive: bool, size: int) -> str:
        """Feed one observation.

        Args:
            now: Clock reading.
            alive: Whether the process is still running.
            size: Total bytes of the watched demo files. Only an increase counts as growth, so moving a finished
                chunk away (a decrease) does not.

        Returns:
            ``"exited"``, ``"hung"`` or ``"running"``.
        """
        if not alive:
            return "exited"
        if self._last_size is not None and size > self._last_size:
            self._last_progress = now
        self._last_size = size
        return "hung" if now - self._last_progress >= self.timeout_s else "running"


def _file_size(path: Path) -> int | None:
    """Size of ``path`` in bytes, None if it does not exist."""
    try:
        return path.stat().st_size
    except OSError:
        return None


def _dir_size(directory: Path) -> int:
    """Total size of the ``*.dem`` files in ``directory``."""
    return sum(_file_size(p) or 0 for p in directory.glob("*.dem"))


def _is_windows() -> bool:
    return os.name == "nt"


class CS2Process:
    """A supervised CS2 process with a hang watchdog.

    Args:
        cmd: Launch command from ``build_launch_command``.
        demo_dir: Directory watched for demo growth.
        hang_timeout_min: Kill and relaunch after this long without demo-file growth
            (``generate.hang_timeout_min``).
        popen: Process factory (``subprocess.Popen``-like; injectable for tests).
        clock: Monotonic clock in seconds.
        size_probe: ``demo_dir -> total demo bytes``.
        run: ``subprocess.run``-like, used for ``taskkill``.

    Raises:
        ValueError: If ``-insecure`` is missing from ``cmd``.
    """

    def __init__(
        self,
        cmd: Sequence[str],
        demo_dir: Path,
        hang_timeout_min: float,
        *,
        popen: Callable[..., Any] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
        size_probe: Callable[[Path], int] = _dir_size,
        run: Callable[..., Any] = subprocess.run,
    ) -> None:
        _require_insecure(cmd)
        self.cmd = list(cmd)
        self.demo_dir = demo_dir
        self.hang_timeout_min = hang_timeout_min
        self._popen = popen
        self._clock = clock
        self._size_probe = size_probe
        self._run = run
        self._proc: Any = None
        self._watchdog: Watchdog | None = None

    def __enter__(self) -> CS2Process:  # noqa: PYI034
        """Launch CS2."""
        _require_insecure(self.cmd)
        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if _is_windows():
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # Ctrl-C in our console must not reach CS2
        log.info("launching CS2: %s", " ".join(self.cmd))
        self._proc = self._popen(self.cmd, **kwargs)
        self._watchdog = Watchdog(self.hang_timeout_min * 60.0, self._clock())
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Kill CS2 if still running."""
        self.kill()

    def rearm(self) -> None:
        """Restart the hang grace period (a new chunk starts)."""
        if self._watchdog is not None:
            self._watchdog.rearm(self._clock())

    @property
    def last_progress(self) -> float:
        """Clock reading of the last demo growth seen by the watchdog."""
        if self._watchdog is None:
            raise RuntimeError("CS2Process is not started")
        return self._watchdog.last_progress

    def poll(self) -> str:
        """Advance the watchdog state machine.

        Returns:
            ``"running"``, ``"exited"`` or ``"hung"``.
        """
        if self._proc is None or self._watchdog is None:
            raise RuntimeError("CS2Process is not started")
        alive = self._proc.poll() is None
        return self._watchdog.observe(self._clock(), alive, self._size_probe(self.demo_dir))

    def kill(self) -> None:
        """Kill the process tree (``taskkill /T /F`` on Windows). Safe to call twice.

        On Windows ``taskkill`` also runs when the main process already exited, because a crash handler or child of
        ``cs2.exe`` may still be alive.
        """
        proc = self._proc
        if proc is None:
            return
        if _is_windows():
            try:
                self._run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False, timeout=30)
            except (OSError, subprocess.SubprocessError) as exc:
                log.warning("taskkill failed: %s", exc)
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log.warning("CS2 (pid %s) did not exit after kill", proc.pid)


# ---------------------------------------------------------------------------------------------------------------------
# Runner


def _git_sha() -> str:
    """The cs2wm git commit (``unknown`` outside a checkout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _parse_error(path: Path) -> str | None:
    """Why ``path`` is not a usable demo, or None if it is (the single seam to ``cs2wm.demo``).

    A chunk cut short by a crash is usable: demoinfocs parses it up to its last flushed frame and reports
    ``ErrUnexpectedEndOfDemo`` (#14). The interface expected from the parser backend (#72) is either
    ``demo.is_parseable(path) -> bool`` (True for such a demo), or ``demo.inspect(path)`` returning ``ok``, ``error``
    and, for a demo that ends early, ``truncated: True`` together with ``last_tick`` (the last tick it parsed). A
    truncated demo with ticks counts as usable even if ``ok`` is False. Either may be missing on older bases, and any
    exception also counts as unparsable.
    """
    try:
        is_parseable = getattr(demo, "is_parseable", None)
        if is_parseable is not None:
            return None if is_parseable(path) else "demo.is_parseable() is False"
        info = demo.inspect(path)
        if info.get("ok", info.get("last_tick") is not None):
            return None
        if info.get("truncated") and (info.get("last_tick") or 0) > 0:
            return None  # ended early but parsed up to there: keep it as a truncated chunk
        return str(info.get("error") or "no ticks parsed")
    except Exception as exc:  # noqa: BLE001 - any parser failure means the chunk is unusable
        return repr(exc)


def _demo_seconds(path: Path) -> float | None:
    """Length of the demo in seconds from its own ticks (``demo.inspect``), None if it cannot be read."""
    try:
        info = demo.inspect(path)
        first, last = info.get("first_tick"), info.get("last_tick")
        if first is None or last is None:
            return None
        rate = (info.get("header") or {}).get("tick_rate")
        return max(float(last) - float(first), 0.0) / float(rate or 64)
    except Exception:  # noqa: BLE001 - the wall-clock estimate stays
        return None


@dataclass
class Hooks:
    """Everything the runner takes from the outside world, replaced by fakes in tests."""

    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    wall: Callable[[], datetime] = datetime.now
    popen: Callable[..., Any] = subprocess.Popen
    run: Callable[..., Any] = subprocess.run
    netcon: Callable[..., NetconClient] = NetconClient
    file_size: Callable[[Path], int | None] = _file_size
    dir_size: Callable[[Path], int] = _dir_size
    build_id: Callable[[Path], int] = demo.cs2_build_id
    parse_error: Callable[[Path], str | None] = lambda path: _parse_error(path)
    demo_seconds: Callable[[Path], float | None] = lambda path: _demo_seconds(path)
    notify: Callable[..., None] = ntfy.notify
    git_sha: Callable[[], str] = _git_sha
    copy: Callable[[Path, Path], Any] = shutil.copyfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class Generator:
    """Runs one ``cs2wm generate`` invocation.

    Args:
        cfg: A validated cs2wm config.
        profile: ``dm``, ``smoke`` or ``human``.
        hours: Demo time to record.
        chunk_min: Chunk length override (``generate.chunk_min``).
        hooks: Clock, process factory, filesystem probe, ... (fakes in tests).
        cfg_dir: Source cfg directory override.

    Raises:
        ValueError: If ``hours`` or the chunk length is not positive.
    """

    def __init__(
        self,
        cfg: DictConfig,
        profile: str,
        hours: float,
        chunk_min: float | None = None,
        hooks: Hooks | None = None,
        cfg_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.g = cfg.generate
        self.profile = profile
        self.chunk_min = float(chunk_min if chunk_min is not None else self.g.chunk_min)
        if not hours > 0:
            raise ValueError(f"--hours must be > 0, got {hours}")
        if not self.chunk_min > 0:
            raise ValueError(f"--chunk-min must be > 0, got {self.chunk_min}")
        self.target_s = hours * 3600.0
        self.h = hooks or Hooks()
        self.cfg_dir = cfg_dir
        self.cs2_dir = Path(cfg.paths.cs2_dir)
        self.csgo_dir = self.cs2_dir / "game" / "csgo"
        self.out_dir = Path(cfg.paths.demos) / profile
        self.rejected_dir = Path(cfg.paths.demos) / REJECTED_DIR
        self.kept: list[Path] = []
        self.generated_s = 0.0  # demo time of kept chunks
        self.cfg_sha = ""
        self.git_sha = ""
        self.started = self.h.wall()
        self._index = 0  # chunk counter, continues across CS2 restarts
        self._progress = 0  # chunks published in the current session (truncated ones only if >= generate.min_chunk_s)
        self._publish_failures = 0  # publishes that failed since the last success
        self._pending_unlink: list[Path] = []  # published sources CS2 still holds open; deleted again later
        self._published_sources: set[Path] = set()  # every source published or rejected in this run (never re-adopted)
        self._unpublished: set[Path] = set()  # sources whose publishing failed and that are still waiting
        self._notified: set[tuple[str, str]] = set()  # (step, error type) already notified in this run
        self._ours = re.compile(rf"^{re.escape(profile)}_\d{{8}}-\d{{6}}_\d{{3}}\.dem$")

    # -- patch guard ------------------------------------------------------------------------------------------------

    def _build(self) -> int | None:
        try:
            return self.h.build_id(self.cs2_dir)
        except (FileNotFoundError, ValueError) as exc:
            log.warning("cannot read the CS2 build id: %s", exc)
            return None

    def _guard(self) -> int | None:
        """Stop with an ntfy alert if CS2 was patched.

        Returns:
            The current build id.

        Raises:
            PatchGuardError: If ``generate.expected_build`` is set and differs from the installed build.
        """
        build = self._build()
        expected = self.g.expected_build
        if expected is not None and build != int(expected):
            msg = f"CS2 build is {build}, expected {expected}. Generation stopped; run the render smoke test (#20)."
            log.error(msg)
            self.h.notify(msg, title="dcs2: CS2 patched", priority="high", tags=["warning"])
            raise PatchGuardError(msg)
        return build

    # -- finishing chunks -------------------------------------------------------------------------------------------

    def _wait_closed(self, path: Path) -> bool:
        """Wait until ``path`` exists and its size has been stable for ``close_stable_s``.

        Returns:
            True if it closed with data in it. False on timeout (e.g. it is still growing) or if it is missing or
            empty; the file is then left where it is and picked up again in a later cycle.
        """
        start = self.h.clock()
        last: int | None = None
        stable_since = start
        while True:
            now = self.h.clock()
            size = self.h.file_size(path)
            if size != last:
                last, stable_since = size, now
            elif size and now - stable_since >= self.g.close_stable_s:
                return True
            if now - start >= self.g.close_timeout_s:
                return False
            self.h.sleep(min(1.0, self.g.close_stable_s))

    def _sidecar(
        self,
        demo_id: str,
        started: datetime | None,
        duration_s: float,
        truncated: bool,
        src: Path,
        nbytes: int,
        sha256: str,
        extra: dict,
    ) -> dict:
        return {
            "demo_id": demo_id,
            "kind": self.profile,
            "profile": self.profile,
            "start_time": started.isoformat(timespec="seconds") if started else None,
            "chunk_min": self.chunk_min,
            "duration_s": round(duration_s, 1),
            "chunking": str(self.g.chunking),
            "truncated": truncated,
            "bytes": nbytes,
            "sha256": sha256,
            "source_name": src.name,
            "cs2_build": self._build(),
            "cfg_sha256": self.cfg_sha,
            "cs2wm_git_sha": self.git_sha,
            **extra,
        }

    def _retry(self, op: Callable[[], Any], what: str) -> OSError | None:
        """Run ``op``, retrying ``OSError`` with backoff for up to ``close_timeout_s``.

        CS2 can hold a demo open for a while after its size stopped changing; on Windows copying then raises
        ``PermissionError``.

        Returns:
            None on success, else the last error.
        """
        start, delay = self.h.clock(), 0.5
        while True:
            try:
                op()
                return None
            except OSError as exc:
                if self.h.clock() - start >= self.g.close_timeout_s:
                    log.error("giving up %s: %s", what, exc)
                    return exc
                log.debug("%s failed (%s), retry in %.1f s", what, exc, delay)
                self.h.sleep(delay)
                delay = min(delay * 2, 10.0)

    @staticmethod
    def _unlink_quiet(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not delete %s: %s", path, exc)

    def _copy(self, src: Path, dest: Path) -> OSError | None:
        err = self._retry(lambda: self.h.copy(src, dest), f"copying {src.name}")
        if err is not None:
            self._unlink_quiet(dest)
        return err

    def _remove_source(self, src: Path) -> None:
        """Delete a published source; if CS2 still holds it, remember it (it is never adopted again)."""
        self._published_sources.add(src)
        self._unpublished.discard(src)
        try:
            src.unlink()
        except OSError as exc:
            log.warning("could not delete %s after copying (retried later): %s", src, exc)
            self._pending_unlink.append(src)

    def _unpublished_failure(self, src: Path, what: str, exc: OSError | None) -> bool:
        """Log that ``src`` could not be moved (the file stays where it is) and notify once per error type."""
        self._publish_failures += 1
        self._unpublished.add(src)
        err = type(exc).__name__ if exc else "error"
        log.error("%s failed for %s (%s); the file stays in %s", what, src.name, err, src.parent)
        if (what, err) not in self._notified:
            self._notified.add((what, err))
            msg = f"{what} failed ({err}: {exc}); demos stay in {src.parent} and are retried"
            self.h.notify(msg, title="dcs2: generate publish failed", priority="high", tags=["warning"])
        return False

    def _unique_id(self, demo_id: str) -> str:
        """Return ``demo_id``, or ``demo_id`` plus a suffix if a demo with that id already exists."""
        candidate, n = demo_id, 1
        while any(
            (d / f"{candidate}{ext}").exists() for d in (self.out_dir, self.rejected_dir) for ext in (".dem", ".json")
        ):
            candidate, n = f"{demo_id}x{n}", n + 1
        return candidate

    def _publish(
        self,
        src: Path,
        demo_id: str,
        started: datetime | None,
        duration_s: float,
        truncated: bool,
        extra: dict | None = None,
    ) -> bool:
        """Keep a closed chunk, or reject a truncated one that does not parse.

        A failure to copy or rename (file locks, full disk) is logged and notified but never raised; ``src`` stays in
        place and the chunk does not count toward ``--hours``.

        Returns:
            True if the chunk was kept or rejected (``src`` is dealt with), False if publishing failed.
        """
        if truncated:
            error = self.h.parse_error(src)
            if error is not None:
                return self._reject(src, demo_id, f"truncated chunk does not parse: {error}")
            # Wall time overstates a crashed chunk: the last tv_delay seconds never reached the file.
            seconds = self.h.demo_seconds(src) if duration_s > 0 else None
            if seconds is not None:
                duration_s = seconds
        demo_id = self._unique_id(demo_id)
        dest = self.out_dir / f"{demo_id}.dem"
        part = dest.with_name(dest.name + ".part")
        sidecar = self.out_dir / f"{demo_id}.json"
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return self._unpublished_failure(src, "creating the demos dir", exc)
        nbytes = self.h.file_size(src) or 0
        err = self._copy(src, part)
        if err is not None:
            return self._unpublished_failure(src, "copying", err)
        try:
            sha256 = _sha256(part)
            # sidecar first, then the .dem: a .dem is never visible without its sidecar
            side = self._sidecar(demo_id, started, duration_s, truncated, src, nbytes, sha256, extra or {})
            _write_json_atomic(sidecar, side)
        except OSError as exc:
            self._unlink_quiet(part)
            self._unlink_quiet(sidecar)
            return self._unpublished_failure(src, "writing the sidecar", exc)
        err = self._retry(lambda: os.replace(part, dest), f"renaming {demo_id}")
        if err is not None:
            self._unlink_quiet(sidecar)
            self._unlink_quiet(part)
            return self._unpublished_failure(src, "renaming", err)
        self._publish_failures = 0
        self._remove_source(src)
        self.kept.append(dest)
        self.generated_s += duration_s
        if not truncated or duration_s >= self.g.min_chunk_s:  # a crash right after `record` is not progress
            self._progress += 1
        log.info("kept %s (%.0f s, %d bytes%s)", dest, duration_s, nbytes, ", truncated" if truncated else "")
        return True

    def _reject(self, src: Path, demo_id: str, reason: str) -> bool:
        demo_id = self._unique_id(demo_id)
        try:
            self.rejected_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return self._unpublished_failure(src, "creating the rejected dir", exc)
        dest = self.rejected_dir / f"{demo_id}.dem"
        part = dest.with_name(dest.name + ".part")
        err = self._copy(src, part)
        if err is not None:
            return self._unpublished_failure(src, "copying a rejected chunk", err)
        err = self._retry(lambda: os.replace(part, dest), f"renaming rejected {demo_id}")
        if err is None:
            try:
                (self.rejected_dir / f"{demo_id}.reason.txt").write_text(reason + "\n", encoding="utf-8")
            except OSError as exc:
                log.warning("could not write the reason file for %s: %s", demo_id, exc)
        else:
            self._unlink_quiet(part)
            return self._unpublished_failure(src, "renaming a rejected chunk", err)
        self._publish_failures = 0
        self._remove_source(src)
        log.warning("rejected %s: %s", demo_id, reason)
        return True

    def _salvage(self, src: Path, demo_id: str, started: datetime, duration_s: float) -> None:
        """Handle the chunk that was being recorded when CS2 died or hung."""
        if self._wait_closed(src):
            self._publish(src, demo_id, started, duration_s, truncated=True)
        else:
            log.warning("%s: no demo data to salvage (missing, empty or still growing)", demo_id)

    def _already_published(self, path: Path) -> bool:
        """Whether a sidecar in ``demos/<kind>`` records ``path`` as its source (same name, size and sha256).

        Covers a run that died between the final rename and deleting the source.
        """
        try:
            size = path.stat().st_size
            sidecars = list(self.out_dir.glob("*.json"))
        except OSError:
            return False
        digest = None
        for sc in sidecars:
            try:
                meta = json.loads(sc.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if meta.get("source_name") == path.name and meta.get("bytes") == size:
                digest = digest or _sha256(path)
                if meta.get("sha256") == digest:
                    return True
        return False

    def _adopt_leftovers(self) -> None:
        """Publish demos this tool recorded earlier but could not move (``<kind>_<stamp>_<nnn>.dem`` in the csgo dir).

        Called while no CS2 of ours is running, after ``_flush_unlinks``. Only this profile's names are considered.
        Skipped: sources already published or rejected in this run (including ones whose delete is still pending) and
        sources a sidecar of an earlier run lists (same name, size and sha256; those are just deleted). The rest is
        parse-checked like a truncated chunk (kept or rejected) and gets ``"adopted": true``, ``duration_s`` 0 (it does
        not count toward ``--hours``) and ``start_time`` null with the file's mtime as ``end_time``, since its start is
        unknown.
        """
        for path in sorted(self.csgo_dir.glob("*.dem")):
            if not self._ours.match(path.name) or path in self._published_sources or path in self._pending_unlink:
                continue
            if self._already_published(path):
                log.info("%s was published by an earlier run; deleting the source", path)
                self._remove_source(path)
                continue
            try:
                end_time = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
            except OSError:
                continue
            log.info("adopting leftover demo %s", path)
            extra = {"adopted": True, "end_time": end_time}
            self._publish(path, path.stem, None, 0.0, truncated=True, extra=extra)

    def _flush_unlinks(self) -> None:
        """Retry deleting published sources; those that still fail stay pending (and are never re-adopted)."""
        pending, self._pending_unlink = self._pending_unlink, []
        for path in pending:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                log.error("%s is a published demo still left in the csgo dir (delete it by hand): %s", path, exc)
                self._pending_unlink.append(path)

    # -- sessions ---------------------------------------------------------------------------------------------------

    def _watch(self, proc: CS2Process, seconds: float) -> str:
        """Sleep ``seconds`` while polling the watchdog; return early with ``exited``/``hung``."""
        end = self.h.clock() + seconds
        while True:
            verdict = proc.poll()
            if verdict != "running" or self.h.clock() >= end:
                return verdict
            self.h.sleep(min(self.g.poll_s, max(end - self.h.clock(), 0.0)))

    def _next_chunk_s(self) -> float:
        return min(self.chunk_min * 60.0, max(self.target_s - self.generated_s, 0.0))

    def _map_ready(self, status: str) -> bool:
        """Whether a ``status`` reply shows the map and a connected local client."""
        humans = _HUMANS_RE.search(status)
        return str(self.g.map) in status and (humans is None or int(humans.group(1)) >= 1)

    def _wait_for_map(self, client: NetconClient, proc: CS2Process) -> None:
        """Poll ``status`` until the map has loaded and the local client is connected, then settle."""
        deadline = self.h.clock() + float(self.g.map_load_timeout_s)
        while not self._map_ready(client.send("status", retries=0)):
            if self.h.clock() >= deadline:
                raise ConnectionError(f"{self.g.map} not loaded after {self.g.map_load_timeout_s} s")
            if self._watch(proc, 2.0) != "running":
                raise ConnectionError("CS2 died while loading")
        if self._watch(proc, float(self.g.map_load_wait_s)) != "running":  # the client needs a moment after the map
            raise ConnectionError("CS2 died while loading")

    def _join_spectators(self, client: NetconClient, proc: CS2Process) -> None:
        """Send ``jointeam 1`` until the server logs the team change (retried), then let it settle."""
        for attempt in range(int(self.g.jointeam_retries) + 1):
            reply = client.send("jointeam 1")
            if not self.g.verify_netcon_replies or _TEAM_OK_RE.search(reply):
                break
            log.warning("jointeam 1 not confirmed (attempt %d): %r", attempt + 1, reply[-200:])
            if self._watch(proc, float(self.g.team_settle_s) / 2) != "running":
                raise ConnectionError("CS2 died while joining the spectators")
        else:
            raise ConnectionError("jointeam 1 never took effect")
        if self._watch(proc, float(self.g.team_settle_s)) != "running":
            raise ConnectionError("CS2 died while joining the spectators")

    def _send_post_load(self, client: NetconClient) -> None:
        """Send the cvars the map load resets, then read single-value ones back and warn if one did not stick."""
        commands = post_load_commands(self.profile, self.cfg_dir)
        if not commands:
            return
        client.send("; ".join(commands))  # the client splits it into lines of at most 400 characters
        for command in commands:
            parts = command.split()
            if len(parts) != 2:
                continue
            name, want = parts
            got = cvar_value(name, client.send(name))
            if got != want:
                log.warning("%s did not stick after the map load: wanted %s, read back %s", name, want, got)

    def _await_demo_data(self, proc: CS2Process, src: Path) -> None:
        """Wait up to ``record_check_s`` for the demo file to hold data (fail fast, not after the hang timeout)."""
        deadline = self.h.clock() + float(self.g.record_check_s)
        while not self.h.file_size(src):
            if self._watch(proc, min(float(self.g.poll_s), 2.0)) != "running":
                raise ConnectionError(f"CS2 died before {src.name} had data")
            if self.h.clock() >= deadline:
                raise _RecordingFailed(f"{src.name} has no data {self.g.record_check_s} s after the start command")

    def _session_netcon(self, proc: CS2Process) -> None:
        """Record chunks over netcon until the target or a failure (``gotv`` and ``netcon`` chunking)."""
        client = self.h.netcon(port=int(self.g.netconport), clock=self.h.clock, sleep=self.h.sleep)
        start_cmd, stop_cmd, start_ok = CHUNK_COMMANDS[str(self.g.chunking)]
        active: tuple[str, datetime, float] | None = None
        end: float | None = None  # when recording really ended: before the close wait, and before a hang's idle tail
        try:
            client.wait_ready(float(self.g.netcon_connect_timeout_s))
            self._wait_for_map(client, proc)
            spectate = self.profile != SPECTATE_EXEMPT_PROFILE
            if spectate:
                # Joining a team after `tv_record` started stalled the GOTV demo, so spectate first and let it settle.
                self._join_spectators(client, proc)
            self._send_post_load(client)
            while self.generated_s < self.target_s and self._publish_failures < self.g.max_failures:
                self._guard()
                if spectate and self.g.restart_after_join:
                    # The recipe that gave a full GOTV demo (#14), before every chunk and never while recording: a round
                    # that ends inside a chunk changes level and ends the GOTV recording (#79).
                    client.send("mp_restartgame 1")
                    if self._watch(proc, float(self.g.restart_wait_s)) != "running":
                        raise ConnectionError("CS2 died while restarting the game")
                demo_id = make_demo_id(self.profile, self.started, self._index)
                self._index += 1
                started, t0 = self.h.wall(), self.h.clock()
                length_s = self._next_chunk_s()
                src = self.csgo_dir / f"{demo_id}.dem"
                reply = client.send(start_cmd.format(name=demo_id))
                proc.rearm()
                if self.g.verify_netcon_replies and start_ok not in reply:
                    raise _RecordingFailed(f"no {start_ok!r} in the reply to the start command: {reply[-200:]!r}")
                active, end = (demo_id, started, t0), None
                try:
                    self._await_demo_data(proc, src)
                except _RecordingFailed:
                    active = None  # nothing worth salvaging
                    raise
                verdict = self._watch(proc, max(length_s - (self.h.clock() - t0), 0.0))
                # Wall time is a proxy for demo time (the listen server runs at 1x); segment uses the real tick counts
                # downstream, so a small drift here only shifts how many chunks --hours produces.
                end = proc.last_progress if verdict == "hung" else self.h.clock()
                if verdict != "running":
                    raise ConnectionError(f"CS2 {verdict} while recording {demo_id}")
                client.send(stop_cmd)
                active = None
                if self._wait_closed(src):
                    self._publish(src, demo_id, started, end - t0, truncated=False)
                else:
                    log.warning("%s: demo missing, empty or still growing after stop; kept for a later cycle", demo_id)
        except (ConnectionError, _RecordingFailed) as exc:
            log.warning("netcon session ended: %s", exc)
            if active is not None:
                end = self.h.clock() if end is None else end
            proc.kill()
            if active is not None and end is not None:
                demo_id, started, t0 = active
                self._salvage(self.csgo_dir / f"{demo_id}.dem", demo_id, started, max(end - t0, 0.0))

    def _leftover_demos(self) -> list[Path]:
        """Demos this tool recorded (``<kind>_<stamp>_<nnn>.dem``) that are still in the csgo dir."""
        return sorted(p for p in self.csgo_dir.glob("*.dem") if self._ours.match(p.name))

    def _session(self) -> None:
        cmd = build_launch_command(self.cfg, self.profile, int(self.g.netconport))
        with CS2Process(
            cmd,
            self.csgo_dir,
            float(self.g.hang_timeout_min),
            popen=self.h.popen,
            clock=self.h.clock,
            size_probe=self.h.dir_size,
            run=self.h.run,
        ) as proc:
            self._session_netcon(proc)

    # -- entry point ------------------------------------------------------------------------------------------------

    def _sweep_stale(self) -> None:
        """Delete leftovers of an interrupted publish: ``*.part`` files and sidecars without a ``.dem``."""
        for directory in (self.out_dir, self.rejected_dir):
            if not directory.is_dir():
                continue
            for path in [*directory.glob("*.dem.part"), *directory.glob("*.json.part")]:
                log.warning("removing stale %s", path)
                self._unlink_quiet(path)
        if self.out_dir.is_dir():
            for sidecar in self.out_dir.glob("*.json"):
                if not sidecar.with_suffix(".dem").exists():
                    log.warning("removing orphan sidecar %s (no .dem)", sidecar)
                    self._unlink_quiet(sidecar)

    def _fresh_start(self) -> None:
        """Sweep stale files, then move ``self.started`` forward until no earlier demo has the same id prefix."""
        self._sweep_stale()
        while any(
            list(d.glob(f"{self.profile}_{self.started:%Y%m%d-%H%M%S}_*"))
            for d in (self.out_dir, self.rejected_dir, self.csgo_dir)
            if d.is_dir()
        ):
            self.started += timedelta(seconds=1)

    def run(self) -> list[Path]:
        """Generate until ``hours`` of demo time are kept.

        Returns:
            Paths of the kept demos.

        Raises:
            PatchGuardError: If CS2 was patched. Demos not yet published are left in the csgo dir and logged.
            GenerateError: If CS2 failed ``generate.max_failures`` sessions in a row without a usable chunk, or demos
                are still unpublished at the end (one summary ntfy alert with the count is sent first).
        """
        rendered = render_cfg(self.profile, self.cfg, self.cfg_dir)
        _, self.cfg_sha = install_cfg(rendered, self.profile, self.cs2_dir)
        self.git_sha = self.h.git_sha()
        self._fresh_start()
        failures = 0
        try:
            while self.generated_s < self.target_s and self._publish_failures < self.g.max_failures:
                self._guard()
                self._flush_unlinks()
                self._adopt_leftovers()
                self._progress = 0
                self._session()
                if self.generated_s >= self.target_s or self._publish_failures >= self.g.max_failures:
                    break
                failures = 0 if self._progress else failures + 1
                log.warning(
                    "CS2 session ended after %d usable chunk(s); restarting (%d failure(s) in a row)",
                    self._progress,
                    failures,
                )
                if failures >= self.g.max_failures:
                    msg = f"CS2 failed {failures} sessions in a row without a usable chunk; giving up."
                    self.h.notify(msg, title="dcs2: generate failed", priority="high", tags=["rotating_light"])
                    raise GenerateError(msg)
        except PatchGuardError:
            left = [p.name for p in self._leftover_demos()]
            if left:
                log.error("stopped by the patch guard with unpublished demos in %s: %s", self.csgo_dir, left)
            raise
        finally:
            self._flush_unlinks()
        self._adopt_leftovers()
        self._flush_unlinks()
        left = sorted(p for p in self._unpublished if p.exists())
        if left:
            msg = f"{len(left)} demo(s) could not be published and remain in {self.csgo_dir} (first: {left[0].name})."
            self.h.notify(msg, title="dcs2: generate failed", priority="high", tags=["rotating_light"])
            raise GenerateError(msg)
        return self.kept


def run(cfg: DictConfig, profile: str, hours: float, chunk_min: float | None = None) -> list[Path]:
    """CLI stage: generate demos for ``hours`` hours.

    Args:
        cfg: A validated cs2wm config.
        profile: ``dm``, ``smoke`` or ``human``.
        hours: Demo time to record. Restarts after a crash or hang continue until this much was kept.
        chunk_min: Chunk length override (``generate.chunk_min``).

    Returns:
        Paths of the kept demos. A truncated last chunk is kept only if ``cs2wm inspect-demo`` parses it.
    """
    hooks = Hooks()
    hooks.notify = partial(ntfy.notify, server=cfg.supervisor.ntfy_server, topic=cfg.supervisor.ntfy_topic)
    return Generator(cfg, profile, hours, chunk_min, hooks).run()
