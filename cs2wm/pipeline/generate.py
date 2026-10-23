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
* ``generate.chunking = netcon``: ``record <demo_id>`` / ``stop`` over ``-netconport`` every chunk. ``gotv``:
  ``tv_autorecord`` with ``mp_timelimit`` and every new demo in the csgo dir is picked up once it is closed.
* CS2 writes into its own ``game/csgo`` dir. A closed chunk is copied next to its destination and renamed into
  ``paths.demos/<kind>/`` atomically (sidecar first, so a ``.dem`` is never visible without it).
* A watchdog restarts CS2 on exit or when no demo file grew for ``generate.hang_timeout_min``. A truncated chunk is
  kept only if ``cs2wm.demo.inspect`` parses it, else it goes to ``demos/_rejected/`` with a reason file.
* The patch guard compares the installed CS2 buildid with ``generate.expected_build`` before each launch and chunk.

Everything marked ``CONFIRM(#14)`` is an assumption about CS2 that the manual debug session must confirm.
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
from cs2wm.pipeline.netcon import NetconClient

log = logging.getLogger(__name__)

REJECTED_DIR = "_rejected"
_EXEC_RE = re.compile(r"^\s*exec\s+(generate_\w+)\s*(?://.*)?$")


class PatchGuardError(RuntimeError):
    """The installed CS2 build differs from ``generate.expected_build``; generation stopped."""


class GenerateError(RuntimeError):
    """Generation gave up (CS2 would not stay up)."""


# ---------------------------------------------------------------------------------------------------------------------
# Launch command


def _require_insecure(cmd: Sequence[str]) -> None:
    """Raise ``ValueError`` unless ``-insecure`` is in ``cmd``, or if it asks CS2 to connect to a server or go secure."""
    problem = cs2wm_config.launch_option_problem(list(cmd))
    if problem:
        raise ValueError(f"refusing to start CS2: {problem} (HLAE and CS2 only ever run offline)")


def build_launch_command(cfg: DictConfig, profile: str, netconport: int | None = None) -> list[str]:
    """Build the CS2 launch command line from ``generate.launch_options``.

    Args:
        cfg: A validated cs2wm config.
        profile: ``dm``, ``smoke`` or ``human``; selects ``config/cs2/generate_<profile>.cfg``.
        netconport: Optional ``-netconport`` for option (B) chunking.

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
    # CONFIRM(#14): `+exec` on the command line runs the cfg after startup, before/while the map in it loads.
    cmd += ["+exec", f"generate_{profile}"]
    return cmd


# ---------------------------------------------------------------------------------------------------------------------
# Console cfg


def render_cfg(profile: str, cfg: DictConfig, cfg_dir: Path | None = None) -> str:
    """Render ``generate_<profile>.cfg`` into the single file that is installed into CS2.

    ``exec generate_*`` lines are replaced by the included file (``generate_smoke`` builds on ``generate_dm``). With
    ``generate.chunking = gotv`` the GOTV autorecord commands are appended.

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
    cfg_dir = cfg_dir or cs2wm_config.DEFAULT_CONFIG_PATH.parent / "cs2"

    def expand(name: str, stack: tuple[str, ...]) -> list[str]:
        if name in stack:
            raise ValueError(f"exec cycle in cfgs: {' -> '.join((*stack, name))}")
        out = [f"// ---- {name}.cfg"]
        for line in (cfg_dir / f"{name}.cfg").read_text(encoding="utf-8").splitlines():
            m = _EXEC_RE.match(line)
            out += expand(m.group(1), (*stack, name)) if m else [line]
        return out

    lines = expand(f"generate_{profile}", ())
    if cfg.generate.chunking == "gotv":
        # CONFIRM(#14): one demo per map cycle; GOTV may need `map` again after `tv_enable`; mp_timelimit takes minutes.
        lines += [
            "// ---- chunking: gotv (generate.chunking)",
            "tv_enable 1",
            "tv_autorecord 1",
            f"mp_timelimit {cfg.generate.chunk_min:g}",
        ]
    return "\n".join(lines) + "\n"


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
        ``cs2.exe`` may still be alive (CONFIRM(#14)).
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
    """Why ``path`` is not a parsable demo, or None if it parses (the single seam to ``cs2wm.demo``).

    ``demo.inspect`` reports failures as ``ok: False`` plus ``error`` and ``demo.is_parseable`` is the boolean form;
    either may be missing on older bases, and any exception also counts as unparsable.
    """
    try:
        is_parseable = getattr(demo, "is_parseable", None)
        if is_parseable is not None:
            return None if is_parseable(path) else "demo.is_parseable() is False"
        info = demo.inspect(path)
        if not info.get("ok", info.get("last_tick") is not None):
            return str(info.get("error") or "no ticks parsed")
        return None
    except Exception as exc:  # noqa: BLE001 - any parser failure means the chunk is unusable
        return repr(exc)


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
    notify: Callable[..., None] = ntfy.notify
    git_sha: Callable[[], str] = _git_sha
    copy: Callable[[Path, Path], Any] = shutil.copyfile


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
        self.generated_s = 0.0  # demo time of kept chunks (plus intact chunks that could not be published yet)
        self.cfg_sha = ""
        self.git_sha = ""
        self.started = self.h.wall()
        self._index = 0  # chunk counter, continues across CS2 restarts
        self._progress = 0  # chunks kept or closed normally in the current session; rejected ones do not count
        self._ignore: set[Path] = set()  # gotv: demos that were present before the run or are published/rejected
        self._stranded: set[Path] = set()  # gotv: demos whose publishing failed; retried in the next session
        self._pending_unlink: list[Path] = []  # copied sources CS2 still holds open; deleted again at the end
        self._notified: set[Path] = set()
        kinds = "|".join(re.escape(str(k)) for k in cfg.episodes.kinds)
        self._ours = re.compile(rf"^(?:{kinds})_\d{{8}}-\d{{6}}_\d{{3}}\.dem$")

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
        self, demo_id: str, started: datetime, duration_s: float, truncated: bool, nbytes: int, extra: dict
    ) -> dict:
        return {
            "demo_id": demo_id,
            "kind": self.profile,
            "profile": self.profile,
            "start_time": started.isoformat(timespec="seconds"),
            "chunk_min": self.chunk_min,
            "duration_s": round(duration_s, 1),
            "chunking": str(self.g.chunking),
            "truncated": truncated,
            "bytes": nbytes,
            "cs2_build": self._build(),
            "cfg_sha256": self.cfg_sha,
            "cs2wm_git_sha": self.git_sha,
            **extra,
        }

    def _copy(self, src: Path, dest: Path) -> bool:
        """Copy ``src`` to ``dest``, retrying with backoff for up to ``close_timeout_s``.

        CS2 can hold a demo open for a while after its size stopped changing; on Windows copying then raises
        ``PermissionError``.
        """
        start, delay = self.h.clock(), 0.5
        while True:
            try:
                self.h.copy(src, dest)
                return True
            except OSError as exc:
                if self.h.clock() - start >= self.g.close_timeout_s:
                    log.error("giving up copying %s: %s", src, exc)
                    dest.unlink(missing_ok=True)
                    return False
                log.debug("copy of %s failed (%s), retry in %.1f s", src, exc, delay)
                self.h.sleep(delay)
                delay = min(delay * 2, 10.0)

    def _remove_source(self, src: Path) -> None:
        try:
            src.unlink()
        except OSError as exc:
            log.warning("could not delete %s after copying (retried at the end of the run): %s", src, exc)
            self._pending_unlink.append(src)

    def _unpublished(self, src: Path, what: str) -> bool:
        """Log and notify (once per file) that ``src`` could not be moved; the source stays where it is."""
        msg = f"{what} failed for {src.name}; the file is kept in {src.parent} and retried later"
        log.error(msg)
        if src not in self._notified:
            self._notified.add(src)
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
        started: datetime,
        duration_s: float,
        truncated: bool,
        extra: dict | None = None,
    ) -> bool:
        """Keep a closed chunk, or reject a truncated one that does not parse.

        A failure to copy (file locks) is logged and notified but never raised; ``src`` stays in place.

        Returns:
            True if the chunk was kept or rejected (``src`` is dealt with), False if publishing failed.
        """
        if truncated:
            error = self.h.parse_error(src)
            if error is not None:
                return self._reject(src, demo_id, f"truncated chunk does not parse: {error}")
        self._progress += 1
        demo_id = self._unique_id(demo_id)
        dest = self.out_dir / f"{demo_id}.dem"
        part = dest.with_name(dest.name + ".part")
        sidecar = self.out_dir / f"{demo_id}.json"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        nbytes = self.h.file_size(src) or 0
        if not self._copy(src, part):
            if not truncated:  # an intact chunk waits in the csgo dir and still counts against the time budget
                self.generated_s += duration_s
            return self._unpublished(src, "copying")
        try:
            # sidecar first, then the .dem: a .dem is never visible without its sidecar
            _write_json_atomic(sidecar, self._sidecar(demo_id, started, duration_s, truncated, nbytes, extra or {}))
            os.replace(part, dest)
        except OSError as exc:
            sidecar.unlink(missing_ok=True)
            part.unlink(missing_ok=True)
            log.error("renaming %s into place failed: %s", demo_id, exc)
            return self._unpublished(src, "renaming")
        self._remove_source(src)
        self.kept.append(dest)
        self.generated_s += duration_s
        log.info("kept %s (%.0f s, %d bytes%s)", dest, duration_s, nbytes, ", truncated" if truncated else "")
        return True

    def _reject(self, src: Path, demo_id: str, reason: str) -> bool:
        demo_id = self._unique_id(demo_id)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        dest = self.rejected_dir / f"{demo_id}.dem"
        part = dest.with_name(dest.name + ".part")
        if not self._copy(src, part):
            return self._unpublished(src, "copying a rejected chunk")
        os.replace(part, dest)
        (self.rejected_dir / f"{demo_id}.reason.txt").write_text(reason + "\n", encoding="utf-8")
        self._remove_source(src)
        log.warning("rejected %s: %s", demo_id, reason)
        return True

    def _salvage(self, src: Path, demo_id: str, started: datetime, duration_s: float) -> None:
        """Handle the chunk that was being recorded when CS2 died or hung."""
        if self._wait_closed(src):
            self._publish(src, demo_id, started, duration_s, truncated=True)
        else:
            log.warning("%s: no demo data to salvage (missing, empty or still growing)", demo_id)

    def _adopt_leftovers(self) -> None:
        """Publish demos this tool recorded earlier but could not move (``<kind>_<stamp>_<nnn>.dem`` in the csgo dir).

        Called while no CS2 of ours is running. Each one is parse-checked like a truncated chunk (kept or rejected)
        and carries ``"adopted": true`` with ``duration_s`` 0, since its length is unknown; it does not count toward
        ``--hours``. GOTV's own ``auto*.dem`` names are never adopted: demos of other runs there stay untouched.
        """
        for path in sorted(self.csgo_dir.glob("*.dem")):
            if not self._ours.match(path.name):
                continue
            mtime = datetime.fromtimestamp(path.stat().st_mtime) if path.exists() else self.h.wall()
            log.info("adopting leftover demo %s", path)
            self._publish(path, path.stem, mtime, 0.0, truncated=True, extra={"adopted": True})

    def _flush_unlinks(self) -> None:
        pending, self._pending_unlink = self._pending_unlink, []
        for path in pending:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                log.error("%s is a published demo still left in the csgo dir (delete it by hand): %s", path, exc)

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

    def _session_netcon(self, proc: CS2Process) -> None:
        """Record chunks with ``record``/``stop`` over netcon until the target or a failure."""
        client = self.h.netcon(port=int(self.g.netconport), clock=self.h.clock, sleep=self.h.sleep)
        active: tuple[str, datetime, float] | None = None
        end: float | None = None  # when recording really ended: before the close wait, and before a hang's idle tail
        try:
            client.wait_ready(float(self.g.netcon_connect_timeout_s))
            if self._watch(proc, float(self.g.map_load_wait_s)) != "running":
                raise ConnectionError("CS2 died while loading")
            while self.generated_s < self.target_s:
                self._guard()
                demo_id = make_demo_id(self.profile, self.started, self._index)
                self._index += 1
                started, t0 = self.h.wall(), self.h.clock()
                length_s = self._next_chunk_s()
                # CONFIRM(#14): `record` writes <csgo>/<name>.dem; a name with '-' needs no quotes; `stop` closes it.
                client.send(f"record {demo_id}")
                proc.rearm()
                active, end = (demo_id, started, t0), None
                verdict = self._watch(proc, length_s)
                # CONFIRM(#14): wall time is a proxy for demo time (the listen server runs at 1x); segment uses the
                # real tick counts downstream, so a small drift here only shifts how many chunks --hours produces.
                end = proc.last_progress if verdict == "hung" else self.h.clock()
                if verdict != "running":
                    raise ConnectionError(f"CS2 {verdict} while recording {demo_id}")
                client.send("stop")
                active = None
                src = self.csgo_dir / f"{demo_id}.dem"
                if self._wait_closed(src):
                    self._publish(src, demo_id, started, end - t0, truncated=False)
                else:
                    log.warning("%s: demo missing, empty or still growing after stop; kept for a later cycle", demo_id)
        except ConnectionError as exc:
            log.warning("netcon session ended: %s", exc)
            if active is not None:
                end = self.h.clock() if end is None else end
            proc.kill()
            if active is not None and end is not None:
                demo_id, started, t0 = active
                self._salvage(self.csgo_dir / f"{demo_id}.dem", demo_id, started, max(end - t0, 0.0))

    def _new_demos(self) -> list[Path]:
        skip = self._ignore | self._stranded
        return sorted(p for p in self.csgo_dir.glob("*.dem") if p not in skip and not self._ours.match(p.name))

    def _session_gotv(self, proc: CS2Process) -> None:
        """Pick up GOTV autorecord demos as they close, until the target or a failure."""
        seen: dict[Path, list[float]] = {}  # path -> [size, first seen, last change]
        while True:
            self.h.sleep(float(self.g.poll_s))
            verdict = proc.poll()
            now = self.h.clock()
            files = self._new_demos()
            for p in files:
                size = self.h.file_size(p) or 0
                rec = seen.setdefault(p, [size, now, now])
                if size != rec[0]:
                    rec[0], rec[2] = size, now
            # CONFIRM(#14): a demo is closed once a newer one exists and it stopped growing.
            for p in files[:-1]:
                if now - seen[p][2] >= self.g.close_stable_s:
                    self._finish_gotv(p, seen.pop(p), truncated=False)
            active = seen[files[-1]] if files and files[-1] in seen else None
            over = active is not None and self.generated_s + (active[2] - active[1]) >= self.target_s
            if verdict != "running" or over:
                proc.kill()
                for p in self._new_demos():
                    if self._wait_closed(p):
                        self._finish_gotv(p, seen.pop(p, [0, now, now]), truncated=True)
                return
            if self.generated_s >= self.target_s:
                return

    def _finish_gotv(self, path: Path, rec: list[float], truncated: bool) -> None:
        demo_id = make_demo_id(self.profile, self.started, self._index)
        # CONFIRM(#14): the start time is approximated by when the demo file was first seen.
        started = self.h.wall() - timedelta(seconds=self.h.clock() - rec[1])
        if self._publish(path, demo_id, started, max(rec[2] - rec[1], 0.0), truncated):
            self._index += 1
            self._ignore.add(path)  # only handled demos are ignored; a failed one is retried in the next session
        else:
            self._stranded.add(path)
        if not truncated:
            self._guard()  # the next map cycle has already started; stop it by leaving the session

    def _session(self) -> None:
        netcon = self.g.chunking == "netcon"
        cmd = build_launch_command(self.cfg, self.profile, int(self.g.netconport) if netcon else None)
        self._stranded.clear()
        with CS2Process(
            cmd,
            self.csgo_dir,
            float(self.g.hang_timeout_min),
            popen=self.h.popen,
            clock=self.h.clock,
            size_probe=self.h.dir_size,
            run=self.h.run,
        ) as proc:
            if netcon:
                self._session_netcon(proc)
            else:
                self._session_gotv(proc)

    # -- entry point ------------------------------------------------------------------------------------------------

    def _fresh_start(self) -> None:
        """Move ``self.started`` forward until no demo of an earlier run in this second has the same id prefix."""
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
            GenerateError: If CS2 failed ``generate.max_failures`` sessions in a row without a kept chunk.
        """
        rendered = render_cfg(self.profile, self.cfg, self.cfg_dir)
        _, self.cfg_sha = install_cfg(rendered, self.profile, self.cs2_dir)
        self.git_sha = self.h.git_sha()
        self._fresh_start()
        self._ignore = set(self.csgo_dir.glob("*.dem"))  # demos of other tools and runs are not ours
        failures = 0
        try:
            while self.generated_s < self.target_s:
                self._guard()
                self._adopt_leftovers()
                self._progress = 0
                self._session()
                if self.generated_s >= self.target_s:
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
            left = [p.name for p in self._new_demos()]
            if left:
                log.error("stopped by the patch guard with unpublished demos in %s: %s", self.csgo_dir, left)
            raise
        finally:
            self._flush_unlinks()
        self._adopt_leftovers()
        self._flush_unlinks()
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
