"""supervisor: runs stages 1-6 unattended for each demo, as one Windows scheduled task (#20).

Per demo: generate -> segment -> render -> extract -> pack -> sync, with up to ``supervisor.max_attempts`` attempts
and exponential backoff per stage, a disk guard (``supervisor.max_working_gb``), a CS2 patch guard (render a 10 s
smoke test when the build id changes), and ntfy start/hourly/alert/morning messages. State is persisted atomically in
``state/pipeline.json`` so a reboot resumes where it stopped.

The night (``run_until``): repeat "generate one block (``supervisor.generate_block_min``), then advance every pending
demo through segment -> render -> extract -> pack, then sync" until the deadline or the target. Only one CS2 can run
at a time, so generation and rendering are interleaved, never parallel. Demos go through the stages one at a time
(depth first), so frames and raw captures are deleted by pack before the next demo starts.

* Deadline: ``--until HH:MM`` is the next such time (23:00 -> 07:30 means tomorrow). Generation stops
  ``supervisor.drain_min`` early so the pipeline can drain; no new demo is started after the deadline, but what is
  packed is synced once more before the summary.
* Retries: a stage gets ``supervisor.max_attempts`` attempts per demo (counted in the demo's ``attempts``), with
  ``backoff_s * 2^(n-1)`` between them and CS2/HLAE killed after a failed render. After the last one the demo is
  ``failed`` and the night goes on.
* Disk guard: generation pauses while the working set exceeds ``max_working_gb`` and stops (with an alert) when the
  drive has less than ``min_free_gb`` free. The demo archive is not counted.
* Patch guard: when the CS2 ``buildid`` differs from the one in ``state.meta``, a ~10 s render smoke test runs before
  anything else (see ``Supervisor.render_smoke_test``). If it fails, generation and rendering stop. Generate's own
  ``PatchGuardError`` stops generation too; what is already generated is still drained.
* ntfy: start, hourly heartbeat, immediate alerts and the morning summary (``ntfy.format_*``).

State is persisted atomically in ``state/pipeline.json`` so a reboot resumes where it stopped. ``Supervisor.run_stage``,
``run_sync`` and ``status_table`` (#19) are also what ``cs2wm <stage> --all-pending``, ``cs2wm sync`` and
``cs2wm status`` call.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from cs2wm import demo as demo_utils
from cs2wm.config import SAFE_NAME_RE
from cs2wm.pipeline import generate, ntfy, proc

log = logging.getLogger(__name__)

STATE_SCHEMA_VERSION = 1
EXIT_ALERT = 3  # exit code of `cs2wm supervise` when the night ended with a guard trip or an alert
WORKING_SET_DIRS = ("frames", "raw", "labels", "hdf5", "hdf5_fullres")  # paths.* counted by the disk guard
BUILD_KEY = "cs2_build_id"  # PipelineState.meta: the last CS2 buildid that rendered fine
REJECTED_DIR = "_rejected"  # demos/_rejected/ holds demos that generate could not parse (see generate.REJECTED_DIR)


class StageState(enum.Enum):
    """Per-demo pipeline state: pending -> segmented -> rendered -> extracted -> packed -> synced | failed."""

    PENDING = "pending"
    SEGMENTED = "segmented"
    RENDERED = "rendered"
    EXTRACTED = "extracted"
    PACKED = "packed"
    SYNCED = "synced"
    FAILED = "failed"


# Per-demo stages in pipeline order, the state a demo must be in to run the stage, and the state it reaches.
STAGES = ("segment", "render", "extract", "pack", "sync")
_INPUT_STATE = {
    "segment": StageState.PENDING,
    "render": StageState.SEGMENTED,
    "extract": StageState.RENDERED,
    "pack": StageState.EXTRACTED,
    "sync": StageState.PACKED,
}
_RENDER_OK = ("rendered", "extracted", "packed", "synced")  # demos whose render succeeded: known-good smoke inputs
_ADVANCEABLE = ("pending", "segmented", "rendered", "extracted")  # states a demo is advanced from by the loop
_OUTPUT_STATE = {
    "segment": StageState.SEGMENTED,
    "render": StageState.RENDERED,
    "extract": StageState.EXTRACTED,
    "pack": StageState.PACKED,
    "sync": StageState.SYNCED,
}


class StateError(ValueError):
    """The state file cannot be read; the message names the file and the backup to restore it from."""


class StageFailure(RuntimeError):
    """A stage ran but did not produce its output (the message is recorded as the failure reason).

    Args:
        message: The failure reason.
        attempts: Launches the stage already used internally (render retries a crash or timeout itself), so the
            supervisor can count them against ``supervisor.max_attempts``.
        timed_out: The stage hit its timeout; it is not retried again (it had its internal retry already).
    """

    def __init__(self, message: str, attempts: int = 1, timed_out: bool = False) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.timed_out = timed_out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class PipelineState:
    """Persistent supervisor state.

    Each demo has a record with ``state`` (a ``StageState`` value), ``kind``, ``failed_stage`` and ``reason`` (set
    while ``failed``), ``stages`` (``{stage: {"wall_s": float, "at": ISO time}}`` of the last run of each stage),
    ``attempts`` (``{stage: n}`` failed attempts of the supervisor, cleared when the stage succeeds), ``updated`` and whatever ``mark`` was given as ``info`` (``demo_s``, ``frames``, ...).

    Attributes:
        path: ``state/pipeline.json``.
        demos: Record per demo id.
        meta: Supervisor-wide values (``cs2_build_id``: the last CS2 build that passed the render smoke test).
    """

    path: Path
    demos: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> PipelineState:
        """Load the state file (empty state if it does not exist).

        Raises:
            StateError: If the file is corrupt, truncated or from another schema. Saves are atomic, so this means
                it was edited or the disk failed; the previous state is kept in ``<name>.bak``.
        """
        path = Path(path)
        hint = f"restore it from {path.name}.bak"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls(path)
        except (ValueError, OSError) as exc:
            raise StateError(f"{path} is corrupt or unreadable ({exc}); {hint}") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != STATE_SCHEMA_VERSION:
            raise StateError(f"{path} has an unsupported pipeline state schema; {hint}")
        return cls(path, dict(raw.get("demos", {})), dict(raw.get("meta", {})))

    def save(self) -> None:
        """Write the state atomically (temp file + fsync + rename), keeping the previous file as ``<name>.bak``."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                payload = {"schema_version": STATE_SCHEMA_VERSION, "demos": self.demos, "meta": self.meta}
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            if self.path.exists():
                shutil.copyfile(self.path, self.path.with_name(self.path.name + ".bak"))
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def state_of(self, demo_id: str) -> StageState | None:
        """State of a demo, or None if it is unknown."""
        rec = self.demos.get(demo_id)
        return StageState(rec["state"]) if rec else None

    def mark(
        self,
        demo_id: str,
        state: StageState | str,
        reason: str | None = None,
        stage_wall_s: float | None = None,
        stage: str | None = None,
        **info: Any,
    ) -> None:
        """Record a transition. Does not save; callers save after each demo so a crash loses at most one stage.

        Args:
            demo_id: Demo id (created as ``pending`` if unknown).
            state: New state. ``failed`` needs ``stage`` and ``reason``; any other state clears an earlier failure.
            reason: Why the demo failed.
            stage_wall_s: Wall time of the stage that ended in this transition.
            stage: The stage that ran; implied by every state except ``failed``.
            **info: Extra fields stored on the record (``kind``, ``demo_s``, ``frames``, ...).

        Raises:
            ValueError: If ``failed`` is marked without a stage and a reason, or ``stage`` is not a stage.
        """
        new = state if isinstance(state, StageState) else StageState(state)
        if new is StageState.FAILED:
            if not stage or not reason:
                raise ValueError("a failed demo needs the failing stage and a reason")
        elif new is not StageState.PENDING:
            stage = next(s for s, out in _OUTPUT_STATE.items() if out is new)
        if stage is not None and stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        rec = self.demos.setdefault(demo_id, {"state": StageState.PENDING.value, "stages": {}})
        rec.update(info)
        rec["state"] = new.value
        rec["updated"] = _now()
        if new is StageState.FAILED:
            rec["failed_stage"], rec["reason"] = stage, reason
        else:
            rec.pop("failed_stage", None)
            rec.pop("reason", None)
        if stage is not None and stage_wall_s is not None:
            rec["stages"][stage] = {"wall_s": round(float(stage_wall_s), 3), "at": rec["updated"]}

    def discover(self, cfg: DictConfig) -> list[str]:
        """Add every ``demos/<kind>/*.dem`` that the state does not know yet as ``pending``.

        Only the folders named after an episode kind are looked at, so ``_rejected`` and anything else is ignored.

        Returns:
            The newly added demo ids, sorted.
        """
        added: list[str] = []
        base = Path(cfg.paths.demos)
        for kind in cfg.episodes.kinds:
            if kind == REJECTED_DIR or not (base / kind).is_dir():
                continue
            for dem in sorted((base / kind).glob("*.dem")):
                if dem.stem not in self.demos:
                    self.mark(dem.stem, StageState.PENDING, kind=kind)
                    added.append(dem.stem)
        return sorted(added)

    def pending_for(self, stage: str, retry_failed: bool = False) -> list[str]:
        """Demo ids ready for ``stage``, sorted by id (demo ids start with their start time).

        Args:
            stage: One of ``STAGES``.
            retry_failed: Also return demos that failed in this stage. Failed demos are never retried otherwise.
        """
        want = _INPUT_STATE[stage].value
        return sorted(
            d
            for d, rec in self.demos.items()
            if rec["state"] == want or (retry_failed and rec["state"] == "failed" and rec.get("failed_stage") == stage)
        )


@dataclass
class StageReport:
    """Outcome of ``Supervisor.run_stage``.

    Attributes:
        done: Demos that finished the stage.
        failed: Failed demos with their reasons.
        outputs: Path written by each finished demo, for the stages that name one (segment: the plan).
        attempts: Launches a failed demo used inside the stage (default 1).
        timed_out: Failed demos whose stage hit its timeout.
    """

    done: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    timed_out: set[str] = field(default_factory=set)


def resolve_deadline(now: datetime, until: str) -> datetime:
    """The next local time ``HH:MM`` after ``now``: ``07:30`` started at 23:00 is tomorrow 07:30.

    Args:
        now: Current local time.
        until: ``HH:MM``.

    Returns:
        The deadline, strictly after ``now`` and at most 24 h away.

    Raises:
        ValueError: If ``until`` is not a valid ``HH:MM``.
    """
    try:
        hour, minute = (int(part) for part in until.split(":"))
        deadline = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    except ValueError as exc:
        raise ValueError(f"--until must be a time as HH:MM (24 h), got {until!r}") from exc
    return deadline if deadline > now else deadline + timedelta(days=1)


def _dir_bytes(path: Path) -> int:
    """Total size of the files under ``path`` (0 if missing; files that vanish while walking are skipped)."""
    total = 0
    for dirpath, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                continue
    return total


def demo_leftovers(cfg: DictConfig, demo_id: str) -> list[Path]:
    """The working-set directories of one demo that exist: frames, raw, labels and pack's ``.staging``/``.old`` dirs.

    Never the demo, its plan or its QA file. A demo id that is not a plain file name yields nothing.
    """
    if not SAFE_NAME_RE.match(demo_id):
        return []
    p = cfg.paths
    paths = [Path(p.frames) / demo_id, Path(p.frames) / f"{demo_id}.deleting", Path(p.raw) / demo_id]
    paths += [Path(p.labels) / demo_id, Path(p.labels) / f"{demo_id}.deleting"]
    for base in (p.hdf5, p.hdf5_fullres):
        for kind in cfg.episodes.kinds:
            paths += [Path(base) / kind / f"{demo_id}{suffix}" for suffix in (".staging", ".old")]
    return [path for path in paths if path.exists()]


def failed_leftovers(cfg: DictConfig, demo_ids: Iterable[str]) -> list[tuple[str, Path, int]]:
    """``(demo id, directory, bytes)`` for the working-set leftovers of the given demos."""
    return [(d, path, _dir_bytes(path)) for d in sorted(demo_ids) for path in demo_leftovers(cfg, d)]


def cleanup_failed(cfg: DictConfig, state: PipelineState, dry_run: bool = False) -> list[tuple[str, Path, int]]:
    """Delete the working-set directories of ``failed`` demos (``cs2wm cleanup --failed``).

    Demos, plans, QA files and the state are never touched. A failed demo that is re-run later must be re-rendered.

    Args:
        cfg: A validated cs2wm config.
        state: The pipeline state.
        dry_run: Only report what would be deleted.

    Returns:
        What was (or would be) deleted, as from ``failed_leftovers``.
    """
    failed = [d for d, rec in state.demos.items() if rec["state"] == StageState.FAILED.value]
    found = failed_leftovers(cfg, failed)
    if not dry_run:
        for _, path, _ in found:
            shutil.rmtree(path, ignore_errors=True)
    return found


def measure_disk(cfg: DictConfig, exclude: Iterable[str] = ()) -> tuple[float, float]:
    """Measure the disk guard's inputs.

    The working set is ``frames``, ``raw``, ``labels``, ``hdf5`` and ``hdf5_fullres`` (the pack ``.staging`` dirs live
    inside them) but not the demo archive, which is never deleted. Free space is that of the drive holding ``paths.root``.

    Args:
        cfg: A validated cs2wm config.
        exclude: Demo ids whose leftovers (``demo_leftovers``) are not counted: failed demos never free them, so
            they must not keep the guard tripped (``cs2wm cleanup --failed`` deletes them).

    Returns:
        ``(working set GB, free GB)``.
    """
    working = sum(_dir_bytes(Path(cfg.paths[name])) for name in WORKING_SET_DIRS)
    working -= sum(size for _, _, size in failed_leftovers(cfg, exclude))
    probe = Path(cfg.paths.root)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return working / 1e9, shutil.disk_usage(probe).free / 1e9


def make_smoke_plan(plan: dict[str, Any], seconds: float) -> dict[str, Any]:
    """Cut a short render plan out of a real one: its first episode, truncated to ``seconds`` of footage.

    The result is still a valid plan of the same demo (same target, same spectate command, same start tick), so a
    render of it exercises exactly what a real render does, in a few seconds of footage.

    Args:
        plan: A parsed ``render_plan.json``.
        seconds: Footage to render (``timing.fps * seconds`` frames, fewer if the episode is shorter).

    Returns:
        A new plan dict (the input is not modified).

    Raises:
        ValueError: If the plan has no episode to cut from.
    """
    fps, tpf = plan["timing"]["fps"], plan["timing"]["ticks_per_frame"]
    segments = sorted((s for s in plan["segments"] if s["episodes"]), key=lambda s: s["segment_idx"])
    if not segments:
        raise ValueError(f"{plan['demo_id']}: the plan has no episode")
    seg = json.loads(json.dumps(segments[0]))
    episode = seg["episodes"][0]
    frames = max(1, min(round(seconds * fps), episode["n_frames"]))
    start = seg["record_start_tick"] + episode["first_frame"] * tpf
    end = start + frames * tpf
    offset = plan["tick_offset"]
    episode.update(first_frame=0, n_frames=frames, rendered_frames=frames + 1, start_tick=start, end_tick=end)
    seg.update(
        episodes=[episode],
        record_start_tick=start,
        record_end_tick=end,
        mirv_record_start_tick=start + offset,
        mirv_record_end_tick=end + offset,
        labeled_frames=frames,
        rendered_frames=frames + 1,
    )
    seg["mirv_spec_tick"] = min(seg["mirv_spec_tick"], seg["mirv_record_start_tick"])
    schedule = []
    for entry in plan["mirv_cmd_schedule"]:
        if entry["segment_idx"] != seg["segment_idx"]:
            continue
        entry = dict(entry)
        if entry["action"] == "record_start":
            entry.update(mirv_tick=start + offset, demo_tick=start)
        elif entry["action"] == "record_end":
            entry.update(mirv_tick=end + offset, demo_tick=end)
        schedule.append(entry)
    schedule.sort(key=lambda e: e["mirv_tick"])
    totals = {"segments": 1, "episodes": 1, "labeled_frames": frames, "rendered_frames": frames + 1}
    return {
        **plan,
        "demo_end_tick": end,
        "segments": [seg],
        "mirv_cmd_schedule": schedule,
        "totals": totals,
        "meta": {**plan.get("meta", {}), "smoke_test": True},
    }


@dataclass
class Hooks:
    """Everything the loop takes from the outside world, replaced by fakes in tests.

    Attributes:
        now: Local wall clock.
        sleep: Sleep, used for the backoff.
        notify: ``ntfy.notify``-like; default posts to ``supervisor.ntfy_server``/``ntfy_topic``.
        disk: ``(cfg, failed demo ids) -> (working set GB, free GB)``; default ``measure_disk``.
        build_id: ``cs2_dir -> CS2 buildid``.
        generate: ``(cfg, profile, hours) -> demos``; default ``generate.run``.
        sync: ``() -> summary with .ok``; default ``Supervisor.run_sync``.
        smoke: ``() -> failure reason, or None if the render smoke test passed``; default
            ``Supervisor.render_smoke_test``.
        kill_cs2: Kills stray CS2/HLAE processes; default ``proc.kill_by_name``.
        backoff: ``failed attempt number -> seconds to wait``; default ``supervisor.backoff_s * 2^(n-1)``.
        stage_fns: Replacements for the per-demo stage functions (``Supervisor._stage_fn``), keyed by stage.
    """

    now: Callable[[], datetime] = datetime.now
    sleep: Callable[[float], None] = time.sleep
    notify: Callable[..., None] | None = None
    disk: Callable[[DictConfig, Iterable[str]], tuple[float, float]] = measure_disk
    build_id: Callable[[Path], int] = demo_utils.cs2_build_id
    generate: Callable[[DictConfig, str, float], Any] | None = None
    sync: Callable[[], Any] | None = None
    smoke: Callable[[], str | None] | None = None
    kill_cs2: Callable[[], None] | None = None
    backoff: Callable[[int], float] | None = None
    stage_fns: dict[str, Callable[[str], dict[str, Any]]] = field(default_factory=dict)


@dataclass
class Night:
    """What happened in one ``run_until`` (also the source of the morning summary).

    Attributes:
        started: Start time.
        deadline: Resolved deadline.
        target_hours: Target of synced QA-passing hours.
        profile: Generation profile.
        reason: Why the night ended.
        alerts: Titles of the alerts sent (guard trips and repeated failures).
        stage_wall_s: Wall seconds per stage (``generate`` and ``smoke`` included).
        stage_runs: Successful runs per stage.
        stopped_generation: Why generation stopped before the end (``None`` while it may continue).
        pause_reason: Why generation is paused right now (disk guard).
        stalled: The pipeline is waiting for a sync to work again (a stall alert was sent).
        blocked: Stages that cannot run tonight (``render`` after a failed smoke test).
    """

    started: datetime
    deadline: datetime
    target_hours: float
    profile: str
    reason: str = ""
    alerts: list[str] = field(default_factory=list)
    stage_wall_s: Counter = field(default_factory=Counter)
    stage_runs: Counter = field(default_factory=Counter)
    stopped_generation: str | None = None
    pause_reason: str = ""
    blocked: set[str] = field(default_factory=set)
    # bookkeeping for the alerts
    failed_in_row: int = 0
    stage_ok: Counter = field(default_factory=Counter)
    stage_failed: Counter = field(default_factory=Counter)
    generate_failures: int = 0
    alerted: set[str] = field(default_factory=set)
    bad_builds: set[int] = field(default_factory=set)
    demo_wall_s: Counter = field(default_factory=Counter)
    stalled: bool = False

    @property
    def exit_code(self) -> int:
        """0 for a normal end (deadline, target, nothing left); ``EXIT_ALERT`` if a guard tripped or an alert was sent."""
        return EXIT_ALERT if self.alerts else 0


class Supervisor:
    """Drives every pending demo through the stages.

    Args:
        cfg: A validated cs2wm config.
        state: Persistent state.
        hooks: Clock, sleep, notifier, disk probe, generate, ... (default: the real ones; fakes in tests).
    """

    def __init__(self, cfg: DictConfig, state: PipelineState, hooks: Hooks | None = None) -> None:
        self.cfg = cfg
        self.state = state
        self.hooks = hooks or Hooks()
        self._notify = self.hooks.notify or partial(
            ntfy.notify, server=cfg.supervisor.ntfy_server, topic=cfg.supervisor.ntfy_topic
        )
        self.night: Night | None = None
        self._current = "idle"  # what the heartbeat reports
        self._next_heartbeat = datetime.max
        self._sync_down = False  # sync failed for the rest of this drain
        self.explicit_paths: dict[str, Path] = {}  # demos given by path on the command line

    def demo_path(self, demo_id: str) -> Path:
        """The ``.dem`` of a demo: the path given on the command line, else ``demos/<kind>/<id>.dem``."""
        if demo_id in self.explicit_paths:
            return self.explicit_paths[demo_id]
        base = Path(self.cfg.paths.demos)
        kind = self.state.demos.get(demo_id, {}).get("kind")
        if kind and (base / kind / f"{demo_id}.dem").is_file():
            return base / kind / f"{demo_id}.dem"
        matches = sorted(base.glob(f"*/{demo_id}.dem"))
        if not matches:
            raise FileNotFoundError(f"demo {demo_id}: no {base}/*/{demo_id}.dem")
        return matches[0]

    def inputs_present(self, stage: str, demo_id: str) -> bool:
        """Whether the files ``stage`` reads for a demo still exist (the demo, the plan, ``render.json``)."""
        plan = Path(self.cfg.paths.plans) / demo_id / "render_plan.json"
        render_json = Path(self.cfg.paths.frames) / demo_id / "render.json"
        if stage == "segment":
            try:
                return self.demo_path(demo_id).is_file()
            except FileNotFoundError:
                return False
        if stage == "render":
            return plan.is_file()
        return plan.is_file() and render_json.is_file()

    def _stage_fn(self, stage: str, **options: Any) -> Callable[[str], dict[str, Any]]:
        """The function that runs one stage on one demo id and returns the info to record; raises on failure."""
        cfg = self.cfg
        if stage in self.hooks.stage_fns:
            return self.hooks.stage_fns[stage]

        def segment(demo_id: str) -> dict[str, Any]:
            from cs2wm.pipeline import segment as seg

            plan_path = seg.run(self.demo_path(demo_id), cfg, options.get("policy"), options.get("player"))
            plan = seg.read_render_plan(plan_path)
            totals = plan["totals"]
            if not totals["episodes"]:
                raise StageFailure("the plan has no episodes (no life long enough to record)")
            return {
                "_output": str(plan_path),
                "demo_s": plan["demo_end_tick"] / plan["timing"]["tickrate"],
                "episodes": totals["episodes"],
                "frames_planned": totals["labeled_frames"],
            }

        def render(demo_id: str) -> dict[str, Any]:
            from cs2wm.pipeline import render as rnd

            result = rnd.run(self.demo_path(demo_id), cfg)
            if not result.ok:
                reason = result.reason or "render failed"
                raise StageFailure(reason, result.attempts, reason.startswith("timeout"))
            return {"rendered_frames": sum(result.frames.values()), "realtime_factor": round(result.realtime_factor, 2)}

        def extract(demo_id: str) -> dict[str, Any]:
            from cs2wm.pipeline import extract_actions

            results = extract_actions.run(self.demo_path(demo_id), cfg)
            return {"qa_episodes": len(results), "qa_passed": sum(r.passed for r in results)}

        def pack(demo_id: str) -> dict[str, Any]:
            from cs2wm.pipeline import pack as pk

            results = pk.run(demo_id, cfg)
            return {"packed_episodes": len(results), "frames": sum(r.n_frames for r in results)}

        return {"segment": segment, "render": render, "extract": extract, "pack": pack}[stage]

    def run_stage(
        self, stage: str, demo_ids: Iterable[str], clock: Callable[[], float] = time.perf_counter, **options: Any
    ) -> StageReport:
        """Run one per-demo stage over ``demo_ids`` in the given order, saving the state after every demo.

        A failing demo is marked ``failed`` with the stage and the reason, logged, and the next demo runs. Only
        ``Exception`` is caught: Ctrl-C and ``SystemExit`` stop the run with the state of finished demos saved.

        Args:
            stage: ``segment``, ``render``, ``extract`` or ``pack``.
            demo_ids: Demos to process.
            clock: Monotonic clock for the wall times.
            **options: Stage options (``policy`` and ``player`` for ``segment``).

        Returns:
            Which demos finished and which failed.
        """
        fn = self._stage_fn(stage, **options)
        report = StageReport()
        for demo_id in demo_ids:
            start = clock()
            try:
                info = fn(demo_id)
            except Exception as exc:
                wall = clock() - start
                expected = isinstance(exc, StageFailure)
                reason = str(exc) if expected else f"{type(exc).__name__}: {exc}"
                log.error("%s %s: FAILED after %.1f s: %s", stage, demo_id, wall, reason, exc_info=not expected)
                self.state.mark(demo_id, StageState.FAILED, reason, wall, stage)
                report.failed[demo_id] = reason
                report.attempts[demo_id] = getattr(exc, "attempts", 1)
                if getattr(exc, "timed_out", False):
                    report.timed_out.add(demo_id)
            else:
                wall = clock() - start
                if "_output" in info:
                    report.outputs[demo_id] = info.pop("_output")
                self.state.mark(demo_id, _OUTPUT_STATE[stage], stage_wall_s=wall, **info)
                log.info("%s %s: ok in %.1f s", stage, demo_id, wall)
                report.done.append(demo_id)
            self.state.save()
        return report

    def run_sync(self, transport: Any = None, clock: Callable[[], float] = time.perf_counter) -> Any:
        """Run ``sync`` and mark the demos whose manifest fully verified on the Mill as synced.

        A demo that did not verify stays ``packed``, so the next sync retries it. A packed demo without any episode
        (none passed QA) has no manifest to verify; its demo and QA files travel with the push, so it is marked
        synced once the pushes succeeded. The wall time of the whole sync is split over the demos it marks in proportion to their packed frames.

        Args:
            transport: Command builder/runner for ``sync.run`` (default: the real ssh/rsync).
            clock: Monotonic clock.

        Returns:
            The ``SyncSummary``.

        Raises:
            SyncError: If the Mill cannot be reached (nothing is marked).
        """
        from cs2wm.pipeline import sync

        start = clock()
        summary = sync.run(self.cfg, transport)
        wall = clock() - start
        done = [d for d in summary.demos if d.synced and self.state.state_of(d.demo_id) is not StageState.SYNCED]
        empty = [
            d
            for d, rec in self.state.demos.items()
            if rec["state"] == "packed" and rec.get("packed_episodes") == 0 and not summary.push_errors
        ]
        marked = [d.demo_id for d in done] + empty
        weight = {d: max(int(self.state.demos.get(d, {}).get("frames", 0)), 1) for d in marked}
        for res in done:
            share = wall * weight[res.demo_id] / sum(weight.values())
            self.state.mark(res.demo_id, StageState.SYNCED, stage_wall_s=share, kind=res.kind)
            log.info("sync %s: ok in %.1f s (%d file(s) verified)", res.demo_id, share, res.verified)
        for demo_id in empty:
            share = wall * weight[demo_id] / sum(weight.values())
            self.state.mark(demo_id, StageState.SYNCED, stage_wall_s=share)
            log.info("sync %s: ok in %.1f s (no episode passed QA; demo and QA pushed)", demo_id, share)
        self.state.save()
        return summary

    # -- the night ----------------------------------------------------------------------------------------------------

    def run_until(self, until: str, target_hours: float, profile: str = "dm") -> Night:
        """Run the pipeline unattended until the deadline or until ``target_hours`` QA-passing hours are synced.

        Each iteration generates one block (unless a guard says otherwise), then advances every pending demo, then
        syncs. A start message is sent first and a morning summary last, also when the loop dies with an exception.

        Args:
            until: Deadline as ``HH:MM`` local time; the next such time, so ``07:30`` at 23:00 is tomorrow.
            target_hours: Stop once this many QA-passing hours are synced.
            profile: Generation profile (``dm`` or ``smoke``).

        Returns:
            What happened; ``exit_code`` is non-zero if a guard tripped or an alert was sent.

        Raises:
            ValueError: If ``until`` is not ``HH:MM``.
        """
        now = self.hooks.now()
        night = self.night = Night(now, resolve_deadline(now, until), target_hours, profile)
        self._current = "starting"
        self._next_heartbeat = now + timedelta(minutes=float(self.cfg.supervisor.heartbeat_min))
        log.info("supervise: target %.2f h, until %s, profile %s", target_hours, night.deadline, profile)
        try:
            self._recover()
            working, free = self._disk()
            self._send(ntfy.format_start(target_hours, night.deadline, now, self._read_build(), working, free, profile))
            night.reason = self._loop()
        except BaseException as exc:
            night.reason = f"stopped by {type(exc).__name__}"
            self._alert("supervisor stopped", f"{type(exc).__name__}: {exc}", "crash")
            raise
        finally:
            self._current = "done"
            self._send(self._summary())
        return night

    def _loop(self) -> str:
        """The generate / drain iterations. Returns why the night ended."""
        night, h = self.night, self.hooks
        while True:
            if self._target_reached():
                return "target reached"
            if h.now() >= night.deadline:
                return "deadline reached"
            self._heartbeat()
            self._check_patch()
            self._check_leftovers()
            generation = self._generate_block()
            progressed = self._drain()
            if self._target_reached():
                return "target reached"
            if h.now() >= night.deadline:
                return "deadline reached"
            if generation == "generated":
                if night.stalled:
                    night.stalled = False
                    self._send(ntfy.Message("dcs2: stall cleared", "Generation resumed.", "default", ("tada",)))
                continue
            if not progressed:
                if not self._can_progress():
                    if generation == "paused":
                        self._alert(
                            "disk guard",
                            f"Generation is paused ({night.pause_reason}) and nothing is left to drain, so nothing "
                            "can free space. The night ends; free disk space or run `cs2wm cleanup --failed`.",
                            "disk-stall",
                        )
                        return f"disk guard: {night.pause_reason}"
                    return night.stopped_generation or "nothing left to do"
                # Something is left but the drain got nowhere (the Mill is down): wait and retry until the deadline.
                why = night.pause_reason if generation == "paused" else night.stopped_generation or "no progress"
                if not night.stalled:
                    night.stalled = True
                    self._alert(
                        "pipeline stalled",
                        f"No progress ({why}); packed demos cannot sync (Mill down?). "
                        f"Retrying every {self.cfg.supervisor.stall_backoff_min} min until the deadline.",
                        "stall",
                    )
                remaining_s = (night.deadline - h.now()).total_seconds()
                h.sleep(min(float(self.cfg.supervisor.stall_backoff_min) * 60, max(remaining_s, 0.0)))

    def _disk(self) -> tuple[float, float]:
        """``(working set GB, free GB)``, not counting the leftovers of failed demos."""
        failed = [d for d, rec in self.state.demos.items() if rec["state"] == StageState.FAILED.value]
        return self.hooks.disk(self.cfg, failed)

    def _check_leftovers(self) -> None:
        """Alert once per night about the working-set dirs of failed demos (they are not counted by the disk guard)."""
        failed = [d for d, rec in self.state.demos.items() if rec["state"] == StageState.FAILED.value]
        found = failed_leftovers(self.cfg, failed) if failed and "leftovers" not in self.night.alerted else []
        if found:
            ids = sorted({d for d, _, _ in found})
            gb = sum(size for _, _, size in found) / 1e9
            shown = ", ".join(ids[:8]) + (f" and {len(ids) - 8} more" if len(ids) > 8 else "")
            self._alert(
                "failed demos hold disk",
                f"{gb:.2f} GB of frames/raw/labels from {len(ids)} failed demo(s): {shown}. They are not counted by "
                "the disk guard; free them with `cs2wm cleanup --failed`.",
                "leftovers",
            )

    def _can_progress(self) -> bool:
        """Whether any demo could still move: something to sync, or a demo whose next stage may run."""
        for rec in self.state.demos.values():
            if rec["state"] == "packed":
                return True
            if rec["state"] in _ADVANCEABLE:
                stage = next(s for s in STAGES if _INPUT_STATE[s].value == rec["state"])
                if stage not in self.night.blocked:
                    return True
        return False

    def _recover(self) -> None:
        """Re-queue failed demos that still have attempts left (a crash between a failure and its retry)."""
        limit = int(self.cfg.supervisor.max_attempts)
        for demo_id, rec in self.state.demos.items():
            stage = rec.get("failed_stage")
            if rec["state"] == "failed" and rec.get("attempts", {}).get(stage, 0) < limit:
                log.info("%s: failed in %s with attempts left; retrying", demo_id, stage)
                self._requeue(demo_id, stage)
        self.state.save()

    # -- generate and the guards ----------------------------------------------------------------------------------------

    def _generate_block(self) -> str:
        """Generate one block unless a guard says no. Returns ``generated``, ``paused`` (disk) or ``stopped``."""
        night, sup = self.night, self.cfg.supervisor
        if night.stopped_generation:
            return "stopped"
        remaining_min = (night.deadline - self.hooks.now()).total_seconds() / 60
        available_min = remaining_min - float(sup.drain_min)
        if available_min < float(self.cfg.generate.chunk_min):
            night.stopped_generation = "deadline close: generation stopped to drain the pipeline"
            log.info("%s (%.0f min to the deadline)", night.stopped_generation, remaining_min)
            return "stopped"
        working, free = self._disk()
        if free < float(sup.min_free_gb):
            night.pause_reason = f"free disk {free:.1f} GB is below supervisor.min_free_gb"
            self._alert(
                "disk almost full",
                f"Only {free:.1f} GB free (minimum {sup.min_free_gb}). Generation is paused until space is freed.",
                "free",
            )
            return "paused"
        if working > float(sup.max_working_gb):
            night.pause_reason = f"working set {working:.1f} GB is over supervisor.max_working_gb"
            log.info(
                "disk guard: working set %.1f GB > %s GB; generation paused until the pipeline drains",
                working,
                sup.max_working_gb,
            )
            return "paused"

        block_min = min(float(sup.generate_block_min), available_min)
        self._current = "generate"
        start = self.hooks.now()
        error: str | None = None
        try:
            self._generate(self.cfg, night.profile, block_min / 60)
        except generate.PatchGuardError as exc:
            self._kill_cs2()
            night.stopped_generation = "CS2 patched (generate's patch guard)"
            self._alert(
                "CS2 patched",
                f"{exc} Generation stopped; the demos already generated are still processed.",
                "patch-guard",
            )
            return "stopped"
        except Exception as exc:  # GenerateError, or anything else a CS2 session can throw
            self._kill_cs2()
            error = f"{type(exc).__name__}: {exc}"
            log.error("generate failed: %s", error, exc_info=not isinstance(exc, generate.GenerateError))
        finally:
            night.stage_wall_s["generate"] += (self.hooks.now() - start).total_seconds()
            self._heartbeat()  # a block can outlast the heartbeat interval; there is no hook inside generate.run
        added = self.state.discover(self.cfg)  # also what a failed block published before it died
        self.state.save()
        if error is None and not added:
            error = "the block produced no demo"
            log.warning("generate: %s", error)
        if error is None:
            night.generate_failures = 0
            night.stage_runs["generate"] += 1
            return "generated"
        night.generate_failures += 1
        if night.generate_failures >= int(sup.max_generate_failures):
            night.stopped_generation = "generate keeps failing"
            self._alert(
                "generate failing",
                f"{night.generate_failures} generate blocks in a row failed (last: {error}). Generation stopped.",
                "generate",
            )
            return "stopped"
        return "generated"

    def _read_build(self) -> int | None:
        try:
            return self.hooks.build_id(Path(self.cfg.paths.cs2_dir))
        except (FileNotFoundError, ValueError) as exc:
            log.warning("cannot read the CS2 build id: %s", exc)
            return None

    def _check_patch(self) -> None:
        """Run the render smoke test when the CS2 build changed since the last one that rendered fine."""
        night = self.night
        build = self._read_build()
        last = self.state.meta.get(BUILD_KEY)
        if build is None or build == last or build in night.bad_builds:
            return
        if last is None:  # first run on this machine: nothing to compare with
            self.state.meta[BUILD_KEY] = build
            self.state.save()
            return
        log.warning("CS2 build changed from %s to %s; running the render smoke test", last, build)
        self._current = "smoke test"
        start = self.hooks.now()
        reason = self._smoke()
        night.stage_wall_s["smoke"] += (self.hooks.now() - start).total_seconds()
        if reason is None:
            night.stage_runs["smoke"] += 1
            self.state.meta[BUILD_KEY] = build
            self.state.save()
            self._send(
                ntfy.Message(
                    "dcs2: CS2 updated",
                    f"Build {last} -> {build}. The render smoke test passed.",
                    "default",
                    ("white_check_mark",),
                )
            )
            return
        night.bad_builds.add(build)
        night.blocked.add("render")
        night.stopped_generation = "CS2 patched and the render smoke test failed"
        self._alert(
            "CS2 patched",
            f"Build {last} -> {build}: the render smoke test failed ({reason}). Generation and rendering are stopped; "
            "demos already rendered are still processed.",
            f"smoke-{build}",
        )

    def render_smoke_test(self) -> str | None:
        """Render ~``supervisor.smoke_s`` seconds of a real, known-good plan to check that rendering survived a patch.

        The most recent demo whose render succeeded (state rendered or later) and that still has its
        ``render_plan.json`` and its ``.dem`` is used. ``make_smoke_plan``
        cuts the first ``smoke_s`` seconds of its first episode into a plan of the same demo, which goes through
        ``render.run`` (HLAE, spectate, capture, downscale, frame-count check) with ``paths.plans``, ``frames`` and
        ``raw`` pointed at ``<root>/smoke``, so the demo's real plan, frames and state are untouched. The scratch
        dir is deleted afterwards. With no such demo (a fresh machine) there is nothing known-good to render: that is
        logged and the test counts as passed.

        Returns:
            The failure reason, or None if the render succeeded (or there was nothing to render).
        """
        from cs2wm.pipeline import render as rnd
        from cs2wm.pipeline import segment as seg

        cfg = self.cfg
        for demo_id in sorted(self.state.demos, reverse=True):
            if self.state.demos[demo_id]["state"] not in _RENDER_OK:
                continue  # a demo that failed to render says nothing about the new build
            plan_path = Path(cfg.paths.plans) / demo_id / "render_plan.json"
            try:
                demo_path = self.demo_path(demo_id)
                plan = make_smoke_plan(seg.read_render_plan(plan_path), float(cfg.supervisor.smoke_s))
            except (OSError, ValueError, KeyError):
                continue
            break
        else:
            log.warning("render smoke test: no demo with a plan to render; the new CS2 build is unverified")
            return None

        scratch = Path(cfg.paths.root) / "smoke"
        shutil.rmtree(scratch, ignore_errors=True)
        smoke_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        smoke_cfg.paths.update(plans=str(scratch / "plans"), frames=str(scratch / "frames"), raw=str(scratch / "raw"))
        smoke_plan = scratch / "plans" / demo_id / "render_plan.json"
        smoke_plan.parent.mkdir(parents=True)
        smoke_plan.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        try:
            result = rnd.run(demo_path, smoke_cfg)
        except Exception as exc:  # noqa: BLE001 - any crash of the launcher means the smoke test failed
            return f"{type(exc).__name__}: {exc}"
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        if result.ok:
            return None
        self._kill_cs2()
        return result.reason or "render failed"

    # -- advancing the demos ----------------------------------------------------------------------------------------

    def _drain(self) -> int:
        """Advance every pending demo (oldest first) through the stages, then sync. Returns the stage runs that finished."""
        night, sup = self.night, self.cfg.supervisor
        if self.state.discover(self.cfg):
            self.state.save()
        self._sync_down = False
        finished = 0
        for demo_id in sorted(d for d, rec in self.state.demos.items() if rec["state"] in _ADVANCEABLE):
            if self.hooks.now() >= night.deadline or self._target_reached():
                break
            finished += self._advance(demo_id)
            self._heartbeat()
            if self._disk()[0] > float(sup.max_working_gb):
                finished += self._sync_packed()  # the packed hdf5 is what is left to free
        return finished + self._sync_packed()

    def _advance(self, demo_id: str) -> int:
        """Run one demo through segment, render, extract and pack. Returns the stages that finished."""
        finished = 0
        while True:
            state = self.state.state_of(demo_id)
            stage = next((s for s in STAGES[:-1] if _INPUT_STATE[s] is state), None)
            if stage is None or stage in self.night.blocked or self.hooks.now() >= self.night.deadline:
                return finished  # the rest of this demo waits for the next night
            if not self._run_with_retries(stage, demo_id):
                return finished
            finished += 1

    def _run_with_retries(self, stage: str, demo_id: str) -> bool:
        """Run one stage of one demo, up to ``supervisor.max_attempts`` attempts with exponential backoff.

        Launches a stage made internally (render retries a crash or timeout ``render.retries`` times) count against
        the same budget, and a timeout is not retried again. A demo that used ``supervisor.max_demo_wall_min`` this
        night is given up on at its next failure and not started on a new stage. A failed attempt kills stray
        CS2/HLAE processes (render), counts in ``attempts[stage]`` and puts the demo back to the stage's input state;
        the last one leaves it ``failed`` with the reason. Nothing is started after the deadline.

        Returns:
            True if the stage finished; False if the demo failed or must wait for the next night.
        """
        night, rec, sup = self.night, self.state.demos[demo_id], self.cfg.supervisor
        limit, wall_cap_s = int(sup.max_attempts), float(sup.max_demo_wall_min) * 60
        while True:
            if self.hooks.now() >= night.deadline or night.demo_wall_s[demo_id] >= wall_cap_s:
                return False  # stays queued for the next night
            self._current = f"{stage} {demo_id}"
            start = self.hooks.now()
            report = self.run_stage(stage, [demo_id])
            wall = (self.hooks.now() - start).total_seconds()
            night.stage_wall_s[stage] += wall
            night.demo_wall_s[demo_id] += wall
            if demo_id in report.done:
                night.stage_runs[stage] += 1
                night.stage_ok[stage] += 1
                if stage == "pack":
                    night.failed_in_row = 0
                if rec.get("attempts", {}).pop(stage, None) is not None or rec.pop("last_error", None):
                    self.state.save()
                return True
            reason = report.failed.get(demo_id, "no result")
            if stage == "render":
                self._kill_cs2()
            attempts = rec.setdefault("attempts", {})
            attempts[stage] = attempts.get(stage, 0) + report.attempts.get(demo_id, 1)
            over_cap = night.demo_wall_s[demo_id] >= wall_cap_s
            if demo_id in report.timed_out:
                attempts[stage] = max(attempts[stage], limit)  # it had its internal retry: no more
            if over_cap and attempts[stage] < limit:
                attempts[stage] = limit
                reason += f" (gave up: {night.demo_wall_s[demo_id] / 60:.0f} min spent on this demo tonight)"
                self.state.demos[demo_id]["reason"] = reason
            if attempts[stage] >= limit:
                self.state.save()
                self._count_failure(stage, demo_id, reason)
                return False
            self._requeue(demo_id, stage)
            self.state.save()
            delay = self._backoff(attempts[stage])
            log.warning(
                "%s %s: attempt %d/%d failed (%s); retrying in %.0f s",
                stage,
                demo_id,
                attempts[stage],
                limit,
                reason,
                delay,
            )
            self.hooks.sleep(delay)

    def _requeue(self, demo_id: str, stage: str) -> None:
        """Put a demo that failed in ``stage`` back to the state the stage starts from, keeping the reason."""
        rec = self.state.demos[demo_id]
        rec["state"] = _INPUT_STATE[stage].value
        rec["last_error"] = rec.pop("reason", None)
        rec.pop("failed_stage", None)

    def _count_failure(self, stage: str, demo_id: str, reason: str) -> None:
        """A demo is out of attempts: alert on a streak of failed demos or a stage that fails for every demo."""
        night, limit = self.night, int(self.cfg.supervisor.alert_failed_in_row)
        night.failed_in_row += 1
        night.stage_failed[stage] += 1
        log.error("%s %s: giving up after %s attempts: %s", stage, demo_id, self.cfg.supervisor.max_attempts, reason)
        self._alert(
            f"{stage} failed for a demo",
            f"{demo_id} failed in {stage} after {self.cfg.supervisor.max_attempts} attempts: {reason}. "
            "The night goes on with the next demo.",
            f"exhausted-{stage}",
        )  # the first one per stage; the streak alerts below add to it
        if night.failed_in_row == limit:  # once per streak
            self._alert(
                "repeated failures",
                f"{night.failed_in_row} demos in a row failed. Last: {stage} {demo_id}: {reason}",
                f"row-{night.alerts.count('repeated failures')}",
            )
        if night.stage_failed[stage] >= limit and not night.stage_ok[stage]:
            self._alert(
                f"{stage} fails for every demo",
                f"{night.stage_failed[stage]} demos failed in {stage} and none passed tonight. Last: {demo_id}: {reason}",
                f"stage-{stage}",
            )

    def _sync_packed(self) -> int:
        """Sync everything packed (up to ``max_attempts`` tries with backoff). Returns the demos that reached the Mill."""
        night, limit = self.night, int(self.cfg.supervisor.max_attempts)
        if self._sync_down or not any(r["state"] == "packed" for r in self.state.demos.values()):
            return 0
        synced_before = self._count("synced")
        ok = False
        for attempt in range(1, limit + 1):
            self._current = "sync"
            start = self.hooks.now()
            try:
                ok = bool((self.hooks.sync or self.run_sync)().ok)
            except Exception as exc:  # noqa: BLE001 - the Mill being down must not stop the night
                log.error("sync failed: %s: %s", type(exc).__name__, exc)
                ok = False
            night.stage_wall_s["sync"] += (self.hooks.now() - start).total_seconds()
            if ok or attempt == limit:
                break
            self.hooks.sleep(self._backoff(attempt))
            if self.hooks.now() >= night.deadline:
                break
        gained = self._count("synced") - synced_before
        if gained:
            night.stage_runs["sync"] += gained
            night.stage_ok["sync"] += gained
        if not ok and not gained:
            self._sync_down = True
            self._alert(
                "sync failing",
                "Nothing could be synced to the Mill. The packed demos stay on disk and are retried.",
                "sync",
            )
        return gained

    # -- bookkeeping --------------------------------------------------------------------------------------------------

    def _count(self, state: str) -> int:
        return sum(1 for rec in self.state.demos.values() if rec["state"] == state)

    def _hours(self, states: tuple[str, ...], key: str, kind: str | None = None) -> float:
        """Hours (at ``timing.fps``) of the ``key`` frames of the demos in ``states`` (of one ``kind``, if given)."""
        recs = (r for r in self.state.demos.values() if r["state"] in states and kind in (None, r.get("kind")))
        return sum(r.get(key, 0) for r in recs) / float(self.cfg.timing.fps) / 3600

    def synced_hours(self, kind: str | None = None) -> float:
        """QA-passing hours that reached the Mill (of one episode kind, if given)."""
        return self._hours(("synced",), "frames", kind)

    def _target_reached(self) -> bool:
        return self.synced_hours(self.night.profile) >= self.night.target_hours  # only the profile's kind counts

    def _snapshot(self) -> ntfy.Snapshot:
        demos = self.state.demos.values()
        ordered = [s.value for s in StageState]
        past_segment = [r for r in demos if r["state"] != "pending"]
        extracted = [r for r in demos if "qa_episodes" in r]
        qa_total = sum(r["qa_episodes"] for r in extracted)
        working, free = self._disk()
        return ntfy.Snapshot(
            generated_h=sum(r.get("demo_s", 0.0) for r in past_segment) / 3600,
            rendered_h=sum(r.get("rendered_frames", 0) for r in demos) / float(self.cfg.timing.fps) / 3600,
            synced_h=self.synced_hours(),
            qa_pass_rate=sum(r["qa_passed"] for r in extracted) / qa_total if qa_total else None,
            working_gb=working,
            free_gb=free,
            stage=self._current,
            counts={s: self._count(s) for s in ordered},
        )

    def _summary(self) -> ntfy.Message:
        night = self.night
        failed = Counter(
            (rec["failed_stage"], rec["reason"]) for rec in self.state.demos.values() if rec["state"] == "failed"
        )
        return ntfy.format_summary(
            ntfy.NightSummary(
                snapshot=self._snapshot(),
                target_hours=night.target_hours,
                started=night.started,
                ended=self.hooks.now(),
                reason=night.reason,
                failures=[(stage, reason[:120], n) for (stage, reason), n in sorted(failed.items())],
                stage_wall_s=dict(night.stage_wall_s),
                stage_runs=dict(night.stage_runs),
                alerts=night.alerts,
            )
        )

    def _heartbeat(self) -> None:
        now = self.hooks.now()
        if now >= self._next_heartbeat:
            self._next_heartbeat = now + timedelta(minutes=float(self.cfg.supervisor.heartbeat_min))
            self._send(ntfy.format_heartbeat(self._snapshot(), self.night.target_hours, now, self.night.deadline))

    def _alert(self, kind: str, detail: str, key: str) -> None:
        """Send an alert once per ``key`` per night and remember it (it makes the exit code non-zero)."""
        if key in self.night.alerted:
            return
        self.night.alerted.add(key)
        self.night.alerts.append(kind)
        log.error("ALERT %s: %s", kind, detail)
        self._send(ntfy.format_alert(kind, detail))

    def _send(self, message: ntfy.Message) -> None:
        """Post a message; a failing notifier never stops the night. Only the title is logged, never the topic."""
        log.info("ntfy: %s", message.title)
        try:
            self._notify(message.body, title=message.title, priority=message.priority, tags=message.tags)
        except Exception as exc:  # noqa: BLE001
            log.warning("ntfy notification failed (%s)", type(exc).__name__)

    def _backoff(self, attempt: int) -> float:
        if self.hooks.backoff:
            return self.hooks.backoff(attempt)
        return float(self.cfg.supervisor.backoff_s) * 2 ** (attempt - 1)

    def _kill_cs2(self) -> None:
        if self.hooks.kill_cs2:
            self.hooks.kill_cs2()
        else:
            proc.kill_by_name(subprocess.run, os.name == "nt", ["cs2.exe", str(self.cfg.render.hlae_exe)])

    def _generate(self, cfg: DictConfig, profile: str, hours: float) -> Any:
        if self.hooks.generate:
            return self.hooks.generate(cfg, profile, hours)
        return generate.run(cfg, profile, hours)

    def _smoke(self) -> str | None:
        return (self.hooks.smoke or self.render_smoke_test)()

    def status_table(self) -> str:
        """Return the state table and totals printed by ``cs2wm status``.

        One row per demo (id, kind, state, wall seconds of the last run of each stage, failure), then the demos per
        state, the QA-passing hours packed (frames / fps / 3600) and the wall time per stage per demo-hour (stage wall
        time over the demo length, for the demos that have both). Demos currently ``failed`` are left out of the per-hour
        figures, since their last timings are of attempts that did not produce output.
        """
        demos = self.state.demos
        header = ["demo", "kind", "state", *STAGES]
        rows = [header]
        for demo_id in sorted(demos):
            rec = demos[demo_id]
            walls = [f"{rec['stages'][s]['wall_s']:.1f}" if s in rec["stages"] else "-" for s in STAGES]
            rows.append([demo_id, rec.get("kind", "?"), rec["state"], *walls])
        widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
        lines = []
        for r, demo_id in zip(rows, ["", *sorted(demos)]):
            line = "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
            rec = demos.get(demo_id)
            if rec and rec["state"] == "failed":
                line += f"  [{rec['failed_stage']}] {rec['reason']}"
            lines.append(line)
        if not demos:
            lines.append("(no demos)")
        lines.append("(stage columns: wall seconds of the last run)")

        counts = {s.value: 0 for s in StageState}
        for rec in demos.values():
            counts[rec["state"]] += 1
        lines += ["", "demos: " + ", ".join(f"{n} {s}" for s, n in counts.items()) + f" ({len(demos)} total)"]
        fps = float(self.cfg.timing.fps)
        packed = sum(rec.get("frames", 0) for rec in demos.values() if rec["state"] in ("packed", "synced"))
        lines.append(f"QA-passing hours packed: {packed / fps / 3600:.3f} ({packed} frames)")
        per_hour = []
        for stage in STAGES:
            timed = [r for r in demos.values() if r["state"] != "failed" and stage in r["stages"] and r.get("demo_s")]
            hours = sum(r["demo_s"] for r in timed) / 3600
            if hours > 0:
                per_hour.append(f"{stage} {sum(r['stages'][stage]['wall_s'] for r in timed) / hours:.0f}")
        lines.append("wall s per demo-hour (failed demos excluded): " + (", ".join(per_hour) if per_hour else "n/a"))
        return "\n".join(lines)
