"""supervisor: runs stages 1-6 unattended for each demo, as one Windows scheduled task (#20).

Per demo: generate -> segment -> render -> extract -> pack -> sync, with up to ``supervisor.max_attempts`` attempts
and exponential backoff per stage, a disk guard (``supervisor.max_working_gb``), a CS2 patch guard (render a 10 s
smoke test when the build id changes), and ntfy start/hourly/alert/morning messages. State is persisted atomically in
``state/pipeline.json`` so a reboot resumes where it stopped.

Done so far (#19): the persistent state, ``Supervisor.run_stage`` / ``run_sync`` (what ``cs2wm <stage> --all-pending``
and ``cs2wm sync`` call) and ``status_table``. The loop, retries, guards and ntfy are still #20.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

log = logging.getLogger(__name__)

STATE_SCHEMA_VERSION = 1
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
_OUTPUT_STATE = {
    "segment": StageState.SEGMENTED,
    "render": StageState.RENDERED,
    "extract": StageState.EXTRACTED,
    "pack": StageState.PACKED,
    "sync": StageState.SYNCED,
}


class StageFailure(RuntimeError):
    """A stage ran but did not produce its output (the message is recorded as the failure reason)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class PipelineState:
    """Persistent supervisor state.

    Each demo has a record with ``state`` (a ``StageState`` value), ``kind``, ``failed_stage`` and ``reason`` (set
    while ``failed``), ``stages`` (``{stage: {"wall_s": float, "at": ISO time}}`` of the last run of each stage),
    ``updated`` and whatever ``mark`` was given as ``info`` (``demo_s``, ``frames``, ...).

    Attributes:
        path: ``state/pipeline.json``.
        demos: Record per demo id.
    """

    path: Path
    demos: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> PipelineState:
        """Load the state file (empty state if it does not exist).

        Raises:
            ValueError: If the file is not a state file. Saves are atomic, so this means someone edited it.
        """
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls(path)
        except ValueError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported pipeline state schema")
        return cls(path, dict(raw.get("demos", {})))

    def save(self) -> None:
        """Write the state atomically (temp file + fsync + rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"schema_version": STATE_SCHEMA_VERSION, "demos": self.demos}, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
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
    """

    done: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


class Supervisor:
    """Drives every pending demo through the stages.

    Args:
        cfg: A validated cs2wm config.
        state: Persistent state.
    """

    def __init__(self, cfg: DictConfig, state: PipelineState) -> None:
        self.cfg = cfg
        self.state = state
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

    def _stage_fn(self, stage: str, **options: Any) -> Callable[[str], dict[str, Any]]:
        """The function that runs one stage on one demo id and returns the info to record; raises on failure."""
        cfg = self.cfg

        def segment(demo_id: str) -> dict[str, Any]:
            from cs2wm.pipeline import segment as seg

            plan_path = seg.run(self.demo_path(demo_id), cfg, options.get("policy"), options.get("player"))
            plan = seg.read_render_plan(plan_path)
            totals = plan["totals"]
            if not totals["episodes"]:
                raise StageFailure("the plan has no episodes (no life long enough to record)")
            return {
                "demo_s": plan["demo_end_tick"] / plan["timing"]["tickrate"],
                "episodes": totals["episodes"],
                "frames_planned": totals["labeled_frames"],
            }

        def render(demo_id: str) -> dict[str, Any]:
            from cs2wm.pipeline import render as rnd

            result = rnd.run(self.demo_path(demo_id), cfg)
            if not result.ok:
                raise StageFailure(result.reason or "render failed")
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
            else:
                wall = clock() - start
                self.state.mark(demo_id, _OUTPUT_STATE[stage], stage_wall_s=wall, **info)
                log.info("%s %s: ok in %.1f s", stage, demo_id, wall)
                report.done.append(demo_id)
            self.state.save()
        return report

    def run_sync(self, transport: Any = None, clock: Callable[[], float] = time.perf_counter) -> Any:
        """Run ``sync`` and mark the demos whose manifest fully verified on the Mill as synced.

        A demo that did not verify stays ``packed``, so the next sync retries it. A packed demo without any episode
        (none passed QA) has no manifest to verify; its demo and QA files travel with the push, so it is marked
        synced once the pushes succeeded. The wall time of the whole sync is split evenly over the demos it marks.

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
        share = wall / max(len(done) + len(empty), 1)
        for res in done:
            self.state.mark(res.demo_id, StageState.SYNCED, stage_wall_s=share, kind=res.kind)
            log.info("sync %s: ok in %.1f s (%d file(s) verified)", res.demo_id, share, res.verified)
        for demo_id in empty:
            self.state.mark(demo_id, StageState.SYNCED, stage_wall_s=share)
            log.info("sync %s: ok in %.1f s (no episode passed QA; demo and QA pushed)", demo_id, share)
        self.state.save()
        return summary

    def run_until(self, until: str, target_hours: float, profile: str = "dm") -> None:
        """Loop until the deadline or target is reached.

        Args:
            until: Deadline as ``HH:MM`` local time.
            target_hours: Stop once this many QA-passing hours are synced.
            profile: Generation profile (``dm`` or ``smoke``).
        """
        # TODO(#20): default schedule "generate 60 min -> advance everything pending", with guards and ntfy.
        raise NotImplementedError("TODO(#20): Supervisor.run_until")

    def status_table(self) -> str:
        """Return the state table and totals printed by ``cs2wm status``.

        One row per demo (id, kind, state, wall seconds of the last run of each stage, failure), then the demos per
        state, the QA-passing hours packed (frames / fps / 3600) and the wall time per stage per demo-hour (stage wall
        time over the demo length, for the demos that have both).
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
            timed = [r for r in demos.values() if stage in r["stages"] and r.get("demo_s")]
            hours = sum(r["demo_s"] for r in timed) / 3600
            if hours > 0:
                per_hour.append(f"{stage} {sum(r['stages'][stage]['wall_s'] for r in timed) / hours:.0f}")
        lines.append("wall s per demo-hour: " + (", ".join(per_hour) if per_hour else "n/a"))
        return "\n".join(lines)
