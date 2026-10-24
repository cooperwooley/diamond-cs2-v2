"""Tests for the overnight supervisor loop, guards and ntfy messages (#20), with fake stages, clock, sleep and notifier.

Nothing here starts CS2 or HLAE: the per-demo stages, ``generate``, ``sync``, the disk probe and the build-id reader are
fakes injected through ``supervisor.Hooks``; the render smoke test is covered with the real ``render.run`` on a fake
launcher.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import requests
from omegaconf import DictConfig

from cs2wm import cli
from cs2wm.config import ConfigError, load_config
from cs2wm.pipeline import ntfy, render, segment
from cs2wm.pipeline.generate import GenerateError, PatchGuardError
from cs2wm.pipeline.supervisor import (
    BUILD_KEY,
    EXIT_ALERT,
    Hooks,
    Night,
    PipelineState,
    StageFailure,
    StageState,
    Supervisor,
    make_smoke_plan,
    resolve_deadline,
)
from cs2wm.pipeline.sync import SyncError
from tests.test_render import FakeLauncher, _hooks_clock, _plan, _write_plan
from tests.test_render import _cfg as render_cfg

T0 = datetime(2026, 3, 1, 23, 0)
FRAMES = 9600  # per demo: 10 min at 16 fps = 1/6 h


class Crash(BaseException):
    """Simulates the machine dying mid-stage (not an ``Exception``, so ``run_stage`` does not swallow it)."""


class World:
    """A fake machine: clock, notifier, disk, CS2 build, and scripted stages that cost fake time."""

    cost: ClassVar[dict[str, int]] = {"segment": 10, "render": 300, "extract": 10, "pack": 20, "sync": 30}

    def __init__(self, cfg: DictConfig, start: datetime = T0, per_block: int = 2) -> None:
        self.cfg = cfg
        self.t = start
        self.per_block = per_block
        self.sent: list[dict[str, Any]] = []
        self.sleeps: list[float] = []
        self.events: list[str] = []
        self.generate_hours: list[float] = []
        self.kills = 0
        self.working, self.free = 0.0, 500.0
        self.build = 100
        self.n_demos = 0
        self.fail: dict[tuple[str, str], int] = {}  # (stage, demo) -> failures left; demo "*" = every demo
        self.crash_at: tuple[str, str] | None = None
        self.generate_script: list[Callable[[], None]] = []
        self.sync_failures = 0
        self.smoke_result: str | None = None
        self.smoke_calls = 0
        self.synced_working_gb = 0.0  # working set freed per synced demo
        self.render_working_gb = 0.0  # working set added per rendered demo

    # -- hooks --------------------------------------------------------------------------------------------------------
    def now(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)

    def notify(self, message: str, title: str | None = None, priority: str = "default", tags: Any = ()) -> None:
        self.sent.append({"title": title, "body": message, "priority": priority, "tags": tuple(tags)})

    def generate(self, cfg: DictConfig, profile: str, hours: float) -> list[Path]:
        assert len(self.generate_hours) < 40, "runaway loop"
        self.generate_hours.append(hours)
        self.events.append("generate")
        self.advance(hours * 3600)
        if self.generate_script:
            self.generate_script.pop(0)()
            return []
        return self.publish(self.per_block)

    def publish(self, n: int) -> list[Path]:
        out = []
        for _ in range(n):
            self.n_demos += 1
            path = Path(self.cfg.paths.demos) / "dm" / f"d{self.n_demos:03d}.dem"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            out.append(path)
        return out

    def stage_fn(self, stage: str) -> Callable[[str], dict[str, Any]]:
        def run(demo_id: str) -> dict[str, Any]:
            self.events.append(f"{stage} {demo_id}")
            self.advance(self.cost[stage])
            if self.crash_at == (stage, demo_id):
                raise Crash
            for key in ((stage, demo_id), (stage, "*")):
                if self.fail.get(key, 0) > 0:
                    self.fail[key] -= 1
                    raise StageFailure(f"{stage} broke")
            if stage == "render":
                self.working += self.render_working_gb
            return {
                "segment": {"demo_s": 600.0, "episodes": 2},
                "render": {"rendered_frames": FRAMES},
                "extract": {"qa_episodes": 2, "qa_passed": 1},
                "pack": {"packed_episodes": 1, "frames": FRAMES},
            }[stage]

        return run

    def sync(self, sup: Supervisor) -> SimpleNamespace:
        self.events.append("sync")
        self.advance(self.cost["sync"])
        if self.sync_failures:
            self.sync_failures -= 1
            raise SyncError("the Mill is unreachable")
        for demo_id in sup.state.pending_for("sync"):
            sup.state.mark(demo_id, StageState.SYNCED, stage_wall_s=1.0)
            self.working -= self.synced_working_gb
        sup.state.save()
        return SimpleNamespace(ok=True)

    def hooks(self, sup_ref: list[Supervisor]) -> Hooks:
        return Hooks(
            now=self.now,
            sleep=self.sleep,
            notify=self.notify,
            disk=lambda cfg: (self.working, self.free),
            build_id=lambda cs2_dir: self.build,
            generate=self.generate,
            sync=lambda: self.sync(sup_ref[0]),
            smoke=self.smoke,
            kill_cs2=self.kill,
            backoff=lambda n: 10.0 * n,
            stage_fns={s: self.stage_fn(s) for s in ("segment", "render", "extract", "pack")},
        )

    def smoke(self) -> str | None:
        self.smoke_calls += 1
        self.events.append("smoke")
        self.advance(15)
        return self.smoke_result

    def kill(self) -> None:
        self.kills += 1

    # -- helpers ------------------------------------------------------------------------------------------------------
    def titles(self) -> list[str]:
        return [m["title"] for m in self.sent]

    def by_title(self, fragment: str) -> list[dict[str, Any]]:
        return [m for m in self.sent if fragment in m["title"]]

    def stage_events(self, stage: str) -> list[str]:
        return [e for e in self.events if e == stage or e.startswith(stage + " ")]


@pytest.fixture
def cfg(tmp_path: Path) -> DictConfig:
    return load_config(
        overrides=[
            f"paths.root={tmp_path / 'data'}",
            f"paths.cs2_dir={tmp_path / 'cs2'}",
            "supervisor.drain_min=30",
            "supervisor.generate_block_min=60",
            "supervisor.max_working_gb=8",
            "supervisor.min_free_gb=15",
        ]
    )


def make(cfg: DictConfig, world: World | None = None, state: PipelineState | None = None) -> tuple[Supervisor, World]:
    world = world or World(cfg)
    state = state or PipelineState(Path(cfg.paths.state) / "pipeline.json")
    ref: list[Supervisor] = []
    sup = Supervisor(cfg, state, world.hooks(ref))
    ref.append(sup)
    return sup, world


def reload(cfg: DictConfig, world: World) -> Supervisor:
    """A new Supervisor on the same state file, as after a crash or reboot."""
    return make(cfg, world, PipelineState.load(Path(cfg.paths.state) / "pipeline.json"))[0]


# --- deadline ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("now", "until", "expected"),
    [
        (datetime(2026, 3, 1, 23, 0), "07:30", datetime(2026, 3, 2, 7, 30)),  # across midnight
        (datetime(2026, 3, 1, 6, 0), "07:30", datetime(2026, 3, 1, 7, 30)),  # same day
        (datetime(2026, 3, 1, 7, 30), "07:30", datetime(2026, 3, 2, 7, 30)),  # exactly now: tomorrow
        (datetime(2026, 12, 31, 23, 59), "00:10", datetime(2027, 1, 1, 0, 10)),  # across a year
    ],
)
def test_resolve_deadline(now: datetime, until: str, expected: datetime) -> None:
    assert resolve_deadline(now, until) == expected


@pytest.mark.parametrize("bad", ["", "7", "25:00", "07:61", "ab:cd", "07:30:00"])
def test_resolve_deadline_rejects_bad_times(bad: str) -> None:
    with pytest.raises(ValueError, match="HH:MM"):
        resolve_deadline(T0, bad)


def test_night_runs_across_midnight_and_stops_generating_early(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    night = sup.run_until("00:50", target_hours=99)  # started 23:00: the deadline is tomorrow 00:50
    assert night.deadline == datetime(2026, 3, 2, 0, 50)
    assert world.generate_hours == [
        pytest.approx(1.0)
    ]  # at 00:11 only 39 min are left: less than the 30 min margin + a chunk
    assert (
        world.t.date() == datetime(2026, 3, 2).date() and world.t < night.deadline
    )  # drained after midnight, stopped early
    assert "deadline" in night.reason
    assert all(r["state"] == "synced" for r in sup.state.demos.values())


def test_last_block_is_shortened_to_leave_the_drain_margin(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    sup.run_until("23:50", target_hours=99)  # 50 min - 30 min drain = 20 min
    assert world.generate_hours == [pytest.approx(20 / 60)]
    assert sup.night.stopped_generation and "deadline" in sup.night.stopped_generation


def test_no_generation_when_less_than_a_chunk_is_left(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    sup.run_until("23:35", target_hours=99)  # 35 - 30 = 5 min < generate.chunk_min
    assert world.generate_hours == []


def test_deadline_mid_drain_starts_no_new_demo_but_syncs_what_is_packed(cfg: DictConfig) -> None:
    world = World(cfg, per_block=3)
    world.cost = {**World.cost, "render": 1500}
    sup, world = make(cfg, world)
    night = sup.run_until("23:50", target_hours=99)  # block ends 23:20; each demo takes ~26 min
    states = {d: r["state"] for d, r in sup.state.demos.items()}
    assert states["d001"] == "synced" and states["d002"] == "synced"  # d002 finished after the deadline, then synced
    assert states["d003"] == "pending"  # never started
    assert night.reason == "deadline reached" and night.exit_code == 0


# --- the loop ----------------------------------------------------------------------------------------------------


def test_demos_go_through_every_stage_in_order_and_the_night_is_reported(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    night = sup.run_until("07:30", target_hours=1 / 3 + 0.01)
    assert world.stage_events("segment d001") and world.events[:7] == [
        "generate",
        "segment d001",
        "render d001",
        "extract d001",
        "pack d001",
        "segment d002",
        "render d002",
    ]
    assert all(rec["state"] == "synced" for rec in sup.state.demos.values())
    assert night.reason == "target reached" and night.exit_code == 0
    assert world.titles()[0] == "dcs2: night started" and world.titles()[-1] == "dcs2: morning summary"
    assert not night.alerts
    for rec in sup.state.demos.values():
        assert set(rec["stages"]) == {"segment", "render", "extract", "pack", "sync"}
    assert PipelineState.load(sup.state.path).demos == sup.state.demos  # saved


def test_target_hours_stops_generation_and_the_loop(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    night = sup.run_until("07:30", target_hours=0.3)  # two demos = 1/3 h
    assert len(world.generate_hours) == 1
    assert night.reason == "target reached" and sup.synced_hours() == pytest.approx(1 / 3)


def test_target_not_reached_before_the_deadline_is_not_an_alert(cfg: DictConfig) -> None:
    sup, _ = make(cfg)
    night = sup.run_until("01:00", target_hours=99)
    assert "deadline" in night.reason and night.exit_code == 0 and sup.synced_hours() > 0


def test_resume_after_a_crash_continues_where_the_state_stopped(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    world.crash_at = ("render", "d002")
    with pytest.raises(Crash):
        sup.run_until("07:30", target_hours=1 / 3 + 0.01)
    assert world.by_title("supervisor stopped") and "morning summary" in world.titles()[-1]
    saved = PipelineState.load(sup.state.path)
    assert saved.state_of("d001") is StageState.PACKED and saved.state_of("d002") is StageState.SEGMENTED

    world.crash_at = None
    world.events.clear()
    world.generate_hours.clear()
    night = reload(cfg, world).run_until("07:30", target_hours=1 / 3 + 0.01)
    assert world.stage_events("segment d001") == [] and world.stage_events("segment d002") == []
    assert world.stage_events("render d002") and night.reason == "target reached"
    assert all(r["state"] == "synced" for r in PipelineState.load(sup.state.path).demos.values())


def test_a_new_night_resumes_demos_left_by_the_last_one(cfg: DictConfig) -> None:
    state = PipelineState(Path(cfg.paths.state) / "pipeline.json")
    state.mark("old1", StageState.EXTRACTED, kind="dm")
    state.mark("old2", StageState.PACKED, kind="dm", frames=FRAMES)
    state.save()
    sup, world = make(cfg, state=PipelineState.load(state.path))
    sup.run_until("23:20", target_hours=99)  # too short to generate: only drains
    assert world.generate_hours == []
    assert world.events == ["pack old1", "sync"]
    assert {r["state"] for r in sup.state.demos.values()} == {"synced"}


# --- retries -----------------------------------------------------------------------------------------------------


def test_a_failed_stage_is_retried_with_backoff_and_kills_cs2(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    world.fail[("render", "d001")] = 2
    night = sup.run_until("07:30", target_hours=0.3)
    assert world.stage_events("render d001") == ["render d001"] * 3
    assert world.sleeps == [10.0, 20.0]  # backoff(1), backoff(2)
    assert world.kills == 2
    rec = sup.state.demos["d001"]
    assert rec["state"] == "synced" and not rec.get("attempts") and "last_error" not in rec
    assert night.reason == "target reached" and not night.alerts


def test_default_backoff_is_exponential(cfg: DictConfig) -> None:
    sup, _ = make(cfg)
    sup.hooks.backoff = None
    assert [sup._backoff(n) for n in (1, 2, 3)] == [60.0, 120.0, 240.0]


def test_a_demo_is_failed_after_max_attempts_and_the_night_goes_on(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    world.fail[("render", "d001")] = 99
    night = sup.run_until("07:30", target_hours=0.3)
    bad = sup.state.demos["d001"]
    assert bad["state"] == "failed" and bad["failed_stage"] == "render" and bad["reason"] == "render broke"
    assert bad["attempts"] == {"render": 3}
    assert world.stage_events("render d001") == ["render d001"] * 3 and world.sleeps == [10.0, 20.0]
    assert world.stage_events("segment d002") and sup.state.demos["d002"]["state"] == "synced"
    assert night.exit_code == 0  # one bad demo is not an alert


def test_non_render_failures_do_not_kill_cs2(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    world.fail[("extract", "d001")] = 1
    sup.run_until("07:30", target_hours=0.3)
    assert world.kills == 0 and sup.state.demos["d001"]["state"] == "synced"


def test_no_retry_is_started_when_the_backoff_ends_after_the_deadline(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    world.fail[("segment", "*")] = 1
    sup.hooks.backoff = lambda n: 7200.0
    sup.run_until("00:30", target_hours=99)
    rec = sup.state.demos["d001"]
    assert rec["state"] == "pending" and rec["attempts"] == {"segment": 1}  # queued for the next night
    assert world.sleeps == [7200.0] and world.stage_events("segment d001") == ["segment d001"]


def test_failed_demos_with_attempts_left_are_retried_on_the_next_start(cfg: DictConfig) -> None:
    state = PipelineState(Path(cfg.paths.state) / "pipeline.json")
    state.mark("a", StageState.PENDING, kind="dm")
    state.mark("a", StageState.FAILED, "boom", stage="segment")
    state.demos["a"]["attempts"] = {"segment": 1}
    state.mark("b", StageState.PENDING, kind="dm")
    state.mark("b", StageState.FAILED, "boom", stage="segment")
    state.demos["b"]["attempts"] = {"segment": 3}
    state.save()
    sup, _ = make(cfg, state=PipelineState.load(state.path))
    sup.run_until("23:20", target_hours=99)
    assert sup.state.state_of("a") is StageState.SYNCED and sup.state.state_of("b") is StageState.FAILED


def test_repeated_failures_alert_immediately_with_high_priority(cfg: DictConfig) -> None:
    sup, world = make(cfg, World(cfg, per_block=4))
    world.fail[("render", "*")] = 99
    night = sup.run_until("07:30", target_hours=99)
    alerts = world.by_title("dcs2 alert")
    kinds = " | ".join(a["title"] for a in alerts)
    assert "repeated failures" in kinds and "render fails for every demo" in kinds
    assert all(a["priority"] == "high" for a in alerts)
    assert night.exit_code == EXIT_ALERT and "repeated failures" in night.alerts
    # the alert is sent when the third demo fails, not at the end of the night
    assert world.titles().index(alerts[0]["title"]) < len(world.titles()) - 1
    assert len(world.by_title("repeated failures")) == 1  # once per streak, not once per demo


def test_a_success_resets_the_failure_streak(cfg: DictConfig) -> None:
    sup, world = make(cfg, World(cfg, per_block=5))
    for demo in ("d001", "d002", "d004", "d005"):
        world.fail[("segment", demo)] = 99
    night = sup.run_until("23:40", target_hours=99)
    assert sup.state.demos["d003"]["state"] == "synced"
    assert night.exit_code == 0  # 2 + 2 failures in a row, never 3


def test_sync_failures_are_retried_then_alert_but_never_fail_the_demos(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    world.sync_failures = 99
    night = sup.run_until("23:40", target_hours=99)
    assert {r["state"] for r in sup.state.demos.values()} == {"packed"}
    assert night.exit_code == EXIT_ALERT and "sync failing" in night.alerts
    assert world.events.count("sync") % 3 == 0 and len(world.by_title("sync failing")) == 1  # alerted once


def test_sync_recovers_on_a_retry(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    world.sync_failures = 1
    night = sup.run_until("07:30", target_hours=0.3)
    assert night.exit_code == 0 and sup.synced_hours() >= 0.3


# --- generate errors ---------------------------------------------------------------------------------------------


def test_a_generate_error_is_survived_and_its_demos_are_processed(cfg: DictConfig) -> None:
    sup, world = make(cfg)

    def publish_then_die() -> None:
        world.publish(1)
        raise GenerateError("CS2 hung")

    world.generate_script = [publish_then_die]
    night = sup.run_until("07:30", target_hours=1 / 3 + 0.01)
    assert world.kills >= 1 and sup.state.demos["d001"]["state"] == "synced"
    assert len(world.generate_hours) == 2 and night.exit_code == 0  # one error is not an alert


def test_generate_that_keeps_failing_stops_generation_and_alerts(cfg: DictConfig) -> None:
    sup, world = make(cfg)

    def die() -> None:
        raise GenerateError("CS2 hung")

    world.generate_script = [die, die, die]
    night = sup.run_until("07:30", target_hours=99)
    assert len(world.generate_hours) == 2  # supervisor.max_generate_failures
    assert "generate failing" in night.alerts and night.exit_code == EXIT_ALERT
    assert night.reason == "generate keeps failing"


def test_a_block_without_demos_counts_as_a_failure(cfg: DictConfig) -> None:
    sup, world = make(cfg, World(cfg, per_block=0))
    night = sup.run_until("07:30", target_hours=99)
    assert len(world.generate_hours) == 2 and "generate failing" in night.alerts


def test_generate_patch_guard_alerts_and_drains_without_generating_more(cfg: DictConfig) -> None:
    sup, world = make(cfg)

    def patched() -> None:
        world.publish(2)  # published before the guard tripped
        raise PatchGuardError("CS2 build is 101, expected 100.")

    world.generate_script = [lambda: None, patched]  # first block: no demo, counts as failure; second: guard
    state = sup.state
    state.mark("pre", StageState.SEGMENTED, kind="dm")
    sup.run_until("07:30", target_hours=99)
    assert len(world.generate_hours) == 2
    assert sup.state.state_of("pre") is StageState.SYNCED  # already generated demos are drained
    assert sup.state.state_of("d001") is StageState.SYNCED and sup.state.state_of("d002") is StageState.SYNCED
    alert = world.by_title("CS2 patched")[0]
    assert alert["priority"] == "high" and "build is 101" in alert["body"]
    assert sup.night.exit_code == EXIT_ALERT


# --- disk guard --------------------------------------------------------------------------------------------------


def test_generation_pauses_over_the_working_set_limit_and_resumes_after_the_drain(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    world.working = 9.5
    world.synced_working_gb = 2.0
    sup.state.mark("old", StageState.PACKED, kind="dm", frames=FRAMES)
    night = sup.run_until("07:30", target_hours=1 / 6 + 1 / 3 + 0.01)
    assert world.events[:2] == ["sync", "generate"]  # drained first, then generated
    assert world.working < 8 and night.exit_code == 0


def test_disk_guard_drains_mid_night_when_a_demo_pushes_the_working_set_over(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    world.render_working_gb = 9.0
    world.synced_working_gb = 9.0
    sup.run_until("07:30", target_hours=0.3)
    first_sync = world.events.index("sync")
    assert first_sync < world.events.index("segment d002")  # synced between the two demos, before more renders


def test_disk_guard_stalls_with_an_alert_when_nothing_can_be_drained(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    world.working = 20.0
    night = sup.run_until("07:30", target_hours=99)
    assert world.generate_hours == []
    assert "disk guard" in night.alerts and night.exit_code == EXIT_ALERT and "disk" in night.reason


def test_low_free_disk_stops_generation_and_alerts(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    world.free = 5.0
    night = sup.run_until("07:30", target_hours=99)
    assert world.generate_hours == [] and "disk almost full" in night.alerts and night.exit_code == EXIT_ALERT
    assert world.by_title("disk almost full")[0]["priority"] == "high"


def test_measure_disk_counts_the_working_set_but_not_the_demo_archive(cfg: DictConfig) -> None:
    from cs2wm.pipeline.supervisor import measure_disk

    for name, size in [("frames", 1000), ("raw", 2000), ("hdf5/dm/x.staging", 3000), ("labels", 400)]:
        path = Path(cfg.paths.root) / name / "f.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
    (Path(cfg.paths.demos) / "dm").mkdir(parents=True)
    (Path(cfg.paths.demos) / "dm" / "big.dem").write_bytes(b"x" * 10_000)
    working, free = measure_disk(cfg)
    assert working == pytest.approx(6400 / 1e9) and free > 0


# --- patch guard -------------------------------------------------------------------------------------------------


def test_first_run_records_the_build_without_a_smoke_test(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    sup.run_until("23:40", target_hours=99)
    assert world.smoke_calls == 0 and sup.state.meta[BUILD_KEY] == 100


def test_unchanged_build_skips_the_smoke_test(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    sup.state.meta[BUILD_KEY] = 100
    sup.run_until("23:40", target_hours=99)
    assert world.smoke_calls == 0


def test_changed_build_runs_the_smoke_test_before_generating_and_resumes_when_it_passes(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    sup.state.meta[BUILD_KEY] = 99
    night = sup.run_until("07:30", target_hours=0.3)
    assert world.events[0] == "smoke" and world.events[1] == "generate"
    assert sup.state.meta[BUILD_KEY] == 100 and PipelineState.load(sup.state.path).meta[BUILD_KEY] == 100
    assert world.by_title("CS2 updated") and night.exit_code == 0


def test_failed_smoke_test_stops_generation_and_rendering_and_alerts(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    sup.state.meta[BUILD_KEY] = 99
    sup.state.mark("rendered", StageState.RENDERED, kind="dm")
    sup.state.mark("waiting", StageState.SEGMENTED, kind="dm")
    world.smoke_result = "no frames captured"
    night = sup.run_until("07:30", target_hours=99)
    assert world.generate_hours == [] and world.smoke_calls == 1  # not retried every iteration
    assert sup.state.meta[BUILD_KEY] == 99  # still unverified: tomorrow tests again
    assert sup.state.state_of("waiting") is StageState.SEGMENTED  # render blocked, nothing burned an attempt
    assert sup.state.state_of("rendered") is StageState.SYNCED  # what needs no CS2 still drains
    alert = world.by_title("CS2 patched")[0]
    assert alert["priority"] == "high" and "no frames captured" in alert["body"]
    assert night.exit_code == EXIT_ALERT and sup.state.demos["waiting"].get("attempts") is None


def test_a_patch_during_the_night_is_caught_at_the_next_iteration(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    sup.state.meta[BUILD_KEY] = 100

    def patch() -> None:
        world.build = 101
        world.publish(1)

    world.generate_script = [patch]
    sup.run_until("07:30", target_hours=99)
    assert world.smoke_calls == 1 and sup.state.meta[BUILD_KEY] == 101
    assert world.events.index("smoke") > world.events.index("generate")


def _smoke_setup(tmp_path: Path) -> tuple[DictConfig, dict, Path]:
    cfg = render_cfg(tmp_path, f"paths.cs2_dir={tmp_path / 'cs2'}", "supervisor.smoke_s=0.25")
    plan = _plan(cfg)
    _write_plan(cfg, plan)
    demo = Path(cfg.paths.demos) / "dm" / f"{plan['demo_id']}.dem"
    demo.parent.mkdir(parents=True)
    demo.touch()
    return cfg, plan, demo


def test_make_smoke_plan_cuts_a_short_valid_plan(tmp_path: Path) -> None:
    cfg, plan, _ = _smoke_setup(tmp_path)
    before = json.dumps(plan, sort_keys=True)
    small = make_smoke_plan(plan, 0.25)  # 4 frames
    assert json.dumps(plan, sort_keys=True) == before  # input untouched
    (seg,) = small["segments"]
    assert seg["labeled_frames"] == 4 and seg["rendered_frames"] == 5
    assert seg["record_end_tick"] - seg["record_start_tick"] == 4 * plan["timing"]["ticks_per_frame"]
    assert small["demo_end_tick"] == seg["record_end_tick"] and small["meta"]["smoke_test"] is True
    sched = {e["action"]: e for e in small["mirv_cmd_schedule"]}
    assert sched["record_end"]["mirv_tick"] == seg["mirv_record_end_tick"]
    assert sched["record_start"]["mirv_tick"] == seg["mirv_record_start_tick"]
    assert sched["spec"]["mirv_tick"] <= sched["record_start"]["mirv_tick"]
    assert render.build_render_cfg(small, cfg)  # render accepts it


def test_make_smoke_plan_keeps_a_short_episode_whole(tmp_path: Path) -> None:
    _, plan, _ = _smoke_setup(tmp_path)
    ep = plan["segments"][0]["episodes"][0]
    assert make_smoke_plan(plan, 10)["segments"][0]["labeled_frames"] == ep["n_frames"]  # 10 s > the episode
    with pytest.raises(ValueError, match="no episode"):
        make_smoke_plan({**plan, "segments": []}, 10)


def test_render_smoke_test_renders_through_render_run_in_a_scratch_dir(tmp_path: Path, monkeypatch) -> None:
    cfg, plan, demo = _smoke_setup(tmp_path)
    state = PipelineState(Path(cfg.paths.state) / "pipeline.json")
    state.mark(plan["demo_id"], StageState.EXTRACTED, kind="dm")
    sup = Supervisor(cfg, state)
    real_run, seen = render.run, {}

    def run_with_fake_hlae(demo_path: Path, smoke_cfg: DictConfig) -> render.RenderResult:
        smoke_plan = segment.read_render_plan(Path(smoke_cfg.paths.plans) / plan["demo_id"] / "render_plan.json")
        seen.update(demo=demo_path, plan=smoke_plan, frames=smoke_cfg.paths.frames)
        return real_run(demo_path, smoke_cfg, _hooks_clock(), FakeLauncher(smoke_plan, [behaviour]))

    monkeypatch.setattr(render, "run", run_with_fake_hlae)
    behaviour = "ok"
    assert sup.render_smoke_test() is None
    assert seen["demo"] == demo and seen["plan"]["meta"]["smoke_test"] is True
    assert Path(seen["frames"]).parent == Path(cfg.paths.root) / "smoke"
    assert not (Path(cfg.paths.root) / "smoke").exists()  # scratch removed
    assert not (Path(cfg.paths.frames) / plan["demo_id"]).exists()  # the real frames dir is untouched
    assert segment.read_render_plan(Path(cfg.paths.plans) / plan["demo_id"] / "render_plan.json")["segments"]

    behaviour = "crash"
    reason = sup.render_smoke_test()
    assert reason and "HLAE exit code 1" in reason


def test_render_smoke_test_without_a_known_good_demo_passes_with_a_warning(
    cfg: DictConfig, caplog: pytest.LogCaptureFixture
) -> None:
    sup = Supervisor(cfg, PipelineState(Path(cfg.paths.state) / "pipeline.json"))
    with caplog.at_level(logging.WARNING):
        assert sup.render_smoke_test() is None
    assert "unverified" in caplog.text


# --- heartbeat and messages --------------------------------------------------------------------------------------


def test_heartbeat_is_sent_every_heartbeat_min(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    sup.run_until("04:00", target_hours=99)
    beats = world.by_title("heartbeat")
    assert 3 <= len(beats) <= 6
    assert "generated" in beats[0]["body"] and "QA pass rate" in beats[0]["body"] and "GB free" in beats[0]["body"]
    assert beats[0]["priority"] == "low"


SNAP = ntfy.Snapshot(
    generated_h=3.5,
    rendered_h=3.0,
    synced_h=2.25,
    qa_pass_rate=0.8,
    working_gb=4.2,
    free_gb=310.0,
    stage="render d012",
    counts={"pending": 2, "synced": 9, "failed": 1, "packed": 0},
)
DEADLINE = datetime(2026, 3, 2, 7, 30)


def test_format_start() -> None:
    msg = ntfy.format_start(20, DEADLINE, T0, 20250331, 0.5, 312.4, "dm")
    assert msg.title == "dcs2: night started" and msg.priority == "default"
    for text in ("Target 20 h", "profile dm", "Mon 07:30", "8h30m", "build 20250331", "0.5 GB", "312 GB free"):
        assert text in msg.body, text
    assert "unknown" in ntfy.format_start(20, DEADLINE, T0, None, 0, 1, "dm").body


def test_format_heartbeat() -> None:
    msg = ntfy.format_heartbeat(SNAP, 20, T0 + timedelta(hours=2), DEADLINE)
    for text in ("3.50 generated", "3.00 rendered", "2.25/20 synced", "QA pass rate 80%", "4.2 GB", "310 GB free"):
        assert text in msg.body, text
    assert "render d012" in msg.body and "9 synced" in msg.body and "0 packed" not in msg.body
    assert "6h30m to the deadline" in msg.body and msg.priority == "low"
    assert "n/a" in ntfy.format_heartbeat(ntfy.Snapshot(0, 0, 0, None, 0, 1), 1, T0, DEADLINE).body


def test_format_alert() -> None:
    msg = ntfy.format_alert("CS2 patched", "Build 1 -> 2.")
    assert (msg.title, msg.body, msg.priority) == ("dcs2 alert: CS2 patched", "Build 1 -> 2.", "high")


def test_format_summary() -> None:
    summary = ntfy.NightSummary(
        snapshot=SNAP,
        target_hours=20,
        started=T0,
        ended=DEADLINE,
        reason="deadline reached",
        failures=[("render", "frame-count mismatch", 2), ("segment", "no episodes", 1)],
        stage_wall_s={"generate": 7200, "render": 5400},
        stage_runs={"generate": 2, "render": 9},
        alerts=["CS2 patched"],
    )
    msg = ntfy.format_summary(summary)
    assert msg.title == "dcs2: morning summary (needs attention)" and msg.priority == "high"
    for text in (
        "deadline reached",
        "Synced 2.25 h of 20 h",
        "render: frame-count mismatch (x2)",
        "segment: no episodes",
    ):
        assert text in msg.body, text
    assert "generate 2h00m (2 runs)" in msg.body and "render 1h30m (9 runs)" in msg.body
    assert "Alerts: CS2 patched" in msg.body
    quiet = ntfy.format_summary(
        ntfy.NightSummary(snapshot=SNAP, target_hours=20, started=T0, ended=DEADLINE, reason="target reached")
    )
    assert quiet.title == "dcs2: morning summary" and quiet.priority == "default" and "No failed demos." in quiet.body


def test_night_messages_carry_real_numbers(cfg: DictConfig) -> None:
    sup, world = make(cfg)
    world.fail[("pack", "d001")] = 99
    sup.run_until("07:30", target_hours=0.3)
    summary = world.by_title("morning summary")[0]["body"]
    assert "target reached" in summary and "pack: pack broke (x1)" in summary
    assert "QA pass rate 50%" in summary and re.search(r"render \d+m \(\d+ runs\)", summary)
    assert world.by_title("night started")[0]["body"].count("build 100") == 1


def test_notify_failures_and_the_topic_never_reach_the_log(caplog: pytest.LogCaptureFixture, monkeypatch) -> None:
    def post(url: str, **kwargs: Any) -> None:
        raise requests.ConnectionError(f"HTTPSConnectionPool: failed to reach {url}")

    monkeypatch.setattr(requests, "post", post)
    with caplog.at_level(logging.DEBUG):
        ntfy.notify("hello", server="https://ntfy.example", topic="s3cret-topic-xyz")
    assert "s3cret-topic-xyz" not in caplog.text and "ConnectionError" in caplog.text


def test_a_failing_notifier_does_not_stop_the_night(cfg: DictConfig) -> None:
    sup, _ = make(cfg)

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("no network")

    sup._notify = boom
    sup.run_until("07:30", target_hours=0.3)
    assert sup.synced_hours() >= 0.3


# --- state, config, cli, scripts ---------------------------------------------------------------------------------


def test_state_meta_round_trips_and_old_files_load(tmp_path: Path) -> None:
    state = PipelineState(tmp_path / "pipeline.json")
    state.meta[BUILD_KEY] = 5
    state.save()
    assert PipelineState.load(state.path).meta == {BUILD_KEY: 5}
    state.path.write_text(json.dumps({"schema_version": 1, "demos": {}}), encoding="utf-8")  # written before #20
    assert PipelineState.load(state.path).meta == {}


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ("supervisor.max_working_gb=0", "max_working_gb"),
        ("supervisor.min_free_gb=-1", "min_free_gb"),
        ("supervisor.max_attempts=0", "max_attempts"),
        ("supervisor.backoff_s=-5", "backoff_s"),
        ("supervisor.generate_block_min=abc", "generate_block_min"),
        ("supervisor.alert_failed_in_row=0", "alert_failed_in_row"),
        ("supervisor.drain_min=null", "drain_min"),
    ],
)
def test_supervisor_config_is_validated(override: str, fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        load_config(overrides=[override])


def test_default_supervisor_config() -> None:
    sup = load_config().supervisor
    assert (sup.generate_block_min, sup.max_working_gb, sup.max_attempts, sup.heartbeat_min) == (60, 8, 3, 60)


def test_cli_supervise_returns_the_night_exit_code(cfg: DictConfig, monkeypatch) -> None:
    calls = []

    def fake_run_until(self: Supervisor, until: str, target: float, profile: str) -> Night:
        calls.append((until, target, profile))
        night = Night(T0, DEADLINE, target, profile)
        night.alerts.append("CS2 patched")
        return night

    monkeypatch.setattr(Supervisor, "run_until", fake_run_until)
    assert cli.main(["supervise", "--until", "06:15", "--target-hours", "4", "--profile", "smoke"]) == EXIT_ALERT
    assert calls == [("06:15", 4.0, "smoke")]


def test_cli_supervise_rejects_a_bad_deadline(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="HH:MM"):
        cli.main(["supervise", "--until", "7pm"])


SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "windows"


def test_install_task_script_registers_the_interactive_task() -> None:
    text = (SCRIPTS / "install_task.ps1").read_text(encoding="utf-8")
    for expected in (
        "dcs2-supervisor",
        "Register-ScheduledTask",
        "New-ScheduledTaskTrigger -Daily",
        "-LogonType Interactive",
        "New-ScheduledTaskSettingsSet",
        "-ExecutionTimeLimit",
        "[switch]$Force",
        "[switch]$Remove",
        "Unregister-ScheduledTask",
        "run_supervisor.ps1",
        "-Until",
        "-TargetHours",
        "-Profile",
    ):
        assert expected in text, expected
    assert "RunLevel Highest" not in text and "-RunLevel" not in text  # no elevation needed


def test_run_supervisor_script_runs_cs2wm_supervise_and_propagates_the_exit_code() -> None:
    text = (SCRIPTS / "run_supervisor.ps1").read_text(encoding="utf-8")
    for expected in (
        r"C:\dcs2\venv",
        "Activate.ps1",
        "cs2wm supervise",
        "--until",
        "--target-hours",
        "--profile",
        "$env:DCS2_ROOT",
        "supervisor-",
        "Tee-Object",
        "exit $",
        "Set-Location",
    ):
        assert expected in text, expected


@pytest.mark.parametrize("name", ["install_task.ps1", "run_supervisor.ps1"])
def test_scripts_are_plain_ascii_and_balanced_and_have_no_stubs(name: str) -> None:
    raw = (SCRIPTS / name).read_bytes()
    raw.decode("ascii")  # PowerShell 5.1 reads BOM-less files as ANSI: keep them ASCII
    text = raw.decode("ascii")
    assert "TODO(#20)" not in text and 'throw "TODO' not in text
    code = re.sub(r"<#.*?#>", "", text, flags=re.DOTALL)
    code = "\n".join(line.split("#")[0] if "'" not in line and '"' not in line else line for line in code.splitlines())
    for open_, close in ("{}", "()", "[]"):
        assert code.count(open_) == code.count(close), (name, open_)
    assert text.count('"') % 2 == 0, name


def test_kill_by_name_uses_taskkill_on_windows_only() -> None:
    from cs2wm.pipeline import proc

    calls: list[list[str]] = []
    proc.kill_by_name(lambda cmd, **kw: calls.append(cmd), True, ["cs2.exe", "HLAE.exe"])
    assert calls == [["taskkill", "/T", "/F", "/IM", "cs2.exe"], ["taskkill", "/T", "/F", "/IM", "HLAE.exe"]]
    calls.clear()
    proc.kill_by_name(lambda cmd, **kw: calls.append(cmd), False, ["cs2.exe"])
    assert calls == []
