"""Tests for the persistent pipeline state, the stage runner, status and qa-summary (#19, #20)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from cs2wm.config import load_config
from cs2wm.pipeline import qa_summary
from cs2wm.pipeline.supervisor import PipelineState, StageFailure, StageState, StateError, Supervisor


@pytest.fixture
def cfg(tmp_path: Path):
    return load_config(overrides=[f"paths.root={tmp_path / 'data'}"])


@pytest.fixture
def state(tmp_path: Path) -> PipelineState:
    return PipelineState(tmp_path / "state" / "pipeline.json")


def touch_demo(cfg, kind: str, demo_id: str) -> None:
    path = Path(cfg.paths.demos) / kind / f"{demo_id}.dem"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


# --- persistence -------------------------------------------------------------------------------------------------


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    state = PipelineState.load(tmp_path / "nope" / "pipeline.json")
    assert state.demos == {} and state.path == tmp_path / "nope" / "pipeline.json"


def test_save_load_round_trip(state: PipelineState) -> None:
    state.mark("d1", StageState.SEGMENTED, stage_wall_s=1.5, kind="dm", demo_s=600.0)
    state.save()
    loaded = PipelineState.load(state.path)
    assert loaded.demos == state.demos
    rec = loaded.demos["d1"]
    assert (rec["state"], rec["kind"], rec["demo_s"]) == ("segmented", "dm", 600.0)
    assert rec["stages"]["segment"]["wall_s"] == 1.5 and "at" in rec["stages"]["segment"]


def test_save_is_atomic(state: PipelineState, monkeypatch: pytest.MonkeyPatch) -> None:
    state.mark("d1", StageState.PENDING, kind="dm")
    state.save()
    good = state.path.read_text()
    state.mark("d1", StageState.SEGMENTED)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        state.save()
    assert state.path.read_text() == good  # the old file is intact
    assert sorted(p.name for p in state.path.parent.iterdir()) == ["pipeline.json", "pipeline.json.bak"]  # no temp file


def test_load_rejects_a_corrupt_or_foreign_file(tmp_path: Path) -> None:
    bad = tmp_path / "p.json"
    bad.write_text("{not json")
    with pytest.raises(StateError, match=r"p\.json is corrupt.*restore it from p\.json\.bak"):
        PipelineState.load(bad)
    bad.write_text('{"schema_version": 99}')
    with pytest.raises(StateError, match="schema.*p\\.json\\.bak"):
        PipelineState.load(bad)


def test_save_keeps_the_previous_state_as_bak(state: PipelineState) -> None:
    state.mark("d1", StageState.PENDING)
    state.save()
    state.mark("d1", StageState.SEGMENTED)
    state.save()
    bak = PipelineState.load(state.path.with_name("pipeline.json.bak"))
    assert bak.demos["d1"]["state"] == "pending" and PipelineState.load(state.path).demos["d1"]["state"] == "segmented"


def test_cli_reports_a_corrupt_state_file_in_one_line(cfg, monkeypatch) -> None:
    from cs2wm import cli

    monkeypatch.setattr(cli, "_cfg", lambda args: cfg)
    path = Path(cfg.paths.state) / "pipeline.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version": 1, "demos": {"a"')  # truncated
    with pytest.raises(SystemExit) as exc:
        cli.main(["status"])
    msg = str(exc.value)
    assert "\n" not in msg and str(path) in msg and "pipeline.json.bak" in msg


# --- transitions -------------------------------------------------------------------------------------------------


def test_mark_failed_needs_stage_and_reason_and_success_clears_it(state: PipelineState) -> None:
    with pytest.raises(ValueError, match="stage and a reason"):
        state.mark("d1", StageState.FAILED, "boom")
    with pytest.raises(ValueError, match="unknown stage"):
        state.mark("d1", "failed", "boom", stage="generate")
    state.mark("d1", "failed", "boom", 3.0, stage="render")
    rec = state.demos["d1"]
    assert (rec["state"], rec["failed_stage"], rec["reason"], rec["stages"]["render"]["wall_s"]) == (
        "failed",
        "render",
        "boom",
        3.0,
    )
    state.mark("d1", StageState.RENDERED, stage_wall_s=4.0)
    assert state.demos["d1"]["state"] == "rendered"
    assert "failed_stage" not in state.demos["d1"] and "reason" not in state.demos["d1"]
    assert state.state_of("d1") is StageState.RENDERED and state.state_of("zzz") is None


# --- discover / pending_for --------------------------------------------------------------------------------------


def test_discover_adds_unknown_demos_of_each_kind_and_skips_rejected(cfg, state: PipelineState) -> None:
    for kind, demo_id in [("dm", "b"), ("dm", "a"), ("smoke", "c"), ("_rejected", "bad"), ("other", "x")]:
        touch_demo(cfg, kind, demo_id)
    state.mark("a", StageState.PACKED, kind="dm")
    assert state.discover(cfg) == ["b", "c"]
    assert {d: (r["state"], r["kind"]) for d, r in state.demos.items()} == {
        "a": ("packed", "dm"),
        "b": ("pending", "dm"),
        "c": ("pending", "smoke"),
    }
    assert state.discover(cfg) == []  # idempotent


def test_discover_without_a_demos_folder(cfg, state: PipelineState) -> None:
    assert state.discover(cfg) == []


def test_pending_for_is_sorted_and_retries_only_on_request(state: PipelineState) -> None:
    state.mark("c", StageState.PENDING)
    state.mark("a", StageState.PENDING)
    state.mark("b", StageState.SEGMENTED)
    state.mark("f1", "failed", "x", stage="segment")
    state.mark("f2", "failed", "x", stage="render")
    assert state.pending_for("segment") == ["a", "c"]
    assert state.pending_for("segment", retry_failed=True) == ["a", "c", "f1"]
    assert state.pending_for("render") == ["b"]
    assert state.pending_for("render", retry_failed=True) == ["b", "f2"]
    assert state.pending_for("sync") == []


# --- run_stage ---------------------------------------------------------------------------------------------------


def fake_stage(monkeypatch: pytest.MonkeyPatch, fn) -> None:
    monkeypatch.setattr(Supervisor, "_stage_fn", lambda self, stage, **options: fn)


def test_failure_marks_the_demo_and_continues_with_the_next(cfg, state: PipelineState, monkeypatch) -> None:
    def fn(demo_id: str) -> dict:
        if demo_id == "b":
            raise ValueError("bad plan")
        if demo_id == "c":
            raise StageFailure("plan has no episodes")
        return {"episodes": 2}

    fake_stage(monkeypatch, fn)
    for d in "abcd":
        state.mark(d, StageState.PENDING, kind="dm")
    ticks = iter(range(0, 100, 5))
    report = Supervisor(cfg, state).run_stage("segment", state.pending_for("segment"), clock=lambda: next(ticks))
    assert report.done == ["a", "d"]
    assert report.failed == {"b": "ValueError: bad plan", "c": "plan has no episodes"}
    saved = PipelineState.load(state.path).demos
    assert [saved[d]["state"] for d in "abcd"] == ["segmented", "failed", "failed", "segmented"]
    assert (saved["b"]["failed_stage"], saved["b"]["reason"]) == ("segment", "ValueError: bad plan")
    assert saved["a"]["episodes"] == 2 and saved["a"]["stages"]["segment"]["wall_s"] == 5.0


def test_failed_demos_are_retried_only_with_the_flag(cfg, state: PipelineState, monkeypatch) -> None:
    calls: list[str] = []
    fake_stage(monkeypatch, lambda d: calls.append(d) or {})
    state.mark("a", "failed", "old", stage="segment")
    state.mark("b", StageState.PENDING)
    sup = Supervisor(cfg, state)
    sup.run_stage("segment", state.pending_for("segment"))
    assert calls == ["b"] and state.demos["a"]["state"] == "failed"
    sup.run_stage("segment", state.pending_for("segment", retry_failed=True))
    assert calls == ["b", "a"] and state.demos["a"]["state"] == "segmented"
    assert "reason" not in state.demos["a"]


def test_state_survives_a_crash_mid_run_and_the_run_resumes(cfg, state: PipelineState, monkeypatch) -> None:
    def crash_on_c(demo_id: str) -> dict:
        if demo_id == "c":
            raise KeyboardInterrupt  # not an Exception: the run stops, like a killed process
        return {}

    fake_stage(monkeypatch, crash_on_c)
    for d in "abcd":
        state.mark(d, StageState.PENDING)
    with pytest.raises(KeyboardInterrupt):
        Supervisor(cfg, state).run_stage("segment", state.pending_for("segment"))

    resumed = PipelineState.load(state.path)  # a new process
    assert [resumed.demos[d]["state"] for d in "ab"] == ["segmented", "segmented"]
    assert resumed.pending_for("segment") == ["c", "d"]  # exactly the unfinished ones; nothing is redone
    calls: list[str] = []
    fake_stage(monkeypatch, lambda d: calls.append(d) or {})
    Supervisor(cfg, resumed).run_stage("segment", resumed.pending_for("segment"))
    assert calls == ["c", "d"] and {r["state"] for r in resumed.demos.values()} == {"segmented"}


def test_demo_path_prefers_an_explicit_path_then_the_recorded_kind(cfg, state: PipelineState, tmp_path: Path) -> None:
    touch_demo(cfg, "smoke", "x")
    sup = Supervisor(cfg, state)
    assert sup.demo_path("x") == Path(cfg.paths.demos) / "smoke" / "x.dem"  # found by search
    sup.explicit_paths["x"] = tmp_path / "elsewhere.dem"
    assert sup.demo_path("x") == tmp_path / "elsewhere.dem"
    with pytest.raises(FileNotFoundError):
        sup.demo_path("missing")


# --- sync marking ------------------------------------------------------------------------------------------------


def test_run_sync_marks_only_fully_verified_demos(cfg, state: PipelineState, monkeypatch) -> None:
    from cs2wm.pipeline import sync
    from cs2wm.pipeline.sync import DemoResult, SyncSummary

    summary = SyncSummary(
        demos=[
            DemoResult("ok", "dm", pushed=True, expected=2, verified=2),
            DemoResult("bad", "dm", pushed=True, expected=2, verified=1, errors=["remote file is missing"]),
        ]
    )
    monkeypatch.setattr(sync, "run", lambda cfg, transport=None: summary)
    state.mark("ok", StageState.PACKED, kind="dm", packed_episodes=1)
    state.mark("bad", StageState.PACKED, kind="dm", packed_episodes=1)
    state.mark("none", StageState.PACKED, kind="dm", packed_episodes=0)  # nothing passed QA: no manifest
    ticks = iter([0.0, 9.0])
    Supervisor(cfg, state).run_sync(clock=lambda: next(ticks))
    got = {d: r["state"] for d, r in PipelineState.load(state.path).demos.items()}
    assert got == {"ok": "synced", "bad": "packed", "none": "synced"}
    assert state.demos["ok"]["stages"]["sync"]["wall_s"] == 4.5  # 9 s shared by the two demos marked


def test_run_sync_splits_wall_time_by_packed_frames(cfg, state: PipelineState, monkeypatch) -> None:
    from cs2wm.pipeline import sync
    from cs2wm.pipeline.sync import DemoResult, SyncSummary

    summary = SyncSummary(demos=[DemoResult("a", "dm", True, 1, 1), DemoResult("b", "dm", True, 1, 1)])
    monkeypatch.setattr(sync, "run", lambda cfg, transport=None: summary)
    state.mark("a", StageState.PACKED, kind="dm", frames=3000, packed_episodes=1)
    state.mark("b", StageState.PACKED, kind="dm", frames=1000, packed_episodes=1)
    ticks = iter([0.0, 8.0])
    Supervisor(cfg, state).run_sync(clock=lambda: next(ticks))
    assert (state.demos["a"]["stages"]["sync"]["wall_s"], state.demos["b"]["stages"]["sync"]["wall_s"]) == (6.0, 2.0)


def test_run_sync_does_not_mark_empty_demos_when_a_push_failed(cfg, state: PipelineState, monkeypatch) -> None:
    from cs2wm.pipeline import sync
    from cs2wm.pipeline.sync import SyncSummary

    monkeypatch.setattr(sync, "run", lambda cfg, transport=None: SyncSummary(push_errors=["rsync demos failed"]))
    state.mark("none", StageState.PACKED, packed_episodes=0)
    Supervisor(cfg, state).run_sync()
    assert state.demos["none"]["state"] == "packed"


# --- status ------------------------------------------------------------------------------------------------------


def test_status_table_rows_and_totals(cfg, state: PipelineState) -> None:
    # Two demos of 30 min: segment took 6 s + 12 s, render 900 s + 1800 s; 16 fps => 57600 frames = 1 h.
    state.mark("d1", StageState.SEGMENTED, stage_wall_s=6, kind="dm", demo_s=1800)
    state.mark("d1", StageState.RENDERED, stage_wall_s=900)
    state.mark("d1", StageState.EXTRACTED, stage_wall_s=30)
    state.mark("d1", StageState.PACKED, stage_wall_s=60, frames=28800)
    state.mark("d1", StageState.SYNCED, stage_wall_s=10)
    state.mark("d2", StageState.SEGMENTED, stage_wall_s=12, kind="dm", demo_s=1800)
    state.mark("d2", StageState.RENDERED, stage_wall_s=1800)
    state.mark("d2", StageState.EXTRACTED, stage_wall_s=30)
    state.mark("d2", StageState.PACKED, stage_wall_s=60, frames=28800)
    state.mark("d3", "failed", "frame-count mismatch", 5.0, stage="render", kind="smoke")
    state.mark("d4", StageState.PENDING, kind="human")
    text = Supervisor(cfg, state).status_table()
    lines = text.splitlines()
    assert lines[0].split() == ["demo", "kind", "state", "segment", "render", "extract", "pack", "sync"]
    assert lines[1].split() == ["d1", "dm", "synced", "6.0", "900.0", "30.0", "60.0", "10.0"]
    assert lines[2].split()[:3] == ["d2", "dm", "packed"]
    assert lines[3].endswith("[render] frame-count mismatch")
    assert "demos: 1 pending, 0 segmented, 0 rendered, 0 extracted, 1 packed, 1 synced, 1 failed (4 total)" in text
    assert "QA-passing hours packed: 1.000 (57600 frames)" in text
    # render: 2700 s over 1 demo-hour; segment: 18 s over 1 demo-hour; sync: only d1 timed (10 s over 0.5 h)
    assert (
        "wall s per demo-hour (failed demos excluded): segment 18, render 2700, extract 60, pack 120, sync 20" in text
    )


def test_status_per_hour_figures_leave_out_failed_demos(cfg, state: PipelineState) -> None:
    state.mark("ok", StageState.SEGMENTED, stage_wall_s=10, kind="dm", demo_s=3600)
    state.mark("bad", StageState.SEGMENTED, stage_wall_s=1000, kind="dm", demo_s=3600)
    state.mark("bad", "failed", "boom", 5.0, stage="render")
    assert "segment 10" in Supervisor(cfg, state).status_table()


def test_status_table_empty(cfg, state: PipelineState) -> None:
    text = Supervisor(cfg, state).status_table()
    assert "(no demos)" in text and "QA-passing hours packed: 0.000" in text and "excluded): n/a" in text


# --- qa-summary --------------------------------------------------------------------------------------------------


def write_qa(cfg, demo_id: str, rows: list[tuple[str, str, int, list[str]]], age_s: float = 0.0) -> None:
    """Rows are ``(episode_id, kind, n_frames, failed reasons)``."""
    path = Path(cfg.paths.qa) / f"{demo_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ep, kind, n, reasons in rows:
            rec = {"episode_id": ep, "demo_id": demo_id, "kind": kind, "n_frames": n, "passed": not reasons}
            f.write(json.dumps({**rec, "reasons": reasons, "stats": {}}) + "\n")
    t = time.time() - age_s
    os.utime(path, (t, t))


def test_qa_summary_counts_rates_reasons_and_hours(cfg) -> None:
    write_qa(
        cfg,
        "d1",
        [
            ("e0", "dm", 1000, []),
            ("e1", "dm", 800, ["fire_recall 0.900 < 0.98", "mouse_saturation_frac 0.300 > 0.02"]),
            ("e2", "dm", 200, []),
            ("e3", "dm", 64, ["missing_tick_frac 0.500 > 0.01"]),
        ],
    )
    write_qa(cfg, "d2", [("s0", "smoke", 57600, [])])
    s = qa_summary.summarize(cfg)
    dm = s.kinds["dm"]
    assert (dm.episodes, dm.passed, dm.frames, dm.passed_frames) == (4, 2, 2064, 1200)
    assert dm.pass_rate == 0.5 and dm.failures == {"fire_recall": 1, "mouse_saturation_frac": 1, "missing_tick_frac": 1}
    assert s.demos == 2 and s.episodes == 5 and s.drop_rate == pytest.approx(0.4)
    assert qa_summary.summarize(cfg, kind="smoke").kinds.keys() == {"smoke"}
    assert qa_summary.summarize(cfg, kind="smoke").demos == 1  # d1 has no smoke episode: not counted
    text = qa_summary.format_summary(s, 16.0)
    assert "dm: 2/4 episodes passed (50.0%); 1200/2064 frames; 0.021 h passing" in text
    assert "failed fire_recall: 1" in text and "smoke: 1/1 episodes passed (100.0%)" in text
    assert "passing hours (all kinds): 1.021" in text
    assert "drop rate: 40.0% (limit < 20%): ABOVE LIMIT" in text
    assert "per-demo snapshot, rewritten when the demo is extracted again" in text


def test_qa_summary_drop_rate_below_the_limit_is_ok(cfg) -> None:
    write_qa(cfg, "d1", [(f"e{i}", "dm", 100, []) for i in range(9)] + [("x", "dm", 100, ["nan_frames 3.000 > 0"])])
    assert "drop rate: 10.0% (limit < 20%): OK" in qa_summary.format_summary(qa_summary.summarize(cfg), 16.0)


def test_qa_summary_since_filters_by_file_age_and_skips_bad_lines(cfg) -> None:
    write_qa(cfg, "old", [("e0", "dm", 100, [])], age_s=3 * 86400)
    write_qa(cfg, "new", [("e1", "dm", 50, [])], age_s=60)
    with open(Path(cfg.paths.qa) / "new.jsonl", "a") as f:
        f.write("garbage\n\n")
    s = qa_summary.summarize(cfg, since_s=qa_summary.parse_since("24h"))
    assert (s.demos, s.episodes, s.bad_lines) == (1, 1, 1)
    assert qa_summary.summarize(cfg).episodes == 2


def test_qa_summary_with_no_data(cfg) -> None:
    s = qa_summary.summarize(cfg)
    assert s.episodes == 0 and s.drop_rate == 0.0
    assert "0 episode(s)" in qa_summary.format_summary(s, 16.0)


@pytest.mark.parametrize(
    ("text", "seconds"), [("90s", 90), ("30m", 1800), ("24h", 86400), ("2d", 172800), ("1.5h", 5400)]
)
def test_parse_since(text: str, seconds: float) -> None:
    assert qa_summary.parse_since(text) == seconds


@pytest.mark.parametrize("text", ["", "h", "24", "-1h", "1w", "1 h"])
def test_parse_since_rejects_garbage(text: str) -> None:
    with pytest.raises(ValueError, match="invalid --since"):
        qa_summary.parse_since(text)
