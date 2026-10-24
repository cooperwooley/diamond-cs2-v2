"""End-to-end wiring of the per-demo stages (#11, #12, #19): segment -> render -> extract -> pack -> sync.

Only the outside world is faked: the demoparser2 parser, the HLAE launcher (it writes TGA takes the way HLAE would,
and the real ``render.run`` downscales them and writes ``render.json``) and the Mill (a local directory).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from cs2wm import cli, demo
from cs2wm.actions import ActionSpec
from cs2wm.config import load_config
from cs2wm.pipeline import extract_actions, pack, render, segment
from cs2wm.pipeline.pack import PackError, read_episode, to_low_res
from cs2wm.pipeline.supervisor import PipelineState, Supervisor
from tests.test_demo import FakeParser
from tests.test_render import FakeLauncher, _hooks_clock
from tests.test_sync import LocalTransport

DEMO = "dm-0001"
END = 200  # last tick of the fake demo
YAW_PER_TICK = -0.1375  # 4 ticks per frame => +10 mouse units to the right on every frame
SPEC = ActionSpec.from_config(load_config())


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    return tmp_path / "mill" / "dcs2"


@pytest.fixture
def cfg(tmp_path: Path, remote: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = load_config(
        overrides=[
            f"paths.root={tmp_path / 'data'}",
            f"paths.cs2_dir={tmp_path / 'cs2'}",
            f"sync.remote_root={remote}",
            "paths.hlae_dir=C:/hlae",
            "segment.settle_ticks=10",
            "render.tick_offset=0",
            "episodes.min_frames=4",
            "episodes.max_frames=20",
            "pack.fullres_mod=1",
        ]
    )
    monkeypatch.setattr(cli, "_cfg", lambda args: cfg)
    return cfg


def player_ticks(user_id: int, name: str, ticks: range, yaw_per_tick: float) -> pd.DataFrame:
    t = np.asarray(ticks)
    return pd.DataFrame(
        {
            "tick": t,
            "user_id": user_id,
            "name": name,
            "steamid": 0,
            "entity_id": 10 + user_id,
            "X": t * (250.0 / 64.0),  # 250 u/s straight ahead at yaw 0: W
            "Y": 0.0,
            "Z": 0.0,
            "yaw": yaw_per_tick * t,
            "pitch": 0.0,
            "is_airborne": False,
            "duck_amount": 0.0,
            "active_weapon_name": "AK-47",
        }
    )


def user_events(rows: list[tuple[int, int]], **extra) -> pd.DataFrame:
    """Event table in demoparser2's layout, from ``(tick, user_id)`` rows."""
    return pd.DataFrame({"tick": [r[0] for r in rows], "user_user_id": [r[1] for r in rows], **extra})


def make_parser(drop: range | None = None, fire: tuple[int, ...] = ()) -> FakeParser:
    """Bot 1 (the target, alive the whole demo) and bot 2 (dies at tick 100, moves differently).

    Args:
        drop: Ticks of bot 1 that are missing from the tick table (provokes a QA failure).
        fire: Ticks at which bot 1 fires; bot 2 fires at every tick in ``fire`` + 1 (must not leak into bot 1).
    """
    one = player_ticks(1, "Bot A", range(END + 1), YAW_PER_TICK)
    if drop is not None:
        one = one[~one["tick"].isin(list(drop))]
    two = player_ticks(2, "Bot B", range(END + 1), 0.3)
    spawn = user_events([(0, 1), (0, 2)], user_name=["Bot A", "Bot B"], user_steamid=[0, 0])
    death = user_events([(100, 2)], user_name=["Bot B"], user_steamid=[0])
    shots = [(t, 1) for t in fire] + [(t + 1, 2) for t in fire]
    return FakeParser(
        ticks=pd.concat([one, two], ignore_index=True),
        events={
            "player_spawn": spawn,
            "player_death": death,
            "player_disconnect": user_events([], user_name=[], user_steamid=[]),
            "weapon_fire": user_events(shots, weapon=["weapon_ak47"] * len(shots)),
        },
        header={"map_name": "de_dust2"},
    )


@pytest.fixture
def use_parser(monkeypatch: pytest.MonkeyPatch):
    def install(parser: FakeParser) -> FakeParser:
        monkeypatch.setattr(demo, "_make_parser", lambda path: parser)
        return parser

    return install


@pytest.fixture
def fake_hlae(monkeypatch: pytest.MonkeyPatch):
    """``render.run`` with a fake HLAE that captures the segment frames as raw TGA takes."""
    real_run = render.run

    def run(demo_path, cfg):
        plan = segment.read_render_plan(Path(cfg.paths.plans) / Path(demo_path).stem / "render_plan.json")
        return real_run(demo_path, cfg, _hooks_clock(), FakeLauncher(plan, ["ok"]))

    monkeypatch.setattr(render, "run", run)


def add_demo(cfg, demo_id: str = DEMO, kind: str = "dm") -> Path:
    path = Path(cfg.paths.demos) / kind / f"{demo_id}.dem"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(demo_id.encode() * 50)
    path.with_suffix(".json").write_text(
        json.dumps({"demo_id": demo_id, "cs2_build": 4242, "cs2wm_git_sha": "deadbeef"}), encoding="utf-8"
    )
    return path


def state_of(cfg, demo_id: str = DEMO) -> dict:
    return PipelineState.load(Path(cfg.paths.state) / "pipeline.json").demos[demo_id]


def read_manifest(cfg, kind: str = "dm", demo_id: str = DEMO) -> list[dict]:
    path = Path(cfg.paths.hdf5) / kind / demo_id / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_manifest(cfg, kind: str = "dm", demo_id: str = DEMO) -> None:
    """The rule of ``scripts/verify_manifests.py``: every record's sha256 matches its file."""
    hdf5, fullres = Path(cfg.paths.hdf5) / kind / demo_id, Path(cfg.paths.hdf5_fullres) / kind / demo_id
    for rec in read_manifest(cfg, kind, demo_id):
        name = rec["file"]
        path = fullres / name[len("fullres/") :] if name.startswith("fullres/") else hdf5 / name
        assert pack.sha256_file(path) == rec["sha256"], name


def run_to_rendered(cfg, use_parser, fake_hlae) -> None:
    add_demo(cfg)
    use_parser(make_parser())
    assert cli.main(["segment", "--all-pending"]) == 0
    assert cli.main(["render", "--all-pending"]) == 0


def run_to_extracted(cfg, use_parser, fake_hlae, fire: tuple[int, ...] = (13, 14, 120)) -> list[dict]:
    """Segment and render, then extract with bot 1 missing the ticks of the last (5-frame) episode."""
    run_to_rendered(cfg, use_parser, fake_hlae)
    eps = segment.episodes_from_plan(segment.read_render_plan(Path(cfg.paths.plans) / DEMO / "render_plan.json"))
    last = eps[-1]
    use_parser(make_parser(drop=range(last.start_tick + 1, last.end_tick), fire=fire))
    assert cli.main(["extract", "--all-pending"]) == 0
    return [json.loads(line) for line in (Path(cfg.paths.qa) / f"{DEMO}.jsonl").read_text().splitlines()]


# --- the full pipeline -------------------------------------------------------------------------------------------


def test_end_to_end_segment_render_extract_pack_sync(cfg, remote, use_parser, fake_hlae, capsys) -> None:
    qa = run_to_extracted(cfg, use_parser, fake_hlae)
    assert [(r["n_frames"], r["passed"]) for r in qa] == [(20, True), (20, True), (5, False)]
    assert state_of(cfg)["state"] == "extracted"

    assert cli.main(["pack", "--all-pending"]) == 0
    manifest = read_manifest(cfg)
    kind_demo = f"dm-{DEMO}"
    assert sorted(r["file"] for r in manifest) == sorted(
        [f"{kind_demo}-s000-c0{c}.hdf5" for c in (0, 1)] + [f"fullres/{kind_demo}-s000-c0{c}.hdf5" for c in (0, 1)]
    )
    assert all(r["frames"] == 20 for r in manifest)  # the QA-failing episode (c02) is not packed
    assert not list((Path(cfg.paths.hdf5) / "dm" / DEMO).glob("*c02*"))
    verify_manifest(cfg)
    assert state_of(cfg)["state"] == "packed" and state_of(cfg)["frames"] == 40
    assert not (Path(cfg.paths.frames) / DEMO).exists() and not (Path(cfg.paths.labels) / DEMO).exists()

    # sync verifies against the manifest, deletes the local HDF5 and the state ends synced
    summary = Supervisor(cfg, PipelineState.load(Path(cfg.paths.state) / "pipeline.json")).run_sync(LocalTransport())
    assert summary.ok and summary.by_demo[DEMO].synced
    assert state_of(cfg)["state"] == "synced"
    assert not list((Path(cfg.paths.hdf5) / "dm" / DEMO).glob("*.hdf5"))
    assert (remote / "hdf5" / "dm" / DEMO / f"dm-{DEMO}-s000-c00.hdf5").is_file()
    assert (remote / "hdf5_fullres" / "dm" / DEMO / f"dm-{DEMO}-s000-c01.hdf5").is_file()
    assert (remote / "qa" / f"{DEMO}.jsonl").is_file()

    capsys.readouterr()
    assert cli.main(["status"]) == 0
    status = capsys.readouterr().out
    assert DEMO in status and "synced" in status and "1 synced" in status
    assert f"QA-passing hours packed: {40 / 16 / 3600:.3f} (40 frames)" in status
    assert "wall s per demo-hour (failed demos excluded):" in status and "n/a" not in status

    assert cli.main(["qa-summary"]) == 0
    text = capsys.readouterr().out
    assert "1 demo(s), 3 episode(s)" in text
    assert "dm: 2/3 episodes passed (66.7%); 40/45 frames" in text
    assert "failed missing_tick_frac: 1" in text
    assert "drop rate: 33.3% (limit < 20%): ABOVE LIMIT" in text
    assert cli.main(["qa-summary", "--kind", "smoke", "--since", "24h"]) == 0
    assert "0 episode(s)" in capsys.readouterr().out


def test_stages_log_one_line_per_demo_with_wall_time(cfg, use_parser, fake_hlae) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    log = (Path(cfg.paths.logs) / "cs2wm.log").read_text()
    for stage in ("segment", "render", "extract"):
        assert f"{stage} {DEMO}: ok in " in log and " s" in log


def test_labels_and_frames_are_aligned_in_the_packed_files(cfg, use_parser, fake_hlae) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    rendered = json.loads((Path(cfg.paths.frames) / DEMO / "render.json").read_text())
    first = {e["episode_id"]: e["first_frame"] for e in rendered["segments"][0]["episodes"]}
    ep1 = f"dm-{DEMO}-s000-c01"
    src = [cv2.imread(str(Path(cfg.paths.frames) / DEMO / "seg000" / f"{first[ep1] + i:06d}.png")) for i in range(20)]
    labels = np.load(Path(cfg.paths.labels) / DEMO / f"{ep1}.npy")
    assert labels.shape == (20, 51) and first[ep1] == 20
    lmb0 = np.load(Path(cfg.paths.labels) / DEMO / f"dm-{DEMO}-s000-c00.npy")[:, SPEC.idx_lmb]
    assert cli.main(["pack", "--all-pending"]) == 0

    frames, actions, attrs = read_episode(Path(cfg.paths.hdf5) / "dm" / DEMO / f"{ep1}.hdf5")
    assert np.array_equal(actions, labels)
    assert np.array_equal(frames, to_low_res(np.stack(src)))  # frames 20..39 of the segment, not 0..19
    assert attrs["start_tick"] == rendered["segments"][0]["start_tick"] + 20 * 4
    assert (attrs["player"], attrs["demo_id"], attrs["kind"], attrs["cs2_build"]) == ("Bot A", DEMO, "dm", 4242)
    assert attrs["user_id"] == 1  # the spectated bot is told apart by user_id (steamid 0 is shared)
    assert (attrs["cs2wm_git_sha"], attrs["demoparser2_version"]) == ("deadbeef", demo.parser_version())
    # the fake shots: bot 1 fires at ticks 13, 14 (episode 0) and 120 (episode 1); bot 2's shots must not leak in
    assert lmb0.sum() >= 1 and lmb0[0] == 1  # tick 13 is in frame 0 (window (10, 14])
    assert labels[:, SPEC.idx_lmb].sum() == 1
    assert (actions[:, SPEC.keys.index("w")] == 1).all()  # moving forward the whole time


def test_a_qa_failing_episode_is_not_packed_or_labeled(cfg, use_parser, fake_hlae) -> None:
    qa = run_to_extracted(cfg, use_parser, fake_hlae)
    bad = qa[2]
    assert bad["passed"] is False and bad["kind"] == "dm" and bad["demo_id"] == DEMO and bad["n_frames"] == 5
    assert any(r.startswith("missing_tick_frac") for r in bad["reasons"]) and "fire_recall" in bad["stats"]
    assert sorted(p.name for p in (Path(cfg.paths.labels) / DEMO).iterdir()) == [
        f"dm-{DEMO}-s000-c00.npy",
        f"dm-{DEMO}-s000-c01.npy",
    ]
    results = pack.run(DEMO, cfg)
    assert [r.n_frames for r in results] == [20, 20]


# --- idempotence -------------------------------------------------------------------------------------------------


def test_rerunning_extract_does_not_duplicate_qa_lines(cfg, use_parser, fake_hlae) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    path = Path(cfg.paths.qa) / f"{DEMO}.jsonl"
    first = path.read_text()
    for _ in range(2):
        extract_actions.run(Path(cfg.paths.demos) / "dm" / f"{DEMO}.dem", cfg)
    assert path.read_text() == first and len(first.splitlines()) == 3
    assert not list(path.parent.glob("*.tmp"))


def test_rerunning_pack_rewrites_the_manifest_without_duplicates(cfg, use_parser, fake_hlae, tmp_path) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    import shutil

    backup = tmp_path / "backup"
    shutil.copytree(Path(cfg.paths.frames) / DEMO, backup / "frames")
    shutil.copytree(Path(cfg.paths.labels) / DEMO, backup / "labels")
    pack.run(DEMO, cfg)
    first = read_manifest(cfg)
    shutil.copytree(backup / "frames", Path(cfg.paths.frames) / DEMO)
    shutil.copytree(backup / "labels", Path(cfg.paths.labels) / DEMO)
    pack.run(DEMO, cfg)
    assert read_manifest(cfg) == first and len(first) == 4
    verify_manifest(cfg)
    assert not list((Path(cfg.paths.hdf5) / "dm" / DEMO).glob("*.tmp"))


def test_pack_replaces_a_stale_multi_line_manifest(cfg, use_parser, fake_hlae) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    manifest = Path(cfg.paths.hdf5) / "dm" / DEMO / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    pack.append_manifest(manifest, {"file": "old.hdf5", "frames": 1, "sha256": "0" * 64})
    pack.run(DEMO, cfg)
    assert "old.hdf5" not in [r["file"] for r in read_manifest(cfg)]


# --- the slicing check -------------------------------------------------------------------------------------------


def test_slicing_check_catches_an_off_by_one(cfg, use_parser, fake_hlae, monkeypatch) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    real = pack.write_episode

    def shifted(frames, actions, meta, out_dir, cfg, fullres_dir=None):
        return real(frames, np.roll(actions, 1, axis=0), meta, out_dir, cfg, fullres_dir)  # labels one frame late

    monkeypatch.setattr(pack, "write_episode", shifted)
    with pytest.raises(PackError, match="write/read round trip"):
        pack.run(DEMO, cfg)
    # nothing half-packed is left for sync to find, and the inputs are kept for a retry
    assert not list((Path(cfg.paths.hdf5) / "dm" / DEMO).glob("*"))
    assert not list((Path(cfg.paths.hdf5_fullres) / "dm" / DEMO).glob("*"))
    assert (Path(cfg.paths.frames) / DEMO).is_dir() and (Path(cfg.paths.labels) / DEMO).is_dir()


def test_corrupt_label_file_is_caught_before_writing(cfg, use_parser, fake_hlae) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    np.save(Path(cfg.paths.labels) / DEMO / f"dm-{DEMO}-s000-c00.npy", np.zeros((19, 51), dtype=np.float32))
    with pytest.raises(ValueError, match="actions must have shape"):
        pack.run(DEMO, cfg)
    assert (Path(cfg.paths.frames) / DEMO).is_dir()


# --- guards ------------------------------------------------------------------------------------------------------


def test_extract_needs_render_json_for_the_same_plan(cfg, use_parser, fake_hlae) -> None:
    add_demo(cfg)
    use_parser(make_parser())
    assert cli.main(["segment", "--all-pending"]) == 0
    dem = Path(cfg.paths.demos) / "dm" / f"{DEMO}.dem"
    with pytest.raises(FileNotFoundError, match="render.json"):
        extract_actions.run(dem, cfg)
    assert cli.main(["render", "--all-pending"]) == 0
    render_json = Path(cfg.paths.frames) / DEMO / "render.json"
    rec = json.loads(render_json.read_text())
    rec["plan_sha256"] = hashlib.sha256(b"other plan").hexdigest()
    render_json.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="different plan"):
        extract_actions.run(dem, cfg)


def test_pack_without_extract_fails_clearly(cfg, use_parser, fake_hlae) -> None:
    run_to_rendered(cfg, use_parser, fake_hlae)
    with pytest.raises(FileNotFoundError, match=r"qa.*jsonl"):
        pack.run(DEMO, cfg)


def test_pack_needs_the_extra_rendered_frame(cfg, use_parser, fake_hlae) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    rendered = json.loads((Path(cfg.paths.frames) / DEMO / "render.json").read_text())
    last_first = rendered["segments"][0]["episodes"][1]["first_frame"]  # episode c01 uses frames 20..40
    (Path(cfg.paths.frames) / DEMO / "seg000" / f"{last_first + 20:06d}.png").unlink()
    with pytest.raises(FileNotFoundError, match="n \\+ 1 rendered frames"):
        pack.run(DEMO, cfg)
    assert not list((Path(cfg.paths.hdf5) / "dm" / DEMO).glob("*"))


def test_all_qa_failing_demo_packs_nothing_and_syncs_without_a_manifest(cfg, use_parser, fake_hlae, remote) -> None:
    run_to_rendered(cfg, use_parser, fake_hlae)
    use_parser(make_parser(drop=range(END + 1)))  # bot 1 has no ticks at all
    assert cli.main(["extract", "--all-pending"]) == 0
    assert cli.main(["pack", "--all-pending"]) == 0
    assert state_of(cfg)["state"] == "packed" and state_of(cfg)["packed_episodes"] == 0
    assert not (Path(cfg.paths.hdf5) / "dm" / DEMO).exists() and not (Path(cfg.paths.frames) / DEMO).exists()
    sup = Supervisor(cfg, PipelineState.load(Path(cfg.paths.state) / "pipeline.json"))
    assert sup.run_sync(LocalTransport()).ok
    assert state_of(cfg)["state"] == "synced" and (remote / "qa" / f"{DEMO}.jsonl").is_file()


# --- the CLI -----------------------------------------------------------------------------------------------------


def test_cli_failure_continues_with_the_next_demo_and_exits_non_zero(cfg, use_parser, fake_hlae, capsys) -> None:
    add_demo(cfg, "dm-0001")
    add_demo(cfg, "dm-0002")
    parser = make_parser()
    use_parser(parser)
    real = demo._make_parser
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if Path(path).stem == "dm-0001":
            raise RuntimeError("corrupt demo")
        return real(path)

    demo._make_parser = flaky
    try:
        assert cli.main(["segment", "--all-pending"]) == 1
    finally:
        demo._make_parser = real
    assert state_of(cfg, "dm-0001")["state"] == "failed"
    assert state_of(cfg, "dm-0001")["failed_stage"] == "segment"
    assert "corrupt demo" in state_of(cfg, "dm-0001")["reason"]
    assert state_of(cfg, "dm-0002")["state"] == "segmented"  # the next demo still ran
    captured = capsys.readouterr()
    assert "segment: 1 done, 1 failed, 2 considered" in captured.out and "FAILED dm-0001" in captured.err

    # a plain re-run skips the failed demo; --retry-failed picks it up
    assert cli.main(["segment", "--all-pending"]) == 0
    assert state_of(cfg, "dm-0001")["state"] == "failed"
    assert cli.main(["segment", "--all-pending", "--retry-failed"]) == 0
    assert state_of(cfg, "dm-0001")["state"] == "segmented" and "reason" not in state_of(cfg, "dm-0001")


def test_all_pending_runs_demos_in_id_order_and_single_demo_updates_state(cfg, use_parser, fake_hlae) -> None:
    for d in ("dm-0003", "dm-0001", "dm-0002"):
        add_demo(cfg, d)
    use_parser(make_parser())
    order: list[str] = []
    real = segment.run

    def spy(path, *a, **k):
        order.append(Path(path).stem)
        return real(path, *a, **k)

    segment.run = spy
    try:
        assert cli.main(["segment", "dm-0002"]) == 0  # a single demo is recorded too
        assert state_of(cfg, "dm-0002")["state"] == "segmented" and state_of(cfg, "dm-0002")["kind"] == "dm"
        assert cli.main(["segment", "--all-pending"]) == 0
    finally:
        segment.run = real
    assert order == ["dm-0002", "dm-0001", "dm-0003"]


def test_cli_argument_errors(cfg) -> None:
    with pytest.raises(SystemExit, match="give a demo id"):
        cli.main(["pack"])
    with pytest.raises(SystemExit, match="not both"):
        cli.main(["extract", DEMO, "--all-pending"])
    with pytest.raises(SystemExit, match="invalid --since"):
        cli.main(["qa-summary", "--since", "soon"])
    with pytest.raises(SystemExit, match="--cfg-only"):
        cli.main(["render", "--cfg-only", "--all-pending"])


def test_status_does_not_write_state_but_shows_undiscovered_demos(cfg, capsys) -> None:
    add_demo(cfg)
    assert cli.main(["status"]) == 0
    assert f"{DEMO}  dm    pending" in capsys.readouterr().out
    assert not (Path(cfg.paths.state) / "pipeline.json").exists()


def test_zero_episode_plan_fails_the_segment_stage(cfg, use_parser) -> None:
    add_demo(cfg)
    parser = make_parser()
    parser._events["player_death"] = user_events([(20, 1), (20, 2)], user_name=["Bot A", "Bot B"], user_steamid=[0, 0])
    use_parser(parser)
    assert cli.main(["segment", "--all-pending"]) == 1
    assert "no episodes" in state_of(cfg)["reason"]


# --- review fixes: pack crash safety -----------------------------------------------------------------------------


def hdf5_dir(cfg) -> Path:
    return Path(cfg.paths.hdf5) / "dm" / DEMO


def snapshot(cfg) -> dict[str, str]:
    """sha256 of every file under hdf5/ and hdf5_fullres/ of the demo, by relative path."""
    out = {}
    for tree in (cfg.paths.hdf5, cfg.paths.hdf5_fullres):
        for p in sorted(Path(tree).rglob("*")):
            if p.is_file():
                out[p.relative_to(tree).as_posix()] = pack.sha256_file(p)
    return out


def test_pack_killed_after_the_manifest_is_idempotent_on_rerun(cfg, use_parser, fake_hlae, monkeypatch) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    real_cleanup = pack._cleanup

    def killed(cfg, demo_id):
        raise KeyboardInterrupt  # power loss after the swap/manifest, before frames and labels are deleted

    monkeypatch.setattr(pack, "_cleanup", killed)
    sup = Supervisor(cfg, PipelineState.load(Path(cfg.paths.state) / "pipeline.json"))
    with pytest.raises(KeyboardInterrupt):
        sup.run_stage("pack", sup.state.pending_for("pack"))
    assert state_of(cfg)["state"] == "extracted" and (Path(cfg.paths.frames) / DEMO).is_dir()
    done = snapshot(cfg)

    # Worst case: the process died inside the delete and left a half-deleted tree, render.json already gone.
    monkeypatch.setattr(pack, "_cleanup", real_cleanup)
    frames = Path(cfg.paths.frames) / DEMO
    frames.rename(frames.with_name(DEMO + ".deleting"))
    (frames.with_name(DEMO + ".deleting") / "render.json").unlink()
    assert cli.main(["pack", "--all-pending"]) == 0
    assert state_of(cfg)["state"] == "packed" and state_of(cfg)["frames"] == 40
    assert snapshot(cfg) == done  # nothing was rewritten
    assert not any(Path(cfg.paths.frames).glob("*")) and not (Path(cfg.paths.labels) / DEMO).exists()
    verify_manifest(cfg)


def test_pack_killed_before_the_delete_leaves_render_json_missing_but_reruns(cfg, use_parser, fake_hlae) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    pack.run(DEMO, cfg)
    frames = Path(cfg.paths.frames) / DEMO
    frames.mkdir(parents=True)  # a partial tree whose render.json is already gone
    (frames / "seg000").mkdir()
    results = pack.run(DEMO, cfg)
    assert [r.n_frames for r in results] == [20, 20] and all(r.low_res_path.is_file() for r in results)
    assert not frames.exists()


def test_render_and_extract_sweep_deleting_dirs(cfg, use_parser, fake_hlae) -> None:
    run_to_rendered(cfg, use_parser, fake_hlae)
    junk = Path(cfg.paths.frames) / f"{DEMO}.deleting"
    junk.mkdir()
    labels_junk = Path(cfg.paths.labels) / f"{DEMO}.deleting"
    labels_junk.mkdir(parents=True)
    extract_actions.run(Path(cfg.paths.demos) / "dm" / f"{DEMO}.dem", cfg)
    assert not junk.exists() and not labels_junk.exists()
    junk.mkdir()
    render.run(Path(cfg.paths.demos) / "dm" / f"{DEMO}.dem", cfg)
    assert not junk.exists()


def test_a_failing_rerun_keeps_the_earlier_complete_output(cfg, use_parser, fake_hlae, monkeypatch, tmp_path) -> None:
    import shutil

    run_to_extracted(cfg, use_parser, fake_hlae)
    backup = tmp_path / "backup"
    shutil.copytree(Path(cfg.paths.frames) / DEMO, backup / "frames")
    shutil.copytree(Path(cfg.paths.labels) / DEMO, backup / "labels")
    pack.run(DEMO, cfg)
    before = snapshot(cfg)
    shutil.copytree(backup / "frames", Path(cfg.paths.frames) / DEMO)
    shutil.copytree(backup / "labels", Path(cfg.paths.labels) / DEMO)
    monkeypatch.setattr(pack, "_done_results", lambda *a: None)  # force a real re-pack, not the idempotent shortcut
    real = pack.write_episode
    calls = {"n": 0}

    def second_fails(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real(*a, **k)

    monkeypatch.setattr(pack, "write_episode", second_fails)
    with pytest.raises(OSError):
        pack.run(DEMO, cfg)
    assert snapshot(cfg) == before  # files and manifest of the earlier complete output are untouched
    verify_manifest(cfg)
    assert not list(Path(cfg.paths.hdf5).glob("*/*.staging")) and not list(
        Path(cfg.paths.hdf5_fullres).glob("*/*.staging")
    )
    assert (Path(cfg.paths.frames) / DEMO).is_dir()  # inputs kept for the retry


def test_a_failing_first_pack_leaves_no_output_at_all(cfg, use_parser, fake_hlae, monkeypatch) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    real = pack.write_episode
    calls = {"n": 0}

    def second_fails(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real(*a, **k)

    monkeypatch.setattr(pack, "write_episode", second_fails)
    with pytest.raises(OSError):
        pack.run(DEMO, cfg)
    assert snapshot(cfg) == {} and not list(Path(cfg.paths.hdf5).glob("*/*"))


def test_frame_alignment_check_catches_wrongly_sliced_frames(cfg, use_parser, fake_hlae, monkeypatch) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    real = pack._read_frames
    calls = {"n": 0}

    def off_by_one(seg_dir, first, n, size):
        calls["n"] += 1
        # Only the slicing in run() is wrong (the first call per episode); the checker's own reads stay right.
        return real(seg_dir, first + 1, n, size) if calls["n"] == 1 else real(seg_dir, first, n, size)

    monkeypatch.setattr(pack, "_read_frames", off_by_one)
    with pytest.raises(PackError, match="not source frames"):
        pack.run(DEMO, cfg)
    assert snapshot(cfg) == {}


# --- review fixes: provenance, explicit runs, plan paths ---------------------------------------------------------


def test_cs2_build_prefers_the_sidecar_and_warns_on_a_disagreement(cfg, use_parser, fake_hlae, caplog) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    plan_path = Path(cfg.paths.plans) / DEMO / "render_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["cs2_build"] = 9999  # planned on another build than the one that recorded the demo (sidecar: 4242)
    plan_path.write_text(json.dumps(plan))
    rj = Path(cfg.paths.frames) / DEMO / "render.json"
    rec = json.loads(rj.read_text())
    rec["plan_sha256"] = pack.sha256_file(plan_path)
    rj.write_text(json.dumps(rec))
    with caplog.at_level("WARNING"):
        (res, *_) = pack.run(DEMO, cfg)
    assert read_episode(res.low_res_path)[2]["cs2_build"] == 4242
    assert "recorded on CS2 build 4242 but planned on build 9999" in caplog.text


def test_unknown_cs2_build_warns_and_stores_empty(cfg, use_parser, fake_hlae, caplog) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    (Path(cfg.paths.demos) / "dm" / f"{DEMO}.json").unlink()
    with caplog.at_level("WARNING"):
        (res, *_) = pack.run(DEMO, cfg)
    assert read_episode(res.low_res_path)[2]["cs2_build"] == ""
    assert "cs2_build is unknown" in caplog.text


def test_explicit_rerun_of_a_later_state_demo_without_inputs_is_refused(
    cfg, use_parser, fake_hlae, remote, capsys
) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    assert cli.main(["pack", "--all-pending"]) == 0
    Supervisor(cfg, PipelineState.load(Path(cfg.paths.state) / "pipeline.json")).run_sync(LocalTransport())
    before = (Path(cfg.paths.state) / "pipeline.json").read_text()
    for stage in ("extract", "pack"):
        with pytest.raises(SystemExit, match=f"{stage}: {DEMO} is already synced.*refusing"):
            cli.main([stage, DEMO])
    assert (Path(cfg.paths.state) / "pipeline.json").read_text() == before
    assert state_of(cfg)["state"] == "synced"


def test_explicit_run_in_the_matching_or_earlier_state_still_runs(cfg, use_parser, fake_hlae) -> None:
    run_to_extracted(cfg, use_parser, fake_hlae)
    assert cli.main(["extract", DEMO]) == 0  # already extracted, but the inputs exist: allowed
    assert state_of(cfg)["state"] == "extracted"
    assert cli.main(["segment", DEMO]) == 0  # segmenting again is allowed too (the .dem is there)


def test_segment_prints_each_plan_path(cfg, use_parser, capsys) -> None:
    add_demo(cfg, "dm-0001")
    add_demo(cfg, "dm-0002")
    use_parser(make_parser())
    assert cli.main(["segment", "--all-pending"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:2] == [f"segment {d}: {Path(cfg.paths.plans) / d / 'render_plan.json'}" for d in ("dm-0001", "dm-0002")]
    assert out[2] == "segment: 2 done, 0 failed, 2 considered"
