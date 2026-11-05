"""Tests for cs2wm.pipeline.generate with a fake CS2, clock and netcon (no CS2, no sleeping)."""

import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar

import pytest
from omegaconf import DictConfig, OmegaConf

from cs2wm.config import load_config
from cs2wm.pipeline import generate
from cs2wm.pipeline.generate import (
    CS2Process,
    GenerateError,
    Generator,
    Hooks,
    PatchGuardError,
    Watchdog,
    build_launch_command,
    install_cfg,
    make_demo_id,
    plan_chunks,
    render_cfg,
)

BUILD = 20260101
T0 = datetime(2026, 10, 23, 22, 0, 0)


def _cfg(tmp_path: Path, *overrides: str) -> DictConfig:
    return load_config(
        overrides=[
            f"paths.root={tmp_path / 'data'}",
            f"paths.cs2_dir={tmp_path / 'steam/steamapps/common/CS'}",
            "generate.expected_build=null",
            "generate.map_load_wait_s=10",
            "generate.team_settle_s=1",
            "generate.restart_wait_s=2",
            "generate.map_load_timeout_s=30",
            "generate.netcon_connect_timeout_s=10",
            "generate.close_stable_s=2",
            "generate.close_timeout_s=10",
            "generate.poll_s=5",
            *overrides,
        ]
    )


# --- command line ------------------------------------------------------------------------------------------------


def test_launch_command_netcon(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cmd = build_launch_command(cfg, "dm", netconport=2121)
    assert Path(cmd[0]) == tmp_path / "steam/steamapps/common/CS/game/bin/win64/cs2.exe"
    assert "-insecure" in cmd
    assert cmd[cmd.index("-netconport") + 1] == "2121"
    assert cmd[-2:] == ["+exec", "generate_dm"]
    assert all("hlae" not in part.lower() for part in cmd)


def test_launch_command_without_netconport(tmp_path: Path) -> None:
    assert "-netconport" not in build_launch_command(_cfg(tmp_path), "smoke")


def test_launch_command_refuses_without_insecure(tmp_path: Path) -> None:
    raw = OmegaConf.create(OmegaConf.to_container(_cfg(tmp_path)))  # load_config rejects this, so bypass it
    raw.generate.launch_options = ["-novid", "-console"]
    with pytest.raises(ValueError, match="-insecure"):
        build_launch_command(raw, "dm")


def test_launch_command_rejects_unknown_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="profile"):
        build_launch_command(_cfg(tmp_path), "ranked")


class FakePopen:
    instances: ClassVar[list["FakePopen"]] = []

    def __init__(self, cmd, **kwargs) -> None:
        self.cmd, self.kwargs = cmd, kwargs
        self.pid = 1000 + len(FakePopen.instances)
        self.returncode: int | None = None
        self.killed = False
        FakePopen.instances.append(self)

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.killed, self.returncode = True, -9

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture(autouse=True)
def _reset_popen():
    FakePopen.instances = []


def test_process_refuses_without_insecure(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="-insecure"):
        CS2Process(["cs2.exe", "-novid"], tmp_path, 5, popen=FakePopen)
    proc = CS2Process(["cs2.exe", "-insecure"], tmp_path, 5, popen=FakePopen)
    proc.cmd.remove("-insecure")  # mutated after construction: __enter__ checks again
    with pytest.raises(ValueError, match="-insecure"), proc:
        pass
    assert FakePopen.instances == []


def test_process_launch_and_kill_posix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate, "_is_windows", lambda: False)
    ran: list = []
    with CS2Process(["cs2.exe", "-insecure"], tmp_path, 5, popen=FakePopen, run=lambda *a, **k: ran.append(a)) as p:
        assert "creationflags" not in FakePopen.instances[0].kwargs
        assert p.poll() == "running"
    assert FakePopen.instances[0].killed
    assert ran == []


def test_process_windows_uses_taskkill_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate, "_is_windows", lambda: True)
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    ran: list = []
    proc = CS2Process(["cs2.exe", "-insecure"], tmp_path, 5, popen=FakePopen, run=lambda *a, **k: ran.append(a[0]))
    with proc:
        assert FakePopen.instances[0].kwargs["creationflags"] == 512
    assert ran == [["taskkill", "/T", "/F", "/PID", str(FakePopen.instances[0].pid)]]
    proc.kill()  # the main process is dead, but its child tree may not be: taskkill runs again
    assert len(ran) == 2


# --- cfg -----------------------------------------------------------------------------------------------------------


def test_render_cfg_inlines_exec_includes(tmp_path: Path) -> None:
    text = render_cfg("smoke", _cfg(tmp_path))
    assert "exec generate_dm" not in text
    assert "bot_quota 10" in text  # from generate_dm.cfg
    assert "bot_allow_grenades 1" in text  # from generate_smoke.cfg
    assert "\ntv_autorecord" not in text  # autorecord is never used


def _commands(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("//")]


@pytest.mark.parametrize("profile", ["dm", "smoke", "human"])
def test_cfg_sets_up_gotv_before_the_map(tmp_path: Path, profile: str) -> None:
    cmds = _commands(render_cfg(profile, _cfg(tmp_path)))
    assert cmds.index("tv_enable 1") < cmds.index("map de_dust2")
    assert cmds.index("tv_transmitall 1") < cmds.index("map de_dust2")
    assert cmds.index("tv_record_immediate 1") < cmds.index("map de_dust2")
    assert cmds.index("game_type 1") < cmds.index("map de_dust2") and "game_mode 2" in cmds


@pytest.mark.parametrize("profile", ["dm", "smoke", "human"])
def test_cfg_leaves_jointeam_and_recording_to_netcon(tmp_path: Path, profile: str) -> None:
    cmds = _commands(render_cfg(profile, _cfg(tmp_path)))
    assert not any(c.startswith(("jointeam", "record", "tv_record ", "tv_autorecord")) for c in cmds)


def test_cfgs_have_no_resolved_markers() -> None:
    for name in (
        "generate_dm",
        "generate_human",
        "generate_smoke",
        "generate_dm_post",
        "generate_human_post",
        "generate_smoke_post",
        "render_base",
    ):
        text = (generate.cs2wm_config.DEFAULT_CONFIG_PATH.parent / "cs2" / f"{name}.cfg").read_text()
        assert "CONFIRM(#14)" not in text and "UNVERIFIED" not in text
        assert "SAFETY: HLAE and CS2 are only ever run offline with -insecure" in text


def test_render_base_cfg_turns_xray_off() -> None:
    text = (generate.cs2wm_config.DEFAULT_CONFIG_PATH.parent / "cs2" / "render_base.cfg").read_text()
    assert "\nspec_show_xray 0\n" in text


def test_render_cfg_detects_exec_cycle(tmp_path: Path) -> None:
    (tmp_path / "generate_a.cfg").write_text("exec generate_b\n")
    (tmp_path / "generate_b.cfg").write_text("exec generate_a\n")
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="cycle"):
        generate.render_cfg("a", cfg, cfg_dir=tmp_path)  # type: ignore[arg-type]


def test_install_cfg_into_cs2_dir(tmp_path: Path) -> None:
    cs2_dir = tmp_path / "cs2"
    text = render_cfg("dm", _cfg(tmp_path))
    path, sha = install_cfg(text, "dm", cs2_dir)
    assert path == cs2_dir / "game/csgo/cfg/generate_dm.cfg"
    assert path.read_text() == text
    assert sha == hashlib.sha256(text.encode()).hexdigest()
    assert install_cfg(text, "dm", cs2_dir)[1] == sha  # idempotent


# --- schedule ------------------------------------------------------------------------------------------------------


def test_plan_chunks() -> None:
    assert plan_chunks(1, 10) == [10.0] * 6
    assert plan_chunks(0.1, 2) == [2.0, 2.0, 2.0]
    assert plan_chunks(0.25, 10) == [10.0, 5.0]
    assert plan_chunks(0, 10) == []


def test_demo_ids_are_unique_sortable_and_safe() -> None:
    ids = [make_demo_id("dm", T0, i) for i in (0, 1, 10)]
    assert ids[0] == "dm_20261023-220000_000"
    assert ids == sorted(ids)
    assert all(c.isalnum() or c in "_-" for i in ids for c in i)


# --- watchdog ------------------------------------------------------------------------------------------------------


def test_watchdog_state_machine() -> None:
    wd = Watchdog(timeout_s=60, now=0)
    assert wd.observe(10, True, 100) == "running"  # baseline
    assert wd.observe(50, True, 100) == "running"  # no growth, but within the timeout
    assert wd.observe(55, True, 200) == "running"  # growth restarts the clock
    assert wd.observe(114, True, 200) == "running"
    assert wd.observe(115, True, 200) == "hung"
    assert wd.observe(116, False, 200) == "exited"  # exit wins


def test_watchdog_ignores_shrinking_and_rearms() -> None:
    wd = Watchdog(timeout_s=60, now=0)
    wd.observe(0, True, 500)
    assert wd.observe(30, True, 100) == "running"  # a finished chunk was moved away: not growth
    assert wd.observe(60, True, 100) == "hung"
    wd.rearm(60)
    assert wd.observe(61, True, 100) == "running"
    assert wd.observe(70, True, 150) == "running"


def test_process_poll_reports_hang(tmp_path: Path) -> None:
    now = [0.0]
    proc = CS2Process(
        ["cs2.exe", "-insecure"], tmp_path, 1, popen=FakePopen, clock=lambda: now[0], size_probe=lambda _: 10
    )
    with proc:
        assert proc.poll() == "running"
        now[0] = 61
        assert proc.poll() == "hung"
        FakePopen.instances[0].returncode = 1
        assert proc.poll() == "exited"


# --- full runs with a fake CS2 ------------------------------------------------------------------------------------


class FakeWorld:
    """A fake CS2 plus clock. ``sleep`` advances time and lets the running process write its demo."""

    def __init__(self, csgo: Path) -> None:
        self.csgo = csgo
        self.commands: list[tuple[float, str]] = []  # (clock, netcon command), in the order received
        self.queries: list[str] = []  # status polls and cvar read-backs (not part of the command sequence)
        self.map_ready_after = 3.0  # seconds after a launch until `status` shows the map
        self.jointeam_failures = 0  # `jointeam 1` replies without the ChangeTeam line before one works
        self.record_reply_ok = True  # the start command prints the success text
        self.record_writes = True  # the start command creates the demo file
        self.cvars: dict[str, str] = {}  # cvars set over netcon
        self.cvar_value = "2"  # what `bot_difficulty` reads back as
        self.demo_secs: float | None = None  # length of a salvaged demo according to its ticks
        self.t = 0.0
        self.build = BUILD
        self.recording: Path | None = None
        self.launches: list[FakePopen] = []
        self.crash_at: list[float] = []  # clock times at which the running process dies
        self.hang_at: list[float] = []  # clock times at which the running process stops writing
        self.exit_on_launch = False
        self.hung = False
        self.launch_t = 0.0
        self.inspect_fails = False
        self.crash_after_launch: float | None = None  # every launched process dies this long after its launch
        self.copy_failures = 0  # copies raising PermissionError before one succeeds (-1: always)
        self.notified: list[dict] = []
        csgo.mkdir(parents=True, exist_ok=True)

    @property
    def proc(self) -> FakePopen | None:
        return self.launches[-1] if self.launches else None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    def sleep(self, s: float) -> None:
        end = self.t + s
        while self.t < end:
            step = min(1.0, end - self.t)
            self.t += step
            if (
                self.crash_after_launch is not None
                and self.alive()
                and self.t - self.launch_t >= self.crash_after_launch
            ):
                self.proc.returncode = 1
            if self.crash_at and self.t >= self.crash_at[0] and self.alive():
                self.crash_at.pop(0)
                self.proc.returncode = 1
            if self.hang_at and self.t >= self.hang_at[0] and self.alive():
                self.hang_at.pop(0)
                self.hung = True
            if self.alive() and not self.hung:
                self._grow(step)

    def _grow(self, step: float) -> None:
        if self.recording is not None:
            with self.recording.open("ab") as f:
                f.write(b"x" * int(step * 100))

    def popen(self, cmd, **kwargs) -> FakePopen:
        p = FakePopen(cmd, **kwargs)
        self.launches.append(p)
        self.hung, self.recording, self.launch_t = False, None, self.t
        if self.exit_on_launch:
            p.returncode = 1
        return p

    def hooks(self) -> Hooks:
        world = self

        class Netcon:
            def __init__(self, **_) -> None:
                pass

            def wait_ready(self, timeout_s: float) -> None:
                if not world.alive():
                    raise ConnectionError("refused")

            def send(self, cmd: str, retries=None) -> str:
                if not world.alive():
                    raise ConnectionError("refused")
                if cmd == "status":
                    world.queries.append(cmd)
                    if world.t - world.launch_t >= world.map_ready_after:
                        return "map     : de_dust2 at: 0 x, 0 y, 0 z\nplayers : 1 humans, 9 bots (0 max)\n"
                    return "map     : <empty>\nplayers : 0 humans, 0 bots (0 max)\n"
                if " " not in cmd and cmd not in ("stop", "tv_stoprecord"):  # a cvar read-back
                    world.queries.append(cmd)
                    value = world.cvar_value if cmd == "bot_difficulty" else world.cvars.get(cmd, "")
                    return f"Named camera 'cam_default' not found\n{cmd} = {value}\n"  # CS2's format (PC, Nov 5)
                for part in cmd.split(";"):
                    words = part.split()
                    if len(words) == 2 and words[0] != "jointeam":
                        world.cvars[words[0]] = words[1]
                world.commands.append((world.t, cmd))
                if cmd == "jointeam 1":
                    if world.jointeam_failures > 0:
                        world.jointeam_failures -= 1
                        return ""
                    return "ChangeTeam() CTMDBG , team 0, req team 1 willSwitch 0\r\n"
                if cmd.startswith("sv_cheats 1; "):  # a chunk starts: `sv_cheats 1; record|tv_record <name>`
                    name = cmd.split()[-1]
                    if world.record_writes:
                        world.recording = world.csgo / f"{name}.dem"
                        world.recording.touch()
                    if not world.record_reply_ok:
                        return ""
                    if "tv_record" in cmd:
                        return f"Recording SourceTV replay from HLTVServerAsync:\r\nplaydemo {name}.dem\r\n"
                    return f"CDemoRecorder::StartRecording({name}.dem)\r\n"
                if cmd in ("stop", "tv_stoprecord"):
                    world.recording = None
                return ""

        def copy(src: Path, dest: Path) -> None:
            if world.copy_failures != 0:
                world.copy_failures = max(world.copy_failures - 1, -1) if world.copy_failures > 0 else -1
                raise PermissionError("file is locked by another process")
            shutil.copyfile(src, dest)

        def parse_error(path: Path) -> str | None:
            return "unexpected EOF" if world.inspect_fails else None

        return Hooks(
            clock=lambda: world.t,
            sleep=world.sleep,
            wall=lambda: T0 + timedelta(seconds=world.t),
            popen=world.popen,
            run=lambda *a, **k: None,
            netcon=Netcon,
            build_id=lambda _: world.build,
            parse_error=parse_error,
            demo_seconds=lambda path: world.demo_secs,
            copy=copy,
            notify=lambda msg, **kw: world.notified.append({"message": msg, **kw}),
            git_sha=lambda: "deadbeef",
        )


def _run(tmp_path: Path, hours: float = 0.1, *overrides: str, profile: str = "dm", chunk_min: float = 2):
    cfg = _cfg(tmp_path, "generate.hang_timeout_min=1", *overrides)
    world = FakeWorld(Path(cfg.paths.cs2_dir) / "game/csgo")
    return cfg, world, Generator(cfg, profile, hours, chunk_min, world.hooks())


def test_run_records_all_chunks_with_sidecars(tmp_path: Path) -> None:
    cfg, world, gen = _run(tmp_path)
    kept = gen.run()
    demos = Path(cfg.paths.demos) / "dm"
    assert [p.name for p in kept] == [f"dm_20261023-220000_00{i}.dem" for i in range(3)]
    assert sorted(p.name for p in demos.iterdir()) == sorted(
        [p.name for p in kept] + [p.with_suffix(".json").name for p in kept]
    )
    assert list(world.csgo.glob("*.dem")) == []  # moved out of CS2's dir
    assert len(world.launches) == 1
    assert world.launches[0].poll() is not None  # CS2 is stopped at the end

    side = json.loads(kept[1].with_suffix(".json").read_text())
    cfg_sha = hashlib.sha256((world.csgo / "cfg/generate_dm.cfg").read_bytes()).hexdigest()
    assert side["demo_id"] == "dm_20261023-220000_001"
    assert side["profile"] == side["kind"] == "dm"
    assert side["chunk_min"] == 2.0
    assert side["chunking"] == "gotv"
    assert side["cs2_build"] == BUILD
    assert side["cfg_sha256"] == cfg_sha
    assert side["cs2wm_git_sha"] == "deadbeef"
    assert side["truncated"] is False
    assert side["start_time"] == (T0 + timedelta(seconds=10 + 120)).isoformat()[:19] or side["start_time"].startswith(
        "2026-10-23T22:0"
    )
    assert 119 <= side["duration_s"] <= 125


def test_netcon_launch_has_port_and_insecure(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path)
    gen.run()
    cmd = world.launches[0].cmd
    assert "-insecure" in cmd and "-netconport" in cmd


def test_crash_relaunches_and_keeps_parsable_truncated_chunk(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path)
    world.crash_at = [200.0]  # during the second chunk
    kept = gen.run()
    assert len(world.launches) == 2
    sides = [json.loads(p.with_suffix(".json").read_text()) for p in kept]
    assert [s["truncated"] for s in sides] == [False, True, False, False]
    assert len({p.name for p in kept}) == len(kept)  # ids keep counting across the restart
    assert sum(s["duration_s"] for s in sides) >= 360


def test_crash_with_unparsable_chunk_is_rejected_with_reason(tmp_path: Path) -> None:
    cfg, world, gen = _run(tmp_path)
    world.crash_at = [200.0]
    world.inspect_fails = True
    kept = gen.run()
    rejected = Path(cfg.paths.demos) / "_rejected"
    assert sorted(p.name for p in rejected.iterdir()) == [
        "dm_20261023-220000_001.dem",
        "dm_20261023-220000_001.reason.txt",
    ]
    assert "unexpected EOF" in (rejected / "dm_20261023-220000_001.reason.txt").read_text()
    assert all(not p.with_suffix(".json").read_text().count('"truncated": true') for p in kept)
    assert list(world.csgo.glob("*.dem")) == []


def test_hang_kills_and_relaunches(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path)
    world.hang_at = [275.0]
    kept = gen.run()
    assert len(world.launches) == 2
    assert world.launches[0].killed
    sides = [json.loads(p.with_suffix(".json").read_text()) for p in kept]
    hung = [s for s in sides if s["truncated"]]
    assert len(hung) == 1
    assert hung[0]["duration_s"] < 25  # recording time up to the last growth, without the 60 s hang timeout
    assert all(s["duration_s"] == pytest.approx(120, abs=1) for s in sides[:2])


def test_gives_up_when_cs2_never_stays_up(tmp_path: Path) -> None:
    cfg, world, gen = _run(tmp_path)
    world.exit_on_launch = True
    with pytest.raises(GenerateError):
        gen.run()
    assert len(world.launches) == cfg.generate.max_failures
    assert world.notified and world.notified[0]["priority"] == "high"


def test_patch_guard_stops_and_notifies(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.1, "generate.expected_build=111")
    with pytest.raises(PatchGuardError):
        gen.run()
    assert world.launches == []  # never started
    (note,) = world.notified
    assert "111" in note["message"] and str(BUILD) in note["message"]
    assert note["priority"] == "high"


def test_patch_guard_triggers_between_chunks(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.1, f"generate.expected_build={BUILD}")
    real = world.sleep

    def sleep(s: float) -> None:
        real(s)
        if world.t > 150:
            world.build = BUILD + 1  # Steam updated CS2 mid-run

    gen.h.sleep = sleep
    with pytest.raises(PatchGuardError):
        gen.run()
    assert len(gen.kept) == 2  # chunks finished before the guard fired are kept
    assert len(world.notified) == 1
    assert world.proc.killed


def test_patch_guard_passes_with_matching_build(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.1, f"generate.expected_build={BUILD}")
    assert len(gen.run()) == 3
    assert world.notified == []


def _chunk_commands(world: FakeWorld) -> list[str]:
    return [c for _, c in world.commands if c.startswith(("sv_cheats", "tv_stoprecord", "stop"))]


def test_gotv_session_sequence(tmp_path: Path) -> None:
    """launch -> port -> map -> `jointeam 1` -> cvars -> restart -> per chunk `sv_cheats 1; tv_record`, `tv_stoprecord`."""
    _, world, gen = _run(tmp_path, 0.09)
    kept = gen.run()
    assert len(kept) == 3
    assert "-netconport" in world.launches[0].cmd and "-insecure" in world.launches[0].cmd
    expected = []
    for name in [p.stem for p in kept]:
        expected += [f"sv_cheats 1; tv_record {name}", "tv_stoprecord"]
    assert [c for _, c in world.commands] == ["jointeam 1", "bot_difficulty 2", "mp_restartgame 1", *expected]
    (t_join, _), (t_cvars, _), (t_restart, _), (t_first, _) = world.commands[:4]
    assert t_join >= 3 + 10  # `status` showed the map (3 s), then map_load_wait_s
    assert t_cvars == t_restart == t_join + 1  # after team_settle_s, before the restart
    assert t_first == t_restart + 2  # after restart_wait_s: joining a team after tv_record stalls the demo
    sides = [json.loads(p.with_suffix(".json").read_text()) for p in kept]
    assert all(s["chunking"] == "gotv" and not s["truncated"] for s in sides)
    assert list(world.csgo.glob("*.dem")) == []  # GOTV demos land in game/csgo like client demos
    assert "bot_difficulty" in world.queries  # read back


def test_waits_for_the_map_instead_of_sleeping(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.04, "generate.hang_timeout_min=3", "generate.map_load_timeout_s=90", chunk_min=1)
    world.map_ready_after = 40.0  # a slow load: far later than map_load_wait_s
    gen.run()
    assert world.commands[0] == (world.commands[0][0], "jointeam 1")
    assert world.commands[0][0] >= 40 + 10
    assert world.queries.count("status") > 5


def test_map_that_never_loads_fails_the_session_and_relaunches(tmp_path: Path) -> None:
    cfg, world, gen = _run(tmp_path, 0.04, "generate.map_load_timeout_s=20", chunk_min=1)
    world.map_ready_after = 1e9
    with pytest.raises(GenerateError):
        gen.run()
    assert len(world.launches) == cfg.generate.max_failures
    assert world.commands == []  # nothing was sent to a game without a map
    assert world.t < 3 * 40  # each session gave up after ~map_load_timeout_s, not after the hang timeout


def test_jointeam_is_retried_until_the_team_change_is_logged(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.04, chunk_min=1)
    world.jointeam_failures = 2
    gen.run()
    cmds = [c for _, c in world.commands]
    assert cmds[:3] == ["jointeam 1"] * 3
    assert cmds[3] == "bot_difficulty 2" and len(world.launches) == 1


def test_jointeam_that_never_works_fails_the_session(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.04, "generate.jointeam_retries=2", chunk_min=1)
    world.jointeam_failures = 10_000
    with pytest.raises(GenerateError):
        gen.run()
    first_session = [c for _, c in world.commands][: 2 + 1]
    assert first_session == ["jointeam 1"] * 3  # the first attempt plus two retries
    assert not any(c.startswith("sv_cheats") for _, c in world.commands)  # never recorded in the wrong state


def test_tv_record_without_the_success_text_fails_fast(tmp_path: Path) -> None:
    cfg, world, gen = _run(tmp_path, 0.04, chunk_min=1)
    world.record_reply_ok = False
    with pytest.raises(GenerateError):
        gen.run()
    assert len(world.launches) == cfg.generate.max_failures
    assert world.t < 3 * 60  # each session was dropped at once, far from the 5 minute hang timeout (60 s here)


def test_tv_record_that_writes_nothing_fails_after_record_check_s(tmp_path: Path) -> None:
    cfg, world, gen = _run(tmp_path, 0.04, "generate.record_check_s=10", chunk_min=1)
    world.record_writes = False
    with pytest.raises(GenerateError):
        gen.run()
    assert len(world.launches) == cfg.generate.max_failures
    assert world.t < 3 * (4 + 10 + 1 + 1 + 2 + 10 + 5)  # launch..record_check_s per session, not the hang timeout


def test_reply_checks_can_be_turned_off(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.04, "generate.verify_netcon_replies=false", chunk_min=1)
    world.record_reply_ok = False  # the file is still written
    world.jointeam_failures = 10_000
    assert gen.run()


def test_post_load_cvars_follow_jointeam_and_precede_the_restart(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.04, profile="smoke", chunk_min=1)
    gen.run()
    cmds = [c for _, c in world.commands]
    assert cmds[0] == "jointeam 1" and cmds[2] == "mp_restartgame 1"
    assert cmds[1].startswith("bot_difficulty 2; mp_ct_default_grenades weapon_smokegrenade")  # dm block + smoke block


def test_long_post_load_block_is_split_into_short_lines(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "cs2"
    cfg_dir.mkdir()
    (cfg_dir / "generate_dm_post.cfg").write_text("\n".join(f"mp_cvar_{i:03d} {'v' * 20}" for i in range(40)) + "\n")
    cmds = generate.post_load_commands("dm", cfg_dir)
    assert len(cmds) == 40
    from cs2wm.pipeline.netcon import MAX_LINE_CHARS, split_lines

    lines = split_lines("; ".join(cmds))
    assert len(lines) > 1 and all(len(line) <= MAX_LINE_CHARS for line in lines)


def test_post_load_commands_expand_includes_and_drop_comments() -> None:
    assert generate.post_load_commands("dm") == ["bot_difficulty 2"]
    assert generate.post_load_commands("smoke")[0] == "bot_difficulty 2"
    assert generate.post_load_commands("nope") == []


def test_a_cvar_that_did_not_stick_is_warned_about(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _, world, gen = _run(tmp_path, 0.04, chunk_min=1)
    world.cvar_value = "1"  # the gamemode cfg won
    with caplog.at_level(logging.WARNING):
        gen.run()
    assert "bot_difficulty did not stick" in caplog.text


def test_salvaged_chunk_is_judged_by_the_ticks_it_holds(tmp_path: Path) -> None:
    """A crashed chunk lost its last ~30 s to tv_delay: wall time says 100 s, the demo says 20 s (< min_chunk_s)."""
    cfg, world, gen = _run(tmp_path, 0.1, "generate.max_failures=3")
    world.crash_at = [17 + 120 + 100]  # 100 s into the second chunk
    world.demo_secs = 20.0
    kept = gen.run()
    cut = [
        json.loads(p.with_suffix(".json").read_text())
        for p in kept
        if 'truncated": true' in p.with_suffix(".json").read_text()
    ]
    assert len(cut) == 1 and cut[0]["duration_s"] == 20.0 and cut[0]["truncated"] is True
    assert cfg.generate.min_chunk_s == 30  # so this chunk was kept but did not count as progress
    assert gen.generated_s >= 20.0


def test_stoprecord_comes_before_the_next_tv_record(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.09)
    gen.run()
    cmds = _chunk_commands(world)
    assert cmds[1::2] == ["tv_stoprecord"] * 3
    assert all(c.startswith("sv_cheats 1; tv_record ") for c in cmds[0::2])
    times = [t for t, _ in world.commands]
    assert times == sorted(times)


def test_netcon_chunking_records_client_demos(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.09, "generate.chunking=netcon")
    kept = gen.run()
    expected = [c for p in kept for c in (f"sv_cheats 1; record {p.stem}", "stop")]
    assert [c for _, c in world.commands] == ["jointeam 1", "bot_difficulty 2", "mp_restartgame 1", *expected]
    assert all(json.loads(p.with_suffix(".json").read_text())["chunking"] == "netcon" for p in kept)


def test_restart_after_join_can_be_turned_off(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.04, "generate.restart_after_join=false", chunk_min=1)
    gen.run()
    cmds = [c for _, c in world.commands]
    assert cmds[:2] == ["jointeam 1", "bot_difficulty 2"] and cmds[2].startswith("sv_cheats 1; tv_record ")
    assert "mp_restartgame 1" not in cmds


def test_human_session_stays_on_a_team(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.04, profile="human", chunk_min=1)
    gen.run()
    assert not {"jointeam 1", "mp_restartgame 1"} & {c for _, c in world.commands}
    assert world.commands[0][1] == "bot_difficulty 2"  # post-load cvars still apply
    assert world.commands[1][1].startswith("sv_cheats 1; tv_record human_")


def test_crash_during_the_team_settle_wait_restarts_without_a_chunk(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.09)
    world.crash_at = [14.5]  # right after `jointeam 1` (map seen at 4 s, +10 s), before the first tv_record
    kept = gen.run()
    assert len(world.launches) == 2 and len(kept) == 3
    first_launch = [c for t, c in world.commands if t < 15]
    assert first_launch == ["jointeam 1"]  # nothing was recorded by the dead session


def test_human_profile_goes_to_its_own_dir(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.04, profile="human", chunk_min=1)
    kept = gen.run()
    assert {p.parent.name for p in kept} == {"human"}
    assert kept[0].name.startswith("human_")
    assert world.launches[0].cmd[-1] == "generate_human"


def test_parse_error_seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "a.dem"
    monkeypatch.setattr(generate.demo, "is_parseable", lambda p: True, raising=False)
    assert generate._parse_error(f) is None
    monkeypatch.setattr(generate.demo, "is_parseable", lambda p: False, raising=False)
    assert generate._parse_error(f) is not None
    monkeypatch.setattr(generate.demo, "is_parseable", None, raising=False)  # fall back to inspect()
    monkeypatch.setattr(generate.demo, "inspect", lambda p: {"ok": False, "error": "eof"})
    assert generate._parse_error(f) == "eof"
    monkeypatch.setattr(generate.demo, "inspect", lambda p: {"ok": True})
    assert generate._parse_error(f) is None
    monkeypatch.setattr(generate.demo, "inspect", lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    assert "boom" in generate._parse_error(f)


def test_parse_error_keeps_a_demo_that_ends_early_but_has_ticks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "a.dem"
    monkeypatch.setattr(generate.demo, "is_parseable", None, raising=False)
    cut = {"ok": False, "truncated": True, "last_tick": 2588, "error": "unexpected end of demo"}
    monkeypatch.setattr(generate.demo, "inspect", lambda p: cut)
    assert generate._parse_error(f) is None
    monkeypatch.setattr(generate.demo, "inspect", lambda p: {**cut, "last_tick": 0})
    assert generate._parse_error(f) == "unexpected end of demo"
    monkeypatch.setattr(generate.demo, "inspect", lambda p: {**cut, "truncated": False})
    assert generate._parse_error(f) == "unexpected end of demo"


def test_parse_error_accepts_both_truncated_signals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """#72 reports a demo that ends early via is_parseable() True and/or inspect()['truncated'] with ticks."""
    f = tmp_path / "a.dem"
    cut = {"ok": False, "truncated": True, "last_tick": 1, "error": "unexpected end of demo"}
    monkeypatch.setattr(generate.demo, "inspect", lambda p: cut)
    monkeypatch.setattr(generate.demo, "is_parseable", lambda p: True, raising=False)
    assert generate._parse_error(f) is None  # signal 1: is_parseable()
    monkeypatch.setattr(generate.demo, "is_parseable", None, raising=False)
    assert generate._parse_error(f) is None  # signal 2: inspect() with truncated + one tick
    monkeypatch.setattr(generate.demo, "inspect", lambda p: {**cut, "last_tick": None})
    assert generate._parse_error(f) == "unexpected end of demo"  # no tick at all: rejected


def test_other_tools_demos_in_the_csgo_dir_are_left_alone(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.09)
    old = world.csgo / "mine.dem"
    old.write_bytes(b"keep me")
    gen.run()
    assert old.read_bytes() == b"keep me"


def test_crash_loop_with_only_rejected_chunks_gives_up(tmp_path: Path) -> None:
    cfg, world, gen = _run(tmp_path)
    world.crash_after_launch = 40  # 30 s after every `record`
    world.inspect_fails = True
    with pytest.raises(GenerateError):
        gen.run()
    assert len(world.launches) == cfg.generate.max_failures
    assert world.notified and world.notified[-1]["priority"] == "high"
    rejected = Path(cfg.paths.demos) / "_rejected"
    assert len(list(rejected.glob("*.dem"))) == cfg.generate.max_failures
    assert gen.kept == []


def test_transient_copy_lock_is_retried(tmp_path: Path) -> None:
    cfg, world, gen = _run(tmp_path)
    world.copy_failures = 3
    kept = gen.run()
    out = Path(cfg.paths.demos) / "dm"
    assert len(kept) == 3
    assert len(list(out.glob("*.json"))) == 3 and list(out.glob("*.part")) == []
    assert world.notified == []


def test_permanent_copy_failure_stops_with_one_summary(tmp_path: Path) -> None:
    cfg, world, gen = _run(tmp_path)
    world.copy_failures = -1
    with pytest.raises(GenerateError, match="3 demo"):
        gen.run()
    out = Path(cfg.paths.demos) / "dm"
    assert list(out.glob("*.json")) == [] and list(out.glob("*.part")) == []  # no orphan sidecar
    assert len(list(world.csgo.glob("dm_*.dem"))) == 3  # sources are kept for a later run
    assert gen.generated_s == 0 and gen.kept == []
    assert len(world.launches) == 1 and len(world.notified) == 2  # one per error type, plus the summary
    assert "could not be published" in world.notified[-1]["message"]


def test_leftover_chunks_are_adopted_by_the_next_run(tmp_path: Path) -> None:
    cfg, world, gen = _run(tmp_path)
    world.copy_failures = -1
    with pytest.raises(GenerateError):
        gen.run()
    leftovers = sorted(world.csgo.glob("dm_*.dem"))
    world.copy_failures = 0
    _, _, gen2 = _run(tmp_path)
    gen2.h = world.hooks()
    gen2.run()
    assert all(not p.exists() for p in leftovers)
    out = Path(cfg.paths.demos) / "dm"
    metas = [json.loads(p.read_text()) for p in out.glob("*.json")]
    adopted = [m for m in metas if m.get("adopted")]
    assert len(adopted) == 3 and all(
        a["duration_s"] == 0 and a["start_time"] is None and a["end_time"] for a in adopted
    )
    assert all(m["source_name"] and len(m["sha256"]) == 64 for m in metas)


def _lock_unlink(monkeypatch: pytest.MonkeyPatch, pattern: str) -> None:
    real = Path.unlink

    def locked(self: Path, missing_ok: bool = False) -> None:
        if self.match(pattern):
            raise PermissionError("locked")
        real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked)


def test_undeletable_source_is_published_exactly_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, world, gen = _run(tmp_path)
    world.crash_at = [200.0]  # a session restart in between
    _lock_unlink(monkeypatch, "*/csgo/dm_*_000.dem")
    gen.run()
    out = Path(cfg.paths.demos) / "dm"
    names = sorted(p.name for p in out.glob("*.dem"))
    assert sum("_000" in n for n in names) == 1 and not any("x1" in n for n in names)
    assert (world.csgo / "dm_20261023-220000_000.dem").exists()  # still locked, but not republished


def test_published_source_left_by_a_dead_run_is_not_adopted_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, world, gen = _run(tmp_path)
    with monkeypatch.context() as m:
        _lock_unlink(m, "*/csgo/dm_*.dem")
        gen.run()  # the run "dies" with every source still in the csgo dir
    out = Path(cfg.paths.demos) / "dm"
    first = sorted(p.name for p in out.glob("*.dem"))
    assert len(first) == 3 and len(list(world.csgo.glob("dm_*.dem"))) == 3
    _, _, gen2 = _run(tmp_path)
    gen2.h = world.hooks()
    gen2.run()
    assert list(world.csgo.glob("dm_20261023-220000_*.dem")) == []  # the sources were just deleted
    second = [p.name for p in out.glob("*.dem") if p.name not in first]
    assert not any(n.startswith("dm_20261023-220000") for n in second)  # no x1 duplicates


def test_other_profiles_leftovers_are_not_adopted(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path)
    other = world.csgo / "smoke_20261023-100000_000.dem"
    other.write_bytes(b"x")
    assert not gen._ours.match(other.name) and gen._ours.match("dm_20261023-100000_000.dem")
    gen.run()
    assert other.read_bytes() == b"x"
    assert list((Path(gen.cfg.paths.demos) / "dm").glob("*smoke*")) == []


def test_truncated_chunk_that_cannot_be_copied_is_not_progress(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path)
    src = world.csgo / "dm_a_000.dem"
    src.write_bytes(b"x" * 10)
    world.copy_failures = -1
    assert gen._publish(src, "dm_a_000", T0, 100.0, truncated=True) is False
    assert gen._progress == 0 and gen.generated_s == 0 and src.exists()


def test_tiny_truncated_chunks_are_not_progress(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path)
    src = world.csgo / "dm_a_000.dem"
    src.write_bytes(b"x" * 10)
    assert gen._publish(src, "dm_a_000", T0, 10.0, truncated=True) is True
    assert gen._progress == 0 and len(gen.kept) == 1  # kept, but too short to count
    src2 = world.csgo / "dm_a_001.dem"
    src2.write_bytes(b"x" * 10)
    assert gen._publish(src2, "dm_a_001", T0, 45.0, truncated=True) is True
    assert gen._progress == 1


def test_crash_loop_of_tiny_parseable_chunks_gives_up(tmp_path: Path) -> None:
    cfg, world, gen = _run(tmp_path)
    world.crash_after_launch = 20  # 10 s after every `record`
    with pytest.raises(GenerateError):
        gen.run()
    assert len(world.launches) == cfg.generate.max_failures
    assert len(gen.kept) == cfg.generate.max_failures


def test_stale_publish_files_are_swept_at_start(tmp_path: Path) -> None:
    cfg, _world, gen = _run(tmp_path)
    out = Path(cfg.paths.demos) / "dm"
    out.mkdir(parents=True)
    for name in ("a.dem.part", "b.json.part", "orphan.json"):
        (out / name).write_text("x")
    (out / "ok.dem").write_text("x")
    (out / "ok.json").write_text("{}")
    gen.run()
    assert not (out / "a.dem.part").exists() and not (out / "b.json.part").exists()
    assert not (out / "orphan.json").exists()
    assert (out / "ok.dem").exists() and (out / "ok.json").exists()


def test_reject_failure_is_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, world, gen = _run(tmp_path)
    src = world.csgo / "dm_a_000.dem"
    src.write_bytes(b"x")
    real = os.replace
    calls = [0]

    def replace(a, b):
        if Path(b).parent.name == "_rejected":
            calls[0] += 1
            if calls[0] <= 2:  # transient: the retry succeeds
                raise PermissionError("locked")
        return real(a, b)

    monkeypatch.setattr(generate.os, "replace", replace)
    assert gen._reject(src, "dm_a_000", "bad") is True
    assert (Path(cfg.paths.demos) / "_rejected" / "dm_a_000.reason.txt").read_text() == "bad\n"
    src2 = world.csgo / "dm_a_001.dem"
    src2.write_bytes(b"x")
    calls[0] = -10_000  # always fails
    assert gen._reject(src2, "dm_a_001", "bad") is False
    assert src2.exists() and not (Path(cfg.paths.demos) / "_rejected" / "dm_a_001.reason.txt").exists()
    assert list((Path(cfg.paths.demos) / "_rejected").glob("*.part")) == []


def test_wait_closed_is_false_while_the_file_still_grows(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path)
    path = world.csgo / "x.dem"
    path.write_bytes(b"x")
    sizes = iter(range(1, 10_000))
    gen.h.file_size = lambda _: next(sizes)
    assert gen._wait_closed(path) is False
    gen.h.file_size = lambda _: 5
    assert gen._wait_closed(path) is True


def test_unique_id_never_clobbers(tmp_path: Path) -> None:
    cfg, _world, gen = _run(tmp_path)
    out = Path(cfg.paths.demos) / "dm"
    out.mkdir(parents=True)
    (out / "dm_a_000.dem").write_bytes(b"old")
    assert gen._unique_id("dm_a_000") == "dm_a_000x1"
    (out / "dm_a_000x1.json").write_text("{}")
    assert gen._unique_id("dm_a_000") == "dm_a_000x2"
    assert gen._unique_id("dm_b_000") == "dm_b_000"


def test_second_run_in_the_same_second_gets_new_ids(tmp_path: Path) -> None:
    _, _world, gen = _run(tmp_path)
    first = gen.run()
    _, _world2, gen2 = _run(tmp_path)
    second = gen2.run()
    assert not {p.name for p in first} & {p.name for p in second}
    assert all(p.read_bytes() for p in first)


def test_failed_unlink_is_retried_at_the_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, world, gen = _run(tmp_path)
    src = world.csgo / "locked.dem"
    src.write_bytes(b"x")
    real = Path.unlink

    def locked(self: Path, missing_ok: bool = False) -> None:
        if self == src:
            raise PermissionError("locked")
        real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked)
    gen._remove_source(src)
    assert gen._pending_unlink == [src] and src.exists()
    monkeypatch.setattr(Path, "unlink", real)
    gen._flush_unlinks()
    assert not src.exists()


def test_patch_guard_logs_demos_left_in_the_csgo_dir(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _, world, gen = _run(tmp_path, 0.2, "generate.expected_build=111")
    left = world.csgo / "dm_20200101-000000_000.dem"
    left.write_bytes(b"x")  # recorded by an earlier run, not yet published
    with caplog.at_level(logging.ERROR), pytest.raises(PatchGuardError):
        gen.run()
    assert "unpublished demos" in caplog.text and left.name in caplog.text
    assert left.exists()


@pytest.mark.parametrize("hours,chunk", [(0, 2), (-1, 2), (0.1, 0), (0.1, -3)])
def test_run_rejects_non_positive_durations(tmp_path: Path, hours: float, chunk: float) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError):
        generate.run(cfg, "dm", hours, chunk)


@pytest.mark.parametrize(
    "opts", [["-secure"], ["+connect", "1.2.3.4"], ["-connect", "x"], ["connect"], ["+CONNECT_lobby"]]
)
def test_connect_and_secure_options_are_refused(tmp_path: Path, opts: list[str]) -> None:
    raw = OmegaConf.create(OmegaConf.to_container(_cfg(tmp_path)))
    raw.generate.launch_options = ["-insecure", *opts]
    with pytest.raises(ValueError, match="not allowed"):
        build_launch_command(raw, "dm")
    with pytest.raises(ValueError, match="not allowed"):
        CS2Process(["cs2.exe", "-insecure", *opts], tmp_path, 5, popen=FakePopen)
    proc = CS2Process(["cs2.exe", "-insecure"], tmp_path, 5, popen=FakePopen)
    proc.cmd += opts
    with pytest.raises(ValueError, match="not allowed"), proc:
        pass
    assert FakePopen.instances == []


@pytest.mark.parametrize(
    ("name", "reply", "want"),
    [
        ("bot_difficulty", "bot_difficulty = 2\r\n", "2"),
        ("bot_difficulty", "Saved user config file\r\nbot_difficulty = 2\r\nother line\r\n", "2"),
        ("bot_difficulty", '"bot_difficulty" = "1" ( def. "1" ) min. 0.000000 max. 3.000000\n', "1"),
        ("mp_t_default_primary", "mp_t_default_primary = \r\nCLocalize::CompileString failed\r\n", ""),
        ("sv_cheats", "sv_cheats = false\r\n", "false"),
        ("bot_difficulty", "Failed to load image data from https://x/y.webm\r\n", None),
        ("bot_difficulty", "bot_difficulty_extra = 3\r\n", None),
    ],
)
def test_cvar_value_parses_cs2_replies(name: str, reply: str, want: str | None) -> None:
    """Replies as CS2 sends them over netcon (probed on the PC, Nov 5), plus the CS:GO-style quoted form."""
    assert generate.cvar_value(name, reply) == want
