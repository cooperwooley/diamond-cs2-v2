"""Tests for cs2wm.pipeline.generate with a fake CS2, clock and netcon (no CS2, no sleeping)."""

import hashlib
import json
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
            "generate.map_load_wait_s=10",
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
    proc.kill()  # already dead: no second taskkill
    assert len(ran) == 1


# --- cfg -----------------------------------------------------------------------------------------------------------


def test_render_cfg_inlines_exec_includes(tmp_path: Path) -> None:
    text = render_cfg("smoke", _cfg(tmp_path))
    assert "exec generate_dm" not in text
    assert "bot_quota 10" in text  # from generate_dm.cfg
    assert "mp_ct_default_grenades weapon_smokegrenade" in text  # from generate_smoke.cfg
    assert "\ntv_autorecord 1" not in text  # netcon is the default; the cfg only has it commented out


def test_render_cfg_gotv_appends_autorecord(tmp_path: Path) -> None:
    text = render_cfg("dm", _cfg(tmp_path, "generate.chunking=gotv", "generate.chunk_min=7.5"))
    assert text.splitlines()[-3:] == ["tv_enable 1", "tv_autorecord 1", "mp_timelimit 7.5"]


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

    def __init__(self, csgo: Path, *, chunking: str = "netcon") -> None:
        self.csgo = csgo
        self.chunking = chunking
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
            if self.crash_at and self.t >= self.crash_at[0] and self.alive():
                self.crash_at.pop(0)
                self.proc.returncode = 1
            if self.hang_at and self.t >= self.hang_at[0] and self.alive():
                self.hang_at.pop(0)
                self.hung = True
            if self.alive() and not self.hung:
                self._grow(step)

    def _grow(self, step: float) -> None:
        if self.chunking == "gotv":
            n = int((self.t - self.launch_t) // 120)
            path = self.csgo / f"auto{self.launch_t:.0f}-{n}.dem"
            with path.open("ab") as f:
                f.write(b"x" * int(step * 100))
        elif self.recording is not None:
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
                if cmd.startswith("record "):
                    world.recording = world.csgo / f"{cmd.split()[1]}.dem"
                    world.recording.touch()
                elif cmd == "stop":
                    world.recording = None
                return ""

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
            notify=lambda msg, **kw: world.notified.append({"message": msg, **kw}),
            git_sha=lambda: "deadbeef",
        )


def _run(tmp_path: Path, hours: float = 0.1, *overrides: str, profile: str = "dm", chunk_min: float = 2):
    cfg = _cfg(tmp_path, "generate.hang_timeout_min=1", *overrides)
    world = FakeWorld(Path(cfg.paths.cs2_dir) / "game/csgo", chunking=cfg.generate.chunking)
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
    assert side["chunking"] == "netcon"
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
    assert any(json.loads(p.with_suffix(".json").read_text())["truncated"] for p in kept)


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


def test_gotv_mode_collects_autorecorded_demos(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.09, "generate.chunking=gotv", "generate.poll_s=1")
    kept = gen.run()
    assert len(kept) == 3
    assert "-netconport" not in world.launches[0].cmd
    assert "tv_autorecord 1" in (world.csgo / "cfg/generate_dm.cfg").read_text()
    sides = [json.loads(p.with_suffix(".json").read_text()) for p in kept]
    assert [s["truncated"] for s in sides] == [False, False, True]
    assert all(s["chunking"] == "gotv" for s in sides)
    assert list(world.csgo.glob("*.dem")) == []


def test_old_demos_in_csgo_dir_are_left_alone(tmp_path: Path) -> None:
    _, world, gen = _run(tmp_path, 0.09, "generate.chunking=gotv", "generate.poll_s=1")
    old = world.csgo / "mine.dem"
    old.write_bytes(b"keep me")
    gen.run()
    assert old.read_bytes() == b"keep me"


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
