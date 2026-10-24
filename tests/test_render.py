"""Tests for cs2wm.pipeline.render with fake HLAE/CS2 processes, clocks and capture directories (no HLAE)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from cs2wm.config import ConfigError, load_config
from cs2wm.pipeline import render, segment
from cs2wm.pipeline.generate import Hooks
from cs2wm.pipeline.render import LaunchResult, RenderError
from cs2wm.pipeline.segment import Life, plan_spectate, split_episodes

SETTLE = 10
DEMO = "dm-0001"


def _cfg(tmp_path: Path, *overrides: str) -> DictConfig:
    return load_config(
        overrides=[
            f"paths.root={tmp_path / 'data'}",
            "paths.hlae_dir=C:/hlae",
            "paths.cs2_dir=C:/Program Files (x86)/Steam/steamapps/common/CS",
            f"segment.settle_ticks={SETTLE}",
            "episodes.min_frames=4",
            "episodes.max_frames=10",
            "render.tick_offset=2",
            *overrides,
        ]
    )


def _lives() -> list[Life]:
    return [Life("Bot B", 2, 0, 100, 1000, 12)]


def _plan(cfg: DictConfig, lives: list[Life] | None = None, demo_id: str = DEMO) -> dict:
    segs = plan_spectate(lives or _lives(), "longest_remaining", cfg, np.random.default_rng(0))
    eps = split_episodes(segs, demo_id, "dm", cfg)
    return segment.build_render_plan(demo_id, segs, eps, cfg, {"map_name": "de_dust2"}, None, {}, 1200)


def _write_plan(cfg: DictConfig, plan: dict) -> Path:
    path = Path(cfg.paths.plans) / plan["demo_id"] / "render_plan.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


# --- cfg ---------------------------------------------------------------------------------------------------------


def test_cfg_has_every_command_at_the_right_tick_in_order(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    plan = _plan(cfg)
    lines = render.build_render_cfg(plan, cfg).splitlines()
    assert "mirv_streams add normal dcs2" in lines
    assert "mirv_streams record fps 16" in lines
    assert "mirv_streams settings edit afxDefault settings afxClassic" in lines
    start = lines.index("mirv_cmd clear")
    assert lines[start - 1] == f'mirv_streams record name "{Path(cfg.paths.raw) / DEMO}"'  # POSIX path off Windows
    expected = []
    for e in plan["mirv_cmd_schedule"]:
        t = e["mirv_tick"]
        if e["action"] == "spec":
            expected += [
                f'mirv_cmd addAtTick {t} "spec_mode 2"',
                f'mirv_cmd addAtTick {t} "spec_player {e["target"]["entity_id"]}"',
            ]
        else:
            expected.append(f'mirv_cmd addAtTick {t} "mirv_streams record {e["action"].split("_")[1]}"')
    assert lines[start + 1 : -1] == expected
    assert lines[-1].endswith('"quit"')


def test_cfg_ticks_include_tick_offset_and_quit_is_last_and_after_record_end(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    plan = _plan(cfg)
    seg = plan["segments"][0]
    text = render.build_render_cfg(plan, cfg)
    assert f'addAtTick {seg["mirv_record_start_tick"]} "mirv_streams record start"' in text
    assert seg["mirv_record_start_tick"] == seg["record_start_tick"] + 2
    quit_tick = int(text.splitlines()[-1].split()[2])
    assert quit_tick > seg["mirv_record_end_tick"]
    assert quit_tick <= seg["mirv_record_end_tick"] + cfg.render.quit_delay_ticks
    ticks = [int(line.split()[2]) for line in text.splitlines() if line.startswith("mirv_cmd addAtTick")]
    assert ticks == sorted(ticks) and ticks[-1] == quit_tick


def test_cfg_quit_is_capped_at_demo_end(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    plan = _plan(cfg)
    end = plan["segments"][-1]["mirv_record_end_tick"]
    plan["demo_end_tick"] = end + 3 - plan["tick_offset"]
    assert render.build_render_cfg(plan, cfg).splitlines()[-1] == f'mirv_cmd addAtTick {end + 3} "quit"'


def test_cfg_uses_configured_spec_fields_and_ffmpeg_preset(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "render.spec_player_field=user_id", "render.spec_mode=1", "render.output_format=ffmpeg")
    plan = _plan(cfg)
    text = render.build_render_cfg(plan, cfg)
    assert '"spec_mode 1"' in text and '"spec_player 2"' in text
    assert "mirv_streams settings edit afxDefault settings afxFfmpegLosslessBest" in text.splitlines()
    assert "mirv_streams settings edit afxDefault settings afxClassic" not in text.splitlines()


def test_cfg_fails_clearly_when_the_target_field_is_null(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    plan = _plan(cfg, [Life("Bot B", 2, 0, 100, 1000, None)])
    with pytest.raises(RenderError, match="no entity_id"):
        render.build_render_cfg(plan, cfg)


def test_cfg_rejects_slot_field_and_empty_plans_and_bad_ids(tmp_path: Path) -> None:
    with pytest.raises(RenderError, match="slot"):
        render.build_render_cfg(_plan(_cfg(tmp_path)), _cfg(tmp_path, "render.spec_player_field=slot"))
    cfg = _cfg(tmp_path)
    empty = _plan(cfg)
    empty["mirv_cmd_schedule"] = []
    with pytest.raises(RenderError, match="no segments"):
        render.build_render_cfg(empty, cfg)
    with pytest.raises(RenderError, match="demo id"):
        render.build_render_cfg(_plan(cfg, demo_id="bad id"), cfg)


def test_console_quote_and_windows_paths() -> None:
    assert render.win_path("C:/dcs2/raw", "dm-1") == r"C:\dcs2\raw\dm-1"
    assert render.win_path("/tmp/raw", "dm-1") == "/tmp/raw/dm-1"
    assert render.console_quote(r"C:\Program Files\x y\a.dem") == r'"C:\Program Files\x y\a.dem"'
    assert render.console_quote("C:\\dir\\") == '"C:\\dir"'  # a trailing backslash would eat the closing quote
    for bad in ('a"b', "a;b", "a\nb"):
        with pytest.raises(RenderError):
            render.console_quote(bad)


# --- HLAE command ------------------------------------------------------------------------------------------------


def test_hlae_command_matches_the_14_step_1_form(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cmd = render.build_hlae_command(cfg, Path("/demos/dm/dm-0001.dem"), "render_dm-0001")
    assert cmd[:4] == [r"C:\hlae\HLAE.exe", "-customLoader", "-noGui", "-autoStart"]
    assert cmd[cmd.index("-hookDllPath") + 1] == r"C:\hlae\x64\AfxHookSource2.dll"
    assert cmd[cmd.index("-programPath") + 1] == (
        r"C:\Program Files (x86)\Steam\steamapps\common\CS\game\bin\win64\cs2.exe"
    )
    assert cmd[-2] == "-cmdLine"
    line = cmd[-1]
    assert line.startswith("-steam -insecure -novid -console -sw -w 1280 -h 720 ")
    assert line.endswith(' +playdemo "/demos/dm/dm-0001.dem" +exec render_dm-0001')


def test_hlae_command_uses_capture_resolution(tmp_path: Path) -> None:
    cmd = render.build_hlae_command(_cfg(tmp_path, "resolution.capture=[540,960]"), Path("d.dem"), "render_d")
    assert "-w 960 -h 540" in cmd[-1]


def test_launch_refuses_without_insecure(tmp_path: Path) -> None:
    raw = OmegaConf.create(OmegaConf.to_container(_cfg(tmp_path)))  # load_config rejects this, so bypass it
    raw.render.launch_options = ["-steam", "-novid"]
    popen_calls: list = []
    hooks = Hooks(popen=lambda *a, **k: popen_calls.append(a))
    with pytest.raises(ValueError, match="-insecure"):
        render.launch_hlae(raw, Path("d.dem"), "render_d", 10, hooks)
    assert popen_calls == []
    with pytest.raises(ConfigError, match="insecure"):
        _cfg(tmp_path, "render.launch_options=[-steam]")


@pytest.mark.parametrize(
    "cmd_line",
    [
        "-steam -insecure -secure",
        "-steam -insecure +connect 1.2.3.4",
        "-insecure -connect 1.2.3.4",
        "-insecure connect x",
    ],
)
def test_launch_refuses_secure_and_connect_options(tmp_path: Path, cmd_line: str) -> None:
    popen_calls: list = []
    hooks = Hooks(popen=lambda *a, **k: popen_calls.append(a))
    raw = OmegaConf.create(OmegaConf.to_container(_cfg(tmp_path)))
    raw.render.launch_options = cmd_line.split()
    with pytest.raises(ValueError, match="not allowed"):
        render.launch_hlae(raw, Path("d.dem"), "render_d", 10, hooks)
    assert popen_calls == []
    with pytest.raises(ConfigError, match="not allowed"):
        _cfg(tmp_path, f"render.launch_options=[{','.join(cmd_line.split())}]")


def test_require_offline_expands_only_the_cmdline_value() -> None:
    from cs2wm.pipeline import proc

    proc.require_offline(["HLAE.exe", "-cmdLine", "-steam -insecure -novid"])
    for bad in ("-steam -novid", "-insecure +connect 1.2.3.4", "-insecure -secure"):
        with pytest.raises(ValueError):
            proc.require_offline(["HLAE.exe", "-cmdLine", bad])
    with pytest.raises(ValueError):
        proc.require_offline(["HLAE.exe", "-steam -insecure"])  # not behind -cmdLine: not a real flag
    with pytest.raises(ValueError, match="unparsable"):
        proc.require_offline(["HLAE.exe", "-cmdLine", '-insecure +playdemo "x'])


def test_a_quoted_path_containing_insecure_is_not_the_flag() -> None:
    from cs2wm.pipeline import proc

    with pytest.raises(ValueError, match="-insecure"):
        proc.require_offline(["HLAE.exe", "-cmdLine", r'-steam +playdemo "C:\a -insecure b\x.dem"'])
    proc.require_offline(["HLAE.exe", "-cmdLine", r'-insecure +playdemo "C:\a -insecure b\x.dem"'])


# --- launch / timeout ----------------------------------------------------------------------------------------------


class FakePopen:
    instances: ClassVar[list[FakePopen]] = []

    def __init__(self, cmd, **kwargs) -> None:
        self.cmd, self.kwargs = cmd, kwargs
        self.pid = 4242
        self.returncode: int | None = None
        self.killed = False
        FakePopen.instances.append(self)

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.killed, self.returncode = True, -9

    def wait(self, timeout=None):
        return self.returncode


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def _reset_popen():
    FakePopen.instances = []


def _hooks(clock: FakeClock, runs: list) -> Hooks:
    return Hooks(clock=clock, sleep=clock.sleep, popen=FakePopen, run=lambda *a, **k: runs.append(a[0]))


def test_timeout_kills_the_whole_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "_is_windows", lambda: True)
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    clock, runs = FakeClock(), []
    result = render.launch_hlae(_cfg(tmp_path), Path("d.dem"), "render_d", 30, _hooks(clock, runs), lambda: True)
    assert result == LaunchResult(None, True)
    assert clock.now >= 30
    assert FakePopen.instances[0].kwargs["creationflags"] == 512
    assert ["taskkill", "/T", "/F", "/PID", "4242"] in runs
    assert ["taskkill", "/T", "/F", "/IM", "cs2.exe"] in runs  # cs2.exe may not be HLAE's child
    assert FakePopen.instances[0].killed


def test_waits_for_cs2_after_hlae_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "_is_windows", lambda: False)
    clock, runs = FakeClock(), []
    state = {"polls": 0}

    def cs2_alive() -> bool:
        state["polls"] += 1
        return state["polls"] < 4

    def popen(cmd, **kwargs):
        p = FakePopen(cmd, **kwargs)
        p.returncode = 0  # HLAE.exe is gone at once, cs2.exe lives on
        return p

    hooks = Hooks(clock=clock, sleep=clock.sleep, popen=popen, run=lambda *a, **k: runs.append(a[0]))
    result = render.launch_hlae(_cfg(tmp_path), Path("d.dem"), "render_d", 100, hooks, cs2_alive)
    assert result == LaunchResult(0, False)
    assert state["polls"] == 4 and runs == []


def test_cs2_running_parses_tasklist(monkeypatch: pytest.MonkeyPatch) -> None:
    class Out:
        stdout = "cs2.exe     1234 Console 1 2,000,000 K\n"

    monkeypatch.setattr(render, "_is_windows", lambda: True)
    assert render.cs2_running(lambda *a, **k: Out(), "cs2.exe")
    Out.stdout = "INFO: No tasks are running which match the specified criteria.\n"
    assert not render.cs2_running(lambda *a, **k: Out(), "cs2.exe")
    monkeypatch.setattr(render, "_is_windows", lambda: False)
    assert not render.cs2_running(lambda *a, **k: Out(), "cs2.exe")


# --- captures ------------------------------------------------------------------------------------------------------


def _tga(path: Path, img: np.ndarray, bpp: int = 32, top_origin: bool = False) -> None:
    h, w = img.shape[:2]
    header = bytearray(18)
    header[2], header[12], header[13], header[14], header[15] = 2, w & 255, w >> 8, h & 255, h >> 8
    header[16], header[17] = bpp, 0x20 if top_origin else 0
    px = img if top_origin else img[::-1]
    if bpp == 32:
        px = np.dstack([px, np.full((h, w), 255, np.uint8)])
    path.write_bytes(bytes(header) + px.tobytes())


def _frame(i: int, h: int = 72, w: int = 128) -> np.ndarray:
    rng = np.random.default_rng(i)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _fake_takes(raw: Path, counts: list[int], stream: str = "dcs2") -> None:
    for t, n in enumerate(counts):
        d = raw / f"take{t:04d}" / stream
        d.mkdir(parents=True)
        for i in range(n):
            _tga(d / f"{i:05d}.tga", _frame(i))


def test_tga_decoder_handles_origin_and_depth(tmp_path: Path) -> None:
    img = _frame(1, 6, 8)
    for bpp in (24, 32):
        for top in (False, True):
            _tga(tmp_path / "a.tga", img, bpp, top)
            assert np.array_equal(render._read_tga(tmp_path / "a.tga"), img)
    (tmp_path / "bad.tga").write_bytes(b"\x00" * 18)
    with pytest.raises(ValueError, match="unsupported"):
        render._read_tga(tmp_path / "bad.tga")


def test_verify_matches_and_flags_mismatches(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    plan = _plan(cfg)
    (seg,) = plan["segments"]
    n = seg["rendered_frames"]
    raw = tmp_path / "raw"
    _fake_takes(raw, [n])
    assert render.verify_frame_counts(plan, raw, cfg) == {seg["segment_idx"]: (n, n)}
    (raw / "take0000" / "dcs2" / f"{n - 1:05d}.tga").unlink()
    assert render.verify_frame_counts(plan, raw, cfg) == {seg["segment_idx"]: (n, n - 1)}
    assert render.verify_frame_counts(plan, tmp_path / "missing", cfg) == {seg["segment_idx"]: (n, 0)}


def test_verify_maps_takes_to_segments_in_schedule_order(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    lives = [Life("A", 1, 0, 0, 200, 11), Life("B", 2, 0, 240, 700, 12)]
    plan = _plan(cfg, lives)
    assert len(plan["segments"]) == 2
    n0, n1 = (s["rendered_frames"] for s in plan["segments"])
    _fake_takes(tmp_path / "raw", [n0, n1])
    assert render.verify_frame_counts(plan, tmp_path / "raw", cfg) == {0: (n0, n0), 1: (n1, n1)}


def test_verify_counts_ffmpeg_video_frames(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "render.output_format=ffmpeg")
    plan = _plan(cfg)
    (seg,) = plan["segments"]
    d = tmp_path / "raw" / "take0000" / "dcs2"
    d.mkdir(parents=True)
    writer = cv2.VideoWriter(str(d / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 16, (128, 72))
    for i in range(seg["rendered_frames"]):
        writer.write(_frame(i))
    writer.release()
    assert render.verify_frame_counts(plan, tmp_path / "raw", cfg) == {0: (seg["rendered_frames"],) * 2}


# --- downscale ---------------------------------------------------------------------------------------------------


def test_downscale_shape_dtype_interpolation_and_raw_deleted(tmp_path: Path) -> None:
    src = tmp_path / "raw" / "take0000" / "dcs2"
    _fake_takes(tmp_path / "raw", [3])
    dst = tmp_path / "frames" / "seg000"
    assert render.downscale_frames(src, dst, (150, 280)) == 3
    assert not src.exists()
    assert sorted(p.name for p in dst.iterdir()) == ["000000.png", "000001.png", "000002.png"]
    assert not dst.with_name("seg000.tmp").exists()
    out = cv2.imread(str(dst / "000001.png"), cv2.IMREAD_UNCHANGED)
    assert out.shape == (150, 280, 3) and out.dtype == np.uint8
    assert np.array_equal(out, cv2.resize(_frame(1), (280, 150), interpolation=cv2.INTER_AREA))  # PNG is lossless


def test_downscale_keeps_raw_and_leaves_nothing_on_failure(tmp_path: Path) -> None:
    src = tmp_path / "raw" / "take0000" / "dcs2"
    _fake_takes(tmp_path / "raw", [3])
    (src / "00002.tga").write_bytes(b"junk")  # frame 2 is corrupt
    dst = tmp_path / "frames" / "seg000"
    with pytest.raises(ValueError):
        render.downscale_frames(src, dst)
    assert src.exists() and len(list(src.iterdir())) == 3
    assert not dst.exists() and not dst.with_name("seg000.tmp").exists()


def test_downscale_replaces_stale_tmp_and_old_segment_atomically(tmp_path: Path) -> None:
    _fake_takes(tmp_path / "raw", [2])
    dst = tmp_path / "frames" / "seg000"
    (dst.with_name("seg000.tmp")).mkdir(parents=True)
    (dst.with_name("seg000.tmp") / "000009.png").write_bytes(b"x")  # leftover of a crashed run
    dst.mkdir()
    (dst / "000005.png").write_bytes(b"old")
    render.downscale_frames(tmp_path / "raw" / "take0000" / "dcs2", dst)
    assert sorted(p.name for p in dst.iterdir()) == ["000000.png", "000001.png"]


def test_a_crash_mid_segment_does_not_publish_the_segment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_takes(tmp_path / "raw", [4])
    dst = tmp_path / "frames" / "seg000"
    real, calls = cv2.imwrite, []

    def flaky(path, img, params=None):
        calls.append(path)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real(path, img, params)

    monkeypatch.setattr(render.cv2, "imwrite", flaky)
    with pytest.raises(KeyboardInterrupt):
        render.downscale_frames(tmp_path / "raw" / "take0000" / "dcs2", dst)
    assert not dst.exists()
    assert (tmp_path / "raw" / "take0000" / "dcs2").exists()


# --- run ---------------------------------------------------------------------------------------------------------


class FakeLauncher:
    """Writes fake takes like HLAE would; one behaviour per call."""

    def __init__(self, plan: dict, behaviours: list[str]) -> None:
        self.plan, self.behaviours, self.calls = plan, list(behaviours), 0
        self.args: list = []

    def __call__(self, cfg, demo_path, cfg_name, timeout_s, hooks):
        behaviour = self.behaviours[min(self.calls, len(self.behaviours) - 1)]
        self.calls += 1
        self.args.append((cfg_name, timeout_s))
        raw = render.raw_dir(cfg, self.plan["demo_id"])
        counts = [s["rendered_frames"] for s in self.plan["segments"]]
        if behaviour == "ok":
            _fake_takes(raw, counts)
        elif behaviour == "short":
            _fake_takes(raw, [counts[0] - 2, *counts[1:]])
        elif behaviour == "one_short":
            _fake_takes(raw, [c - 1 for c in counts])
        elif behaviour == "timeout":
            _fake_takes(raw, [1])
            return LaunchResult(None, True)
        elif behaviour == "crash":
            return LaunchResult(1, False)
        elif behaviour == "oserror":
            raise FileNotFoundError("HLAE.exe")
        return LaunchResult(0, False)


def _setup(tmp_path: Path, *overrides: str) -> tuple[DictConfig, dict, Path]:
    cfg = _cfg(tmp_path, f"paths.cs2_dir={tmp_path / 'cs2'}", *overrides)
    plan = _plan(cfg)
    _write_plan(cfg, plan)
    return cfg, plan, tmp_path / "demos" / "dm" / f"{DEMO}.dem"


def _hooks_clock(step: float = 10.0) -> Hooks:
    t = {"now": 0.0}

    def clock() -> float:
        t["now"] += step
        return t["now"]

    return Hooks(clock=clock)


def test_run_success_writes_frames_manifest_and_installs_cfg(tmp_path: Path) -> None:
    cfg, plan, demo = _setup(tmp_path)
    launcher = FakeLauncher(plan, ["ok"])
    res = render.run(demo, cfg, _hooks_clock(), launcher)
    (seg,) = plan["segments"]
    n = seg["rendered_frames"]
    assert res.ok and res.reason is None and res.attempts == 1 and res.frames == {0: n}
    out = Path(cfg.paths.frames) / DEMO
    assert res.out_dir == out and len(list((out / "seg000").glob("*.png"))) == n
    assert not render.raw_dir(cfg, DEMO).exists()  # raw captures deleted
    installed = Path(cfg.paths.cs2_dir) / "game/csgo/cfg" / f"render_{DEMO}.cfg"
    assert installed.read_text() == render.build_render_cfg(plan, cfg)
    assert launcher.args[0][0] == f"render_{DEMO}"
    assert launcher.args[0][1] == pytest.approx(3.0 * 1200 / 64 + 180)  # timeout_factor x demo length + startup
    assert res.wall_s > 0 and res.realtime_factor == pytest.approx(n / 16 / res.wall_s)


def test_run_manifest_contents(tmp_path: Path) -> None:
    cfg, plan, demo = _setup(tmp_path)
    render.run(demo, cfg, _hooks_clock(), FakeLauncher(plan, ["ok"]))
    man = json.loads((Path(cfg.paths.frames) / DEMO / "render.json").read_text())
    (seg,) = plan["segments"]
    plan_path = Path(cfg.paths.plans) / DEMO / "render_plan.json"
    assert man["plan_sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert (man["demo_id"], man["fps"], man["size"], man["image_format"]) == (DEMO, 16, [150, 280], "png")
    (ms,) = man["segments"]
    assert ms == {
        "segment_idx": 0,
        "dir": "seg000",
        "frames": seg["rendered_frames"],
        "start_tick": seg["record_start_tick"],
        "end_tick": seg["record_end_tick"],
        "target": seg["target"],
        "episodes": [
            {"episode_id": e["episode_id"], "first_frame": e["first_frame"], "n_frames": e["n_frames"]}
            for e in seg["episodes"]
        ],
    }


def test_run_retries_once_after_a_mismatch_then_succeeds(tmp_path: Path) -> None:
    cfg, plan, demo = _setup(tmp_path)
    launcher = FakeLauncher(plan, ["short", "ok"])
    res = render.run(demo, cfg, _hooks_clock(), launcher)
    assert res.ok and res.attempts == 2 and launcher.calls == 2


@pytest.mark.parametrize(
    ("behaviour", "reason"),
    [
        ("short", "frame-count mismatch"),
        ("one_short", "end tick not captured"),
        ("timeout", "timeout"),
        ("crash", "(HLAE exit code 1)"),
        ("oserror", "start HLAE"),
    ],
)
def test_run_fails_after_exactly_one_retry_with_a_reason(tmp_path: Path, behaviour: str, reason: str) -> None:
    cfg, plan, demo = _setup(tmp_path)
    launcher = FakeLauncher(plan, [behaviour])
    res = render.run(demo, cfg, _hooks_clock(), launcher)
    assert not res.ok and launcher.calls == 2 and res.attempts == 2
    assert reason in res.reason
    assert not (Path(cfg.paths.frames) / DEMO).exists()  # no partial output
    assert not render.raw_dir(cfg, DEMO).exists()
    assert res.realtime_factor == 0


def test_run_retries_follow_config(tmp_path: Path) -> None:
    cfg, plan, demo = _setup(tmp_path, "render.retries=0")
    launcher = FakeLauncher(plan, ["crash"])
    assert not render.run(demo, cfg, _hooks_clock(), launcher).ok and launcher.calls == 1


def test_run_extra_take_is_a_mismatch(tmp_path: Path) -> None:
    cfg, plan, demo = _setup(tmp_path)
    n = plan["segments"][0]["rendered_frames"]

    def launcher(cfg, demo_path, cfg_name, timeout_s, hooks):
        _fake_takes(render.raw_dir(cfg, DEMO), [n, 5])
        return LaunchResult(0, False)

    res = render.run(demo, cfg, _hooks_clock(), launcher)
    assert not res.ok and "2 takes captured for 1 segments" in res.reason


def test_run_fails_without_launching_for_bad_plans(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, f"paths.cs2_dir={tmp_path / 'cs2'}")
    launcher = FakeLauncher({"demo_id": DEMO, "segments": []}, ["ok"])
    demo = tmp_path / f"{DEMO}.dem"
    assert "bad render plan" in render.run(demo, cfg, _hooks_clock(), launcher).reason  # no plan file
    cfg2 = _cfg(tmp_path, f"paths.cs2_dir={tmp_path / 'cs2'}", "render.spec_player_field=user_id")
    plan = _plan(cfg2, [Life("Bot B", 2, 0, 100, 1000, 12)])
    plan["segments"][0]["target"]["user_id"] = None
    plan["mirv_cmd_schedule"][0]["target"]["user_id"] = None
    _write_plan(cfg2, plan)
    res = render.run(demo, cfg2, _hooks_clock(), launcher)
    assert not res.ok and "no user_id" in res.reason and launcher.calls == 0


def test_run_downscale_failure_is_retried_and_reported(tmp_path: Path) -> None:
    cfg, plan, demo = _setup(tmp_path)

    def launcher(cfg, demo_path, cfg_name, timeout_s, hooks):
        raw = render.raw_dir(cfg, DEMO)
        _fake_takes(raw, [plan["segments"][0]["rendered_frames"]])
        (raw / "take0000" / "dcs2" / "00000.tga").write_bytes(b"junk")
        return LaunchResult(0, False)

    res = render.run(demo, cfg, _hooks_clock(), launcher)
    assert not res.ok and res.reason.startswith("downscale failed")


# --- review fixes ----------------------------------------------------------------------------------------------


def test_waits_for_a_late_cs2_and_gives_up_if_it_never_shows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "_is_windows", lambda: False)

    def popen(cmd, **kwargs):
        p = FakePopen(cmd, **kwargs)
        p.returncode = 0
        return p

    cfg = _cfg(tmp_path)
    seq = iter([False, False, True, True, False])  # cs2.exe appears two polls after HLAE exited
    clock = FakeClock()
    hooks = Hooks(clock=clock, sleep=clock.sleep, popen=popen, run=lambda *a, **k: None)
    assert render.launch_hlae(cfg, Path("d.dem"), "r", 100, hooks, lambda: next(seq)) == LaunchResult(0, False)
    assert next(seq, "drained") == "drained"  # polled until cs2.exe was seen AND gone

    clock2, polls = FakeClock(), []
    hooks2 = Hooks(clock=clock2, sleep=clock2.sleep, popen=popen, run=lambda *a, **k: None)
    res = render.launch_hlae(cfg, Path("d.dem"), "r", 100, hooks2, lambda: polls.append(1) or False)
    assert res == LaunchResult(0, False)
    assert (
        clock2.now == 10 and len(polls) == 6
    )  # grace = 5 x poll_s, then it stops waiting for a cs2.exe that never came


def test_takes_are_ordered_numerically(tmp_path: Path) -> None:
    for name in ("take10000", "take9999", "take0001", "take0000"):
        (tmp_path / name).mkdir()
    assert [p.name for p in render.list_takes(tmp_path)] == ["take0000", "take0001", "take9999", "take10000"]


def test_run_reports_an_unwritable_cs2_cfg_dir(tmp_path: Path) -> None:
    cfg, plan, demo = _setup(tmp_path)
    (tmp_path / "cs2").write_text("a file, so game/csgo/cfg cannot be created")
    launcher = FakeLauncher(plan, ["ok"])
    res = render.run(demo, cfg, _hooks_clock(), launcher)
    assert not res.ok and "could not install the render cfg" in res.reason and launcher.calls == 0


def test_stream_name_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="stream_name"):
        _cfg(tmp_path, "render.stream_name=a b")
    raw = OmegaConf.create(OmegaConf.to_container(_cfg(tmp_path)))
    raw.render.stream_name = 'x"; quit'
    with pytest.raises(RenderError, match="stream_name"):
        render.build_render_cfg(_plan(_cfg(tmp_path)), raw)


def test_cfg_quotes_a_raw_dir_with_spaces(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "paths.raw=C:/my data/raw")
    assert r'mirv_streams record name "C:\my data\raw\dm-0001"' in render.build_render_cfg(_plan(cfg), cfg).splitlines()
    cmd = render.build_hlae_command(_cfg(tmp_path), Path("/a b/x.dem"), "render_d")
    assert '+playdemo "/a b/x.dem"' in cmd[-1]


def test_cfg_only_writes_files_and_builds_the_command_without_a_demo(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, f"paths.cs2_dir={tmp_path / 'cs2'}")
    plan_path = _write_plan(cfg, _plan(cfg))
    out = render.write_cfg_only(plan_path, cfg)  # the demo file does not exist
    text = render.build_render_cfg(_plan(cfg), cfg)
    assert out.installed.read_text() == text == out.copy.read_text()
    assert out.copy == plan_path.with_name(f"render_{DEMO}.cfg") and out.installed.name == f"render_{DEMO}.cfg"
    assert out.argv[0] == r"C:\hlae\HLAE.exe" and "-insecure" in out.argv[-1]
    assert out.powershell.startswith(r"& 'C:\hlae\HLAE.exe' --% -customLoader -noGui -autoStart")
    assert out.powershell.endswith(f' +exec render_{DEMO}"')
    assert FakePopen.instances == []


def test_cli_render_cfg_only_with_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from cs2wm import cli

    cfg = _cfg(tmp_path, f"paths.cs2_dir={tmp_path / 'cs2'}")
    plan_path = _write_plan(cfg, _plan(cfg))
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text(OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(cfg))), encoding="utf-8")
    cli.main(["--config", str(yaml), "render", "--cfg-only", "--plan", str(plan_path)])
    out = capsys.readouterr().out
    assert "PowerShell:" in out and "HLAE.exe" in out and f"render_{DEMO}.cfg" in out
