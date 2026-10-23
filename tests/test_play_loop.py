"""Tests for the world-model step, the benchmark and a headless PlaySession (tiny random models, CPU)."""

import json
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from agent import Agent
from cs2wm.actions import ACTION_DIM, decode
from cs2wm.config import load_config
from cs2wm.pipeline.pack import to_low_res, write_episode
from cs2wm.play import loop
from cs2wm.play.agent import DEFAULT_AGENT_CONFIG_PATH
from cs2wm.play.input import InputMapper
from cs2wm.play.world_model import (
    WorldModel,
    blank_context,
    frames_to_tensor,
    load_context,
    load_sampler_cfg,
    tensor_to_rgb_uint8,
)

H, W = 150, 280


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def agent(tiny_model_dotlist) -> Agent:
    agent_cfg = OmegaConf.merge(OmegaConf.load(DEFAULT_AGENT_CONFIG_PATH), OmegaConf.from_dotlist(tiny_model_dotlist()))
    torch.manual_seed(0)
    return Agent(instantiate(agent_cfg, num_actions=ACTION_DIM)).eval().requires_grad_(False)


@pytest.fixture
def episode(tmp_path, cfg):
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (6, H, W, 3), dtype=np.uint8)
    actions = rng.integers(0, 2, (6, ACTION_DIM)).astype(np.float32)
    meta = {
        "episode_id": "ep0",
        "demo_id": "d",
        "kind": "dm",
        "player": None,
        "start_tick": 0,
        "cs2_build": "1",
        "demoparser2_version": "0.42.0",
        "cs2wm_git_sha": "abc",
    }
    path = write_episode(frames, actions, meta, tmp_path / "hdf5", cfg).low_res_path
    return path, to_low_res(frames), actions


def idle(cfg):
    return blank_context(1, cfg.resolution.low_res, loop.ActionSpec.from_config(cfg))[1][0]


def test_sampler_presets() -> None:
    fast, hq = load_sampler_cfg(1), load_sampler_cfg(3)
    assert (fast.num_steps_denoising, hq.num_steps_denoising) == (1, 3)
    assert fast.s_cond == 0.005 and fast.s_tmax == float("inf")
    assert load_sampler_cfg(3, s_cond=0.0).s_cond == 0.0
    with pytest.raises(ValueError, match="denoising_steps"):
        load_sampler_cfg(2)


def test_load_context_converts_bgr_to_rgb_in_range(episode) -> None:
    path, low_bgr, actions = episode
    frames, acts = load_context(path, 4)
    assert frames.shape == (4, 3, 30, 56) and frames.dtype == torch.float32
    assert frames.min() >= -1 and frames.max() <= 1
    np.testing.assert_allclose(frames[0, 0].numpy(), low_bgr[0, :, :, 2] / 255 * 2 - 1, atol=1e-6)  # R = BGR[2]
    np.testing.assert_array_equal(acts, actions[:4])
    with pytest.raises(ValueError, match="at least 9"):
        load_context(path, 9)


def test_tensor_to_rgb_uint8_clamps_instead_of_wrapping() -> None:
    x = torch.tensor([-1.5, -1.0, 0.0, 1.0, 1.2]).reshape(1, 1, 5).expand(3, 1, 5)
    out = tensor_to_rgb_uint8(x)
    assert out.dtype == np.uint8 and out.shape == (1, 5, 3)
    assert out[0, :, 0].tolist() == [0, 0, 128, 255, 255]


def test_blank_context(cfg) -> None:
    frames, acts = blank_context(4, cfg.resolution.low_res, loop.ActionSpec.from_config(cfg))
    assert frames.shape == (4, 3, 30, 56) and acts.shape == (4, ACTION_DIM)
    assert decode(acts[0], loop.ActionSpec.from_config(cfg)).keys == ()


@pytest.mark.parametrize("use_up", [False, True])
@pytest.mark.parametrize("steps", [1, 3])
def test_step_shape_range_and_rolling_context(agent, cfg, use_up, steps) -> None:
    world = WorldModel(agent, steps, use_upsampler=use_up)
    assert world.n_cond == 4 and world.uses_upsampler == use_up
    frames = torch.rand(4, 3, 30, 56) * 2 - 1
    acts = np.zeros((4, ACTION_DIM), np.float32)
    world.reset(frames, acts)
    assert world.full_res.shape == (3, H, W)
    action = idle(cfg).copy()
    action[0] = 1.0  # w
    out = world.step(action)
    assert out.shape == (3, H, W) and out.min() >= -1 and out.max() <= 1
    assert world.low_res.shape == (3, 30, 56)
    # The context rolled: oldest frame dropped, newest appended; the new action sits in the second-to-last slot.
    assert torch.equal(world._ctx_obs[:3], frames[1:])
    assert torch.equal(world._ctx_obs[3], world.low_res)
    assert world._ctx_act[2, 0] == 1.0
    first_low = world.low_res.clone()
    assert world.step(idle(cfg)).shape == (3, H, W)
    assert torch.equal(world._ctx_obs[2], first_low)
    assert world._ctx_act[2, 0] == 0.0 and world._ctx_act[1, 0] == 1.0


def test_first_action_replaces_the_last_context_action(agent, cfg) -> None:
    world = WorldModel(agent, 1, use_upsampler=False)
    world.reset(torch.zeros(4, 3, 30, 56), np.ones((4, ACTION_DIM), np.float32))
    world.step(idle(cfg))
    assert torch.equal(world._ctx_act[2], torch.from_numpy(idle(cfg)))
    assert world._ctx_act[0].sum() == ACTION_DIM


def test_restart_restores_the_initial_context(agent, cfg) -> None:
    world = WorldModel(agent, 1, use_upsampler=True)
    frames = torch.rand(4, 3, 30, 56) * 2 - 1
    world.reset(frames, np.zeros((4, ACTION_DIM), np.float32))
    start_full = world.full_res.clone()
    for _ in range(2):
        world.step(idle(cfg))
    world.restart()
    assert torch.equal(world._ctx_obs, frames) and torch.equal(world.full_res, start_full)


def test_world_model_errors(agent) -> None:
    with pytest.raises(RuntimeError, match="before reset"):
        WorldModel(agent, 1, use_upsampler=False).step(np.zeros(ACTION_DIM, np.float32))
    with pytest.raises(ValueError, match="CUDA"):
        WorldModel(agent, 1, bf16=True)
    with pytest.raises(ValueError, match="no upsampler"):
        WorldModel(SimpleNamespace(upsampler=None), 1, use_upsampler=True)
    with pytest.raises(ValueError, match="context frames"):
        WorldModel(agent, 1, use_upsampler=False).reset(torch.zeros(2, 3, 30, 56), np.zeros((2, ACTION_DIM)))


def test_frames_to_tensor_channel_order() -> None:
    bgr = np.zeros((1, 2, 2, 3), np.uint8)
    bgr[..., 0] = 255  # blue
    t = frames_to_tensor(bgr)
    assert t[0, 2].min() == 1.0 and t[0, 0].max() == -1.0


def test_benchmark_on_cpu_writes_valid_json(agent, cfg, tmp_path) -> None:
    out = tmp_path / "results" / "play_fps.json"
    report = loop.benchmark(cfg, tmp_path / "ckpt.pt", tmp_path / "v1.pt", 2, out, agent=agent)
    assert json.loads(out.read_text()) == report
    assert report["device"] == "cpu" and report["gpu"] is None and report["torch"] == torch.__version__
    res = report["results"]
    assert len(res) == 8
    ok = {k: v for k, v in res.items() if v["status"] == "ok"}
    assert set(ok) == {f"{m}_{s}step_fp32" for m in ("denoiser", "upsampler") for s in (1, 3)}
    for v in ok.values():
        assert v["fps"] > 0 and v["ms_per_frame"] == pytest.approx(1000 / v["fps"])
    assert all(v["status"] == "skipped" and "CUDA" in v["reason"] for k, v in res.items() if k.endswith("bf16"))


def test_benchmark_skips_upsampler_configs_without_one(agent, cfg, tmp_path, monkeypatch) -> None:
    denoiser_only = SimpleNamespace(upsampler=None, denoiser=agent.denoiser, device=agent.device)
    report = loop.benchmark(cfg, tmp_path / "c.pt", None, 1, tmp_path / "o.json", agent=denoiser_only)
    assert report["upsampler_ckpt"] is None
    assert report["results"]["upsampler_1step_fp32"] == {"status": "skipped", "reason": "no upsampler checkpoint"}
    assert report["results"]["denoiser_1step_fp32"]["status"] == "ok"


def test_run_requires_an_upsampler_choice(cfg, tmp_path) -> None:
    with pytest.raises(SystemExit, match="--upsampler-ckpt"):
        loop.run(cfg, tmp_path / "c.pt")


def test_run_dispatches_benchmark(cfg, tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loop, "benchmark", lambda *a, **k: calls.append(a))
    loop.run(cfg, tmp_path / "c.pt", None, benchmark_frames=7)
    assert calls[0][3] == 7 and calls[0][4].as_posix() == "results/play_fps.json"


@pytest.fixture
def dummy_pygame(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pg = pytest.importorskip("pygame")
    yield pg
    pg.quit()


def scripted_events(pg, per_frame):
    frames = iter(per_frame)
    return lambda: next(frames, [pg.event.Event(pg.QUIT)])


def test_play_session_headless_with_record(dummy_pygame, agent, cfg, episode, tmp_path, monkeypatch) -> None:
    import cv2

    pg = dummy_pygame
    key = lambda t, k: pg.event.Event(t, key=k)
    seen = []
    events = [
        [pg.event.Event(pg.MOUSEMOTION, rel=(30, 5), pos=(0, 0), buttons=(0, 0, 0)), key(pg.KEYDOWN, pg.K_w)],
        [key(pg.KEYDOWN, pg.K_F1)],  # overlay off
        [key(pg.KEYDOWN, pg.K_BACKSPACE), key(pg.KEYDOWN, pg.K_4)],  # reset the context
        [key(pg.KEYDOWN, pg.K_F1)],  # overlay back on
        [key(pg.KEYDOWN, pg.K_ESCAPE)],
    ]
    out = tmp_path / "rec" / "session.mp4"
    session = loop.PlaySession(
        cfg,
        tmp_path / "c.pt",
        tmp_path / "v1.pt",
        1,
        episode[0],
        out,
        agent=agent,
        event_source=scripted_events(pg, events),
        max_frames=50,
    )
    original = InputMapper.frame_action

    def spy(self):
        seen.append(original(self))
        return seen[-1]

    monkeypatch.setattr(InputMapper, "frame_action", spy)
    session.run()
    assert len(seen) == 5  # Esc stops the loop after its frame
    spec = loop.ActionSpec.from_config(cfg)
    assert decode(seen[0], spec).keys == ("w",)
    assert decode(seen[2], spec).keys == ("3",)  # Backspace dropped the held w
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    ok, frame = cap.read()
    assert ok and frame.shape == (H * 4, W * 4, 3)
    cap.release()


class FakeClock:
    """Advances ``step`` seconds per call, plus ``jump`` seconds from call number ``jump_at`` on."""

    def __init__(self, step: float = 0.1, jump_at: int | None = None, jump: float = 0.0) -> None:
        self.calls, self.step, self.jump_at, self.jump = 0, step, jump_at, jump

    def __call__(self) -> float:
        self.calls += 1
        extra = self.jump if self.jump_at is not None and self.calls >= self.jump_at else 0.0
        return self.calls * self.step + extra


def make_session(cfg, agent, tmp_path, **kw):
    return loop.PlaySession(cfg, tmp_path / "c.pt", tmp_path / "v1.pt", agent=agent, **kw)


def test_record_plays_in_real_time_when_the_loop_is_slow(dummy_pygame, agent, cfg, tmp_path) -> None:
    import cv2

    clock = FakeClock(step=0.1)  # call 1 starts the session; each loop then makes 2 calls, so 0.2 s per frame (5 FPS)
    out = tmp_path / "slow.mp4"
    make_session(cfg, agent, tmp_path, record=out, event_source=list, max_frames=10, clock=clock).run()
    elapsed = 0.2 * 10
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(16)
    cap.release()
    assert abs(n - elapsed * 16) <= 1


class FakeWriter:
    instances: ClassVar[list] = []

    def __init__(self, *args) -> None:
        self.released, self.written = False, 0
        FakeWriter.instances.append(self)

    def isOpened(self) -> bool:
        return True

    def write(self, frame) -> None:
        self.written += 1

    def release(self) -> None:
        self.released = True


def spy_grab(pg, monkeypatch) -> list:
    calls = []
    real = pg.event.set_grab
    monkeypatch.setattr(pg.event, "set_grab", lambda v: (calls.append(v), real(v))[1])
    return calls


def test_failure_mid_loop_still_cleans_up(dummy_pygame, agent, cfg, tmp_path, monkeypatch) -> None:
    import cv2

    pg = dummy_pygame
    FakeWriter.instances = []
    monkeypatch.setattr(cv2, "VideoWriter", FakeWriter)
    grabs = spy_grab(pg, monkeypatch)
    calls = iter(range(10))

    def events():
        if next(calls) == 2:
            raise RuntimeError("boom")
        return []

    session = make_session(cfg, agent, tmp_path, record=tmp_path / "x.mp4", event_source=events, max_frames=10)
    with pytest.raises(RuntimeError, match="boom"):
        session.run()
    assert FakeWriter.instances[0].released and FakeWriter.instances[0].written == 2
    assert grabs[-1] is False and not pg.get_init()


def test_failure_during_writer_setup_still_releases_the_grab(dummy_pygame, agent, cfg, tmp_path, monkeypatch) -> None:
    import cv2

    pg = dummy_pygame

    def broken(*args):
        raise OSError("no codec")

    monkeypatch.setattr(cv2, "VideoWriter", broken)
    grabs = spy_grab(pg, monkeypatch)
    session = make_session(cfg, agent, tmp_path, record=tmp_path / "x.mp4", event_source=list, max_frames=3)
    with pytest.raises(OSError, match="no codec"):
        session.run()
    assert grabs == [True, False] and not pg.get_init()


def test_stall_drops_the_accumulated_mouse_motion(dummy_pygame, agent, cfg, tmp_path, monkeypatch, capsys) -> None:
    pg = dummy_pygame
    motion = lambda: pg.event.Event(pg.MOUSEMOTION, rel=(500, 0), pos=(0, 0), buttons=(0, 0, 0))
    seen = []
    original = InputMapper.frame_action
    monkeypatch.setattr(InputMapper, "frame_action", lambda self: seen.append(original(self)) or seen[-1])
    # Calls: 1 = session start, then 2 per frame (start, end). Frame 2 starts at call 4: a 5 s stall precedes it.
    clock = FakeClock(jump_at=4, jump=5.0)
    events = scripted_events(pg, [[motion()], [motion()], [motion()]])
    make_session(cfg, agent, tmp_path, event_source=events, max_frames=3, clock=clock).run()
    spec = loop.ActionSpec.from_config(cfg)
    xs = [decode(a, spec).mouse_x_deg for a in seen]
    assert xs[0] > 0 and xs[1] == 0.0 and xs[2] > 0
    assert "dropped the mouse motion" in capsys.readouterr().out


def test_warmup_step_and_first_frame_is_not_polluted(dummy_pygame, agent, cfg, tmp_path, monkeypatch) -> None:
    pg = dummy_pygame
    steps = []
    original = WorldModel.step
    monkeypatch.setattr(WorldModel, "step", lambda self, a: steps.append(a) or original(self, a))
    # Motion queued before the loop (e.g. during warm-up) must be discarded.
    pg.init()
    pg.event.post(pg.event.Event(pg.MOUSEMOTION, rel=(900, 900), pos=(0, 0), buttons=(0, 0, 0)))
    session = make_session(cfg, agent, tmp_path, max_frames=1)
    session.run()
    assert len(steps) == 2  # warm-up + the one real frame
    spec = loop.ActionSpec.from_config(cfg)
    assert decode(steps[1], spec).mouse_x_deg == 0.0


def test_step_passes_context_in_the_right_order_to_the_samplers(agent, cfg) -> None:
    world = WorldModel(agent, 1, use_upsampler=True)
    frames = torch.rand(4, 3, 30, 56) * 2 - 1
    acts = np.zeros((4, ACTION_DIM), np.float32)
    world.reset(frames, acts)
    prev_full = world.full_res.clone()
    calls = {"den": [], "up": []}
    for key, sampler in (("den", world.sampler), ("up", world.up_sampler)):
        real = sampler.sample
        sampler.sample = lambda obs, act, real=real, key=key: (calls[key].append((obs.clone(), act)), real(obs, act))[1]
    action = idle(cfg).copy()
    action[0] = 1.0
    world.step(action)

    obs, act = calls["den"][0]
    assert obs.shape == (1, 4, 3, 30, 56) and torch.equal(obs[0], frames)
    assert act.shape == (1, 4, ACTION_DIM)
    assert torch.equal(act[0, -1], torch.from_numpy(action))  # the newest action is last
    assert act[0, :3].sum() == 0

    cond, up_act = calls["up"][0]
    assert up_act is None and cond.shape == (1, 2, 3, 150, 280)
    assert torch.equal(cond[0, 0], prev_full)  # previous full-res frame first
    expected = torch.nn.functional.interpolate(world.low_res[None], scale_factor=5, mode="bicubic")[0]
    torch.testing.assert_close(cond[0, 1], expected)  # then the bicubic of the new low-res frame


def test_benchmark_notes_what_is_excluded(agent, cfg, tmp_path) -> None:
    report = loop.benchmark(cfg, tmp_path / "c.pt", tmp_path / "v.pt", 1, tmp_path / "o.json", agent=agent)
    assert "excludes display" in report["note"]
