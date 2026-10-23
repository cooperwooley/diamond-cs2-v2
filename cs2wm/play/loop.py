"""Interactive play loop and FPS benchmark (#28, #35).

Opens a pygame window with relative mouse mode, feeds ``InputMapper`` actions to the denoiser (and optionally the v1
upsampler), draws the input overlay and can record to MP4 (``--record``, #35). pygame is imported inside the functions
that need it, so this module imports headless.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import DictConfig

from cs2wm.actions import ActionSpec, decode
from cs2wm.play.agent import build_denoiser_agent, build_play_agent
from cs2wm.play.input import CMD_QUIT, CMD_RESET, CMD_TOGGLE_OVERLAY, InputMapper
from cs2wm.play.world_model import WorldModel, blank_context, load_context, tensor_to_rgb_uint8

if TYPE_CHECKING:
    from agent import Agent

WARMUP_STEPS = 3
_OVERLAY_ROWS = (("w", "a", "s", "d"), ("space", "ctrl", "shift"), ("1", "2", "3", "r"), ("lmb", "rmb"))
_KEY_LABELS = {"space": "SPC", "ctrl": "CTRL", "shift": "SHIFT"}


def load_agent(ckpt: Path, upsampler_ckpt: Path | None, device: torch.device | str) -> Agent:
    """Load the denoiser (and the v1 upsampler when ``upsampler_ckpt`` is given) in eval mode.

    Args:
        ckpt: v2 agent checkpoint.
        upsampler_ckpt: v1 checkpoint providing the upsampler, or None for a denoiser-only agent.
        device: Device to put the agent on.

    Returns:
        The agent.
    """
    if upsampler_ckpt is None:
        return build_denoiser_agent(ckpt, device)
    return build_play_agent(ckpt, upsampler_ckpt, device)


def initial_context(cfg: DictConfig, n: int, context: Path | None) -> tuple[torch.Tensor, np.ndarray]:
    """Frames and actions the model starts from (and returns to on ``Backspace``).

    Args:
        cfg: A validated cs2wm config.
        n: Context length.
        context: Held-out HDF5 episode, or None for a blank grey context.

    Returns:
        ``(frames[n, 3, h, w], actions[n, 51])``.
    """
    if context is not None:
        return load_context(context, n)
    print(
        "No --context given: starting from a blank grey frame with idle actions. The model is out of distribution "
        "for the first frames; pass a held-out test HDF5 with --context for a realistic start."
    )
    return blank_context(n, cfg.resolution.low_res, ActionSpec.from_config(cfg))


def draw_overlay(surface: Any, action: np.ndarray, spec: ActionSpec, fps: float) -> None:
    """Draw pressed keys, the mouse bins as arrows and the FPS onto a pygame surface (bottom-left and top-left).

    Args:
        surface: Target ``pygame.Surface``.
        action: ``float[51]`` action of the frame just played.
        spec: Action layout.
        fps: Achieved frames per second.
    """
    import pygame as pg

    pg.font.init()
    font = pg.font.Font(None, max(18, surface.get_height() // 22))
    d = decode(action, spec)
    active = set(d.keys) | ({"lmb"} if d.lmb else set()) | ({"rmb"} if d.rmb else set())
    h = surface.get_height()
    box = font.get_height() + 8
    y = h - 8 - box * len(_OVERLAY_ROWS)
    for row in _OVERLAY_ROWS:
        x = 8
        for name in row:
            text = font.render(_KEY_LABELS.get(name, name.upper()), True, (255, 255, 255))
            rect = pg.Rect(x, y, max(box, text.get_width() + 12), box - 2)
            fill = (230, 120, 20) if name in active else (30, 30, 30)
            pg.draw.rect(surface, fill, rect, border_radius=4)
            pg.draw.rect(surface, (200, 200, 200), rect, width=1, border_radius=4)
            surface.blit(text, text.get_rect(center=rect.center))
            x = rect.right + 4
        y += box

    # Mouse bins as an arrow from the centre of a dial: +x right, +y down; length saturates at the outermost bin.
    r = h // 8
    cx, cy = surface.get_width() - r - 12, h - r - 12
    pg.draw.circle(surface, (30, 30, 30), (cx, cy), r)
    pg.draw.circle(surface, (200, 200, 200), (cx, cy), r, width=1)
    ex = d.mouse_x_deg / abs(spec.mouse_x_centers_deg[-1])
    ey = d.mouse_y_deg / abs(spec.mouse_y_centers_deg[-1])
    # sqrt keeps the small, common bins visible next to the +-1000-unit extremes.
    tip = (cx + int(np.sign(ex) * np.sqrt(abs(ex)) * r), cy + int(np.sign(ey) * np.sqrt(abs(ey)) * r))
    pg.draw.line(surface, (230, 120, 20), (cx, cy), tip, width=3)
    pg.draw.circle(surface, (230, 120, 20), tip, 4)
    label = font.render(f"x {d.mouse_x_deg:+.2f}  y {d.mouse_y_deg:+.2f} deg", True, (255, 255, 255))
    surface.blit(label, label.get_rect(midbottom=(cx, cy - r - 2)))
    surface.blit(font.render(f"{fps:4.1f} FPS", True, (255, 255, 0)), (8, 8))


class PlaySession:
    """One interactive session.

    Args:
        cfg: A validated cs2wm config.
        ckpt: v2 agent checkpoint.
        upsampler_ckpt: v1 checkpoint providing the upsampler, or None to display bicubic.
        denoising_steps: 1 or 3.
        context: Held-out HDF5 episode for the 4-frame initial context (``Backspace`` resets).
        record: Output MP4 with the overlay drawn in.
        no_upsampler: Display bicubic even if ``upsampler_ckpt`` is given.
        bf16: Sample under bf16 autocast (CUDA only).
        compile: ``torch.compile`` the models.
        agent: A ready agent, used instead of loading the checkpoints (tests).
        event_source: Returns the frame's pygame events; defaults to ``pygame.event.get`` (tests).
        max_frames: Stop after this many frames (tests); None plays until Esc or the window is closed.
    """

    def __init__(
        self,
        cfg: DictConfig,
        ckpt: Path,
        upsampler_ckpt: Path | None = None,
        denoising_steps: int = 1,
        context: Path | None = None,
        record: Path | None = None,
        no_upsampler: bool = False,
        bf16: bool = False,
        compile: bool = False,
        agent: Agent | None = None,
        event_source: Callable[[], Iterable[Any]] | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.ckpt = ckpt
        self.upsampler_ckpt = upsampler_ckpt
        self.denoising_steps = denoising_steps
        self.context = context
        self.record = record
        self.no_upsampler = no_upsampler
        self.bf16 = bf16
        self.compile = compile
        self.agent = agent
        self.event_source = event_source
        self.max_frames = max_frames
        self.fps = 0.0

    def _build_world_model(self) -> WorldModel:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_upsampler = not self.no_upsampler and (self.agent is not None or self.upsampler_ckpt is not None)
        if self.agent is None:
            self.agent = load_agent(self.ckpt, self.upsampler_ckpt if use_upsampler else None, device)
        world = WorldModel(
            self.agent,
            self.denoising_steps,
            use_upsampler=use_upsampler and self.agent.upsampler is not None,
            bf16=self.bf16,
            compile=self.compile,
        )
        world.reset(*initial_context(self.cfg, world.n_cond, self.context))
        return world

    def run(self) -> None:
        """Run until Esc, the window is closed or ``max_frames`` is reached (``F1`` toggles the overlay)."""
        import cv2
        import pygame as pg

        cfg, spec = self.cfg, ActionSpec.from_config(self.cfg)
        world = self._build_world_model()
        mapper = InputMapper(cfg)
        h, w = (int(v) for v in cfg.resolution.render)
        size = (w * int(cfg.play.window_scale), h * int(cfg.play.window_scale))

        pg.init()
        screen = pg.display.set_mode(size)
        pg.display.set_caption("cs2wm play  |  Esc quit  Backspace reset  F1 overlay")
        pg.mouse.set_visible(False)
        pg.event.set_grab(True)
        if hasattr(pg.mouse, "set_relative_mode"):
            pg.mouse.set_relative_mode(True)
        get_events = self.event_source or pg.event.get

        writer = None
        if self.record is not None:
            self.record.parent.mkdir(parents=True, exist_ok=True)
            # Written at the dataset frame rate, so the video plays in world time even if the model runs slower.
            writer = cv2.VideoWriter(str(self.record), cv2.VideoWriter_fourcc(*"mp4v"), float(cfg.timing.fps), size)
            if not writer.isOpened():
                raise RuntimeError(f"cannot open {self.record} for writing")

        clock = pg.time.Clock()
        stamps: deque[float] = deque(maxlen=16)
        show_overlay, frames, running = True, 0, True
        try:
            while running and (self.max_frames is None or frames < self.max_frames):
                for event in get_events():
                    cmd = mapper.handle_event(event)
                    if cmd == CMD_QUIT:
                        running = False
                    elif cmd == CMD_RESET:
                        world.restart()
                    elif cmd == CMD_TOGGLE_OVERLAY:
                        show_overlay = not show_overlay
                action = mapper.frame_action()
                frame = tensor_to_rgb_uint8(world.step(action))
                surf = pg.transform.scale(pg.surfarray.make_surface(frame.swapaxes(0, 1)), size)
                screen.blit(surf, (0, 0))
                if show_overlay:
                    draw_overlay(screen, action, spec, self.fps)
                pg.display.flip()
                if writer is not None:
                    rgb = pg.surfarray.array3d(screen).swapaxes(0, 1)
                    writer.write(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
                frames += 1
                # Cap at the dataset frame rate: one mouse-count accumulation window must be 1/16 s, as in the labels.
                clock.tick(float(cfg.timing.fps))
                stamps.append(time.perf_counter())
                if len(stamps) > 1:
                    self.fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        finally:
            if writer is not None:
                writer.release()
            pg.event.set_grab(False)
            pg.mouse.set_visible(True)
            pg.quit()
        print(f"Played {frames} frames at {self.fps:.1f} FPS" + (f"; recorded to {self.record}" if writer else ""))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(
    cfg: DictConfig,
    ckpt: Path,
    upsampler_ckpt: Path | None,
    n_frames: int,
    out: Path,
    agent: Agent | None = None,
) -> dict:
    """Measure FPS for denoiser-only vs with upsampler, 1 vs 3 denoising steps, fp32 vs bf16 autocast.

    Runs headless from a blank context with idle actions (speed does not depend on the input). bf16 needs CUDA and is
    reported as skipped on CPU; configs with the upsampler are skipped without an upsampler checkpoint.

    Args:
        cfg: A validated cs2wm config.
        ckpt: Agent checkpoint.
        upsampler_ckpt: v1 checkpoint with the upsampler, or None.
        n_frames: Timed frames per configuration (after ``WARMUP_STEPS`` untimed ones).
        out: Output JSON (``results/play_fps.json``).
        agent: A ready agent, used instead of loading the checkpoints (tests).

    Returns:
        The report that was written: device info plus ``results[name]`` with ``status``, ``fps`` and ``ms_per_frame``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if agent is None:
        agent = load_agent(ckpt, upsampler_ckpt, device)
    has_upsampler = agent.upsampler is not None
    spec = ActionSpec.from_config(cfg)
    idle = blank_context(1, cfg.resolution.low_res, spec)[1][0]

    results: dict[str, dict[str, Any]] = {}
    for use_up in (False, True):
        for steps in (1, 3):
            for precision in ("fp32", "bf16"):
                name = f"{'upsampler' if use_up else 'denoiser'}_{steps}step_{precision}"
                if use_up and not has_upsampler:
                    results[name] = {"status": "skipped", "reason": "no upsampler checkpoint"}
                elif precision == "bf16" and device.type != "cuda":
                    results[name] = {"status": "skipped", "reason": "bf16 autocast needs CUDA"}
                else:
                    world = WorldModel(agent, steps, use_upsampler=use_up, bf16=precision == "bf16")
                    world.reset(*blank_context(world.n_cond, cfg.resolution.low_res, spec))
                    for _ in range(WARMUP_STEPS):
                        world.step(idle)
                    _sync(device)
                    t0 = time.perf_counter()
                    for _ in range(n_frames):
                        world.step(idle)
                    _sync(device)
                    dt = time.perf_counter() - t0
                    results[name] = {"status": "ok", "fps": n_frames / dt, "ms_per_frame": 1000 * dt / n_frames}
                r = results[name]
                print(
                    f"{name:28s} "
                    + (
                        f"{r['fps']:7.2f} FPS  {r['ms_per_frame']:8.1f} ms/frame"
                        if r["status"] == "ok"
                        else f"skipped ({r['reason']})"
                    )
                )

    report = {
        "device": device.type,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "n_frames": n_frames,
        "warmup_steps": WARMUP_STEPS,
        "ckpt": str(ckpt),
        "upsampler_ckpt": str(upsampler_ckpt) if upsampler_ckpt else None,
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out}")
    return report


def run(
    cfg: DictConfig,
    ckpt: Path,
    upsampler_ckpt: Path | None = None,
    no_upsampler: bool = False,
    denoising_steps: int = 1,
    context: Path | None = None,
    record: Path | None = None,
    benchmark_frames: int | None = None,
    bf16: bool = False,
    compile: bool = False,
) -> None:
    """CLI entry: play interactively, or benchmark when ``benchmark_frames`` is set.

    Args:
        cfg: A validated cs2wm config.
        ckpt: v2 agent checkpoint.
        upsampler_ckpt: v1 checkpoint providing the upsampler.
        no_upsampler: Display bicubic instead of upsampling.
        denoising_steps: 1 or 3.
        context: Held-out HDF5 episode for the initial context.
        record: Output MP4.
        benchmark_frames: If set, run ``benchmark`` with this many frames instead of playing.
        bf16: Sample under bf16 autocast (CUDA only).
        compile: ``torch.compile`` the models.

    Raises:
        SystemExit: If neither ``upsampler_ckpt`` nor ``no_upsampler`` is given for interactive play.
    """
    if benchmark_frames is not None:
        benchmark(cfg, ckpt, upsampler_ckpt, benchmark_frames, Path("results") / "play_fps.json")
        return
    if upsampler_ckpt is None and not no_upsampler:
        raise SystemExit("pass --upsampler-ckpt <v1 checkpoint>, or --no-upsampler to display a bicubic upscale")
    PlaySession(
        cfg, ckpt, upsampler_ckpt, denoising_steps, context, record, no_upsampler, bf16=bf16, compile=compile
    ).run()
