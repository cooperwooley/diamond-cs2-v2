"""Interactive play loop and FPS benchmark (#28, #35).

Opens a pygame window with relative mouse mode, feeds ``InputMapper`` actions to the denoiser (and optionally the v1
upsampler), draws the input overlay and can record to MP4 (``--record``, #35).
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig


class PlaySession:
    """One interactive session.

    Args:
        cfg: A validated cs2wm config.
        ckpt: v2 agent checkpoint.
        upsampler_ckpt: v1 checkpoint providing the upsampler, or None to display bicubic.
        denoising_steps: 1 or 3.
        context: Held-out HDF5 episode for the 4-frame initial context (``Backspace`` resets).
        record: Output MP4 with the overlay drawn in.
    """

    def __init__(
        self,
        cfg: DictConfig,
        ckpt: Path,
        upsampler_ckpt: Path | None = None,
        denoising_steps: int = 1,
        context: Path | None = None,
        record: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.ckpt = ckpt
        self.upsampler_ckpt = upsampler_ckpt
        self.denoising_steps = denoising_steps
        self.context = context
        self.record = record

    def run(self) -> None:
        """Run until the window is closed (``F1`` toggles the overlay)."""
        # TODO(#28): cs2wm.play.agent.build_play_agent (#5), pygame loop, overlay; TODO(#35): record with overlay.
        raise NotImplementedError("TODO(#28): PlaySession.run")


def benchmark(cfg: DictConfig, ckpt: Path, upsampler_ckpt: Path | None, n_frames: int, out: Path) -> dict:
    """Measure FPS for denoiser-only vs with upsampler, 1 vs 3 denoising steps, fp32 vs bf16 autocast.

    Args:
        cfg: A validated cs2wm config.
        ckpt: Agent checkpoint.
        upsampler_ckpt: v1 checkpoint with the upsampler, or None.
        n_frames: Frames per configuration.
        out: Output JSON (``results/play_fps.json``).

    Returns:
        FPS by configuration.
    """
    raise NotImplementedError("TODO(#28): benchmark")


def run(
    cfg: DictConfig,
    ckpt: Path,
    upsampler_ckpt: Path | None = None,
    no_upsampler: bool = False,
    denoising_steps: int = 1,
    context: Path | None = None,
    record: Path | None = None,
    benchmark_frames: int | None = None,
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
    """
    # TODO(#28): dispatch to benchmark() or PlaySession.run(); TODO(#35): recorded sessions.
    raise NotImplementedError("TODO(#28): play.run")
