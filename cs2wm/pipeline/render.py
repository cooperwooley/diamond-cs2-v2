"""render: HLAE ``mirv_streams`` renderer at a fixed 16 fps, captured before the UI, Windows only (#15).

Per demo: write ``render_<demo_id>.cfg`` from the render plan, launch CS2 through the HLAE custom loader with
``-insecure`` (asserted) plus ``+playdemo`` and ``+exec``, wait up to ``render.timeout_factor`` x demo length, verify
each segment's frame count, and downscale the captures to 280x150 BGR (``cv2.INTER_AREA``, like v1) into
``frames/<demo_id>/<segment>/``. One retry, then the demo is marked ``failed``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from omegaconf import DictConfig


def build_render_cfg(plan: Mapping, cfg: DictConfig) -> str:
    """Build the text of ``render_<demo_id>.cfg``.

    Contents: ``config/cs2/render_base.cfg``, the scheduled ``spec_mode``/``spec_player`` switches, record
    start/end per segment (``mirv_cmd addAtTick``) and a final ``quit``.

    Args:
        plan: Parsed ``render_plan.json``.
        cfg: A validated cs2wm config (``render.*``).

    Returns:
        The cfg file content.
    """
    # TODO(#15): apply render.tick_offset; spec_mode value and spec_player form come from #14.
    raise NotImplementedError("TODO(#15): build_render_cfg")


def launch_hlae(cfg: DictConfig, demo_path: Path, render_cfg: Path, timeout_s: float) -> int:
    """Launch CS2 through the HLAE custom loader and wait for it to exit.

    Args:
        cfg: A validated cs2wm config.
        demo_path: Demo to play back.
        render_cfg: Generated ``render_<demo_id>.cfg``.
        timeout_s: Kill CS2 after this many seconds.

    Returns:
        The process exit code.

    Raises:
        ValueError: If ``-insecure`` is missing from the command line.
        TimeoutError: If the timeout elapsed.
    """
    # TODO(#15): HLAE.exe -customLoader -noGui -autoStart -hookDllPath ... -cmdLine "-steam -insecure ...".
    raise NotImplementedError("TODO(#15): launch_hlae")


def verify_frame_counts(plan: Mapping, frames_dir: Path) -> dict[str, tuple[int, int]]:
    """Compare captured frame counts with the plan.

    Args:
        plan: Parsed ``render_plan.json``.
        frames_dir: Directory with the raw captures.

    Returns:
        Mapping from segment to ``(expected, actual)`` for every mismatching segment (empty when all match).
    """
    raise NotImplementedError("TODO(#15): verify_frame_counts")


def downscale_frames(src_dir: Path, dst_dir: Path, size: tuple[int, int] = (150, 280)) -> int:
    """Downscale a capture directory to ``size`` BGR with ``cv2.INTER_AREA`` and delete the raw files.

    Args:
        src_dir: Directory with the raw TGA/PNG sequence.
        dst_dir: Output directory for ``uint8[H, W, 3]`` frames.
        size: ``(H, W)``.

    Returns:
        Number of frames written.
    """
    raise NotImplementedError("TODO(#15): downscale_frames")


def run(demo_path: Path, cfg: DictConfig) -> Path:
    """CLI stage: render one demo.

    Args:
        demo_path: Path to the ``.dem`` file.
        cfg: A validated cs2wm config.

    Returns:
        ``frames/<demo_id>/``.

    Raises:
        RuntimeError: After the retry also fails (crash, timeout or frame-count mismatch).
    """
    # TODO(#15): build cfg, launch_hlae, verify_frame_counts, downscale_frames; one retry.
    raise NotImplementedError("TODO(#15): render.run")
