"""render: HLAE ``mirv_streams`` renderer at a fixed 16 fps, captured before the UI, Windows only (#15).

Per demo: write ``render_<demo_id>.cfg`` from the render plan, launch CS2 through the HLAE custom loader with
``-insecure`` (asserted) plus ``+playdemo`` and ``+exec``, wait up to ``render.timeout_factor`` x demo length, verify
each segment's frame count, and downscale the captures to 280x150 BGR (``cv2.INTER_AREA``, like v1) into
``frames/<demo_id>/seg<idx:03d>/<frame:06d>.png``. One retry, then the demo is reported as failed with a reason.

How it works:

* The cfg is installed as ``<cs2_dir>/game/csgo/cfg/render_<demo_id>.cfg`` and run with ``+exec`` on the command line,
  next to ``+playdemo <demo>``. ``mirv_cmd addAtTick`` ticks are demo ticks, so the schedule only needs to be
  registered before playback reaches them; it does not matter that the exec runs while the demo is still loading.
  CONFIRM(#14): HLAE keeps the ``mirv_cmd`` schedule across the demo load. If #14 shows it is dropped, switch to an
  ``exec`` over ``-netconport`` (``generate.NetconClient``) once the demo is loaded and paused.
* HLAE writes one take per ``record start``: ``<raw>/<demo_id>/take0000/<stream>/00000.tga`` (TGA) or a video file
  (FFmpeg output). Takes are mapped to segments in schedule order. CONFIRM(#14): the layout and file names.
* Every segment is downscaled into ``segNNN.tmp/`` and renamed to ``segNNN/`` only when complete, so a crash never
  leaves a half-written segment that looks finished. ``frames/<demo_id>/render.json`` is written last.
* Frames are stored as PNG, not ``.npy``: a 150x280x3 frame is 126 kB raw but ~50-70 kB as PNG, so 1000-frame
  episodes of a long run fit the <= 10 GB working set (the raw 1280x720 captures are deleted per segment), and PNG
  is directly viewable by the label-check tool (#18). Compression level 1 is nearly as small as the default 3 at a
  fraction of the CPU time, which matters at >= 3x realtime.

Everything marked ``CONFIRM(#14)`` is an assumption about HLAE/CS2 that the manual debug session must confirm.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from omegaconf import DictConfig

from cs2wm import config as cs2wm_config
from cs2wm.pipeline import proc as proc_utils
from cs2wm.pipeline.generate import Hooks
from cs2wm.pipeline.segment import read_render_plan

log = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 1
_TAKE_RE = re.compile(r"^take\d+$")
_VIDEO_SUFFIXES = (".mp4", ".mkv", ".avi", ".mov")
_IMAGE_SUFFIXES = (".tga", ".png", ".bmp")
_WINDOWS_PATH_RE = re.compile(r"^([A-Za-z]:|\\\\)")


class RenderError(RuntimeError):
    """A render problem that retrying cannot fix (bad plan, no usable spectate target)."""


@dataclass(frozen=True)
class LaunchResult:
    """How one HLAE launch ended.

    Attributes:
        returncode: Exit code of HLAE.exe (``None`` if it never exited, i.e. a timeout). CONFIRM(#14): HLAE may
            exit with 0 while cs2.exe keeps running, and the code of ``quit`` is unknown, so it is informational.
        timed_out: The timeout elapsed and the process tree was killed.
    """

    returncode: int | None
    timed_out: bool


@dataclass
class RenderResult:
    """Outcome of ``run``, recorded by the supervisor (#20).

    Attributes:
        demo_id: Demo id.
        ok: All segments were rendered and downscaled.
        reason: Why the render failed (``None`` on success).
        frames: Frames written per segment index.
        wall_s: Wall time of the whole stage, retries included.
        realtime_factor: Rendered demo seconds per wall second (0 on failure).
        attempts: Launches used.
        out_dir: ``frames/<demo_id>/``.
    """

    demo_id: str
    ok: bool
    reason: str | None = None
    frames: dict[int, int] = field(default_factory=dict)
    wall_s: float = 0.0
    realtime_factor: float = 0.0
    attempts: int = 0
    out_dir: Path | None = None


# ---------------------------------------------------------------------------------------------------------------------
# Paths and console quoting


def _is_windows() -> bool:
    return os.name == "nt"


def win_path(base: str | Path, *parts: str) -> str:
    """Join path parts; a Windows-style base (``C:/x`` or ``\\\\host``) is joined and written with backslashes.

    Args:
        base: Base directory.
        *parts: Path components below ``base``.

    Returns:
        The joined path as a string.
    """
    base = str(base)
    if not _WINDOWS_PATH_RE.match(base):
        return str(Path(base, *parts))
    return "\\".join([base.replace("/", "\\").rstrip("\\"), *(p.replace("/", "\\").strip("\\") for p in parts)])


def console_quote(text: str) -> str:
    """Quote ``text`` as one CS2 console argument.

    Backslashes and spaces survive inside double quotes; a trailing backslash is dropped because it would sit
    next to the closing quote. CONFIRM(#14): backslashes in quoted console arguments (the #14 step 4 form).

    Args:
        text: A path or name.

    Returns:
        ``"text"``.

    Raises:
        RenderError: If ``text`` holds a quote, semicolon or newline, which cannot be quoted safely.
    """
    if any(c in text for c in '";\r\n'):
        raise RenderError(f"cannot quote {text!r} for the CS2 console (quote, semicolon or newline)")
    return f'"{text.rstrip(chr(92))}"'


def raw_dir(cfg: DictConfig, demo_id: str) -> Path:
    """Directory that HLAE records into for ``demo_id``: ``paths.raw/<demo_id>``."""
    return Path(cfg.paths.raw) / demo_id


def _check_safe_name(what: str, value: str) -> None:
    """Raise ``RenderError`` unless ``value`` is safe in file names and console commands."""
    if not cs2wm_config.SAFE_NAME_RE.match(value):
        raise RenderError(f"{what} {value!r} may only hold letters, digits, '_', '.' and '-'")


# ---------------------------------------------------------------------------------------------------------------------
# cfg


def _spec_player_arg(target: Mapping, field_name: str, segment_idx: int) -> str:
    """The ``spec_player`` argument for a target, per ``render.spec_player_field`` (CONFIRM(#14))."""
    if field_name == "slot":
        raise RenderError("render.spec_player_field=slot is not supported: the plan has no slot (decide in #14)")
    value = target.get(field_name)
    if value is None or value == "":
        raise RenderError(f"segment {segment_idx}: target {target.get('name')!r} has no {field_name}")
    text = str(value)
    if field_name == "name" and any(c in text for c in ' \t";'):
        raise RenderError(f"segment {segment_idx}: name {text!r} cannot be passed to spec_player unquoted")
    return text


def build_render_cfg(plan: Mapping, cfg: DictConfig) -> str:
    """Build the text of ``render_<demo_id>.cfg``.

    Contents: ``config/cs2/render_base.cfg``, ``mirv_streams record name <raw dir>``, ``mirv_cmd clear``, per
    ``mirv_cmd_schedule`` entry the ``spec_mode``/``spec_player`` switch or ``mirv_streams record start|end``
    (``mirv_cmd addAtTick``, ticks already include ``render.tick_offset``) and a final ``quit``.

    Args:
        plan: Parsed ``render_plan.json``.
        cfg: A validated cs2wm config (``render.*``).

    Returns:
        The cfg file content.

    Raises:
        RenderError: If the plan has nothing to render, the demo id or stream name is unusable, or a target lacks the field that
            ``render.spec_player_field`` selects.
    """
    demo_id = plan["demo_id"]
    _check_safe_name("demo id", demo_id)
    _check_safe_name("render.stream_name", str(cfg.render.stream_name))
    schedule = plan["mirv_cmd_schedule"]
    if not schedule:
        raise RenderError(f"{demo_id}: the plan has no segments to render")
    r = cfg.render
    settings = r.ffmpeg_preset if r.output_format == "ffmpeg" else "afxClassic"
    base = (Path(cs2wm_config.DEFAULT_CONFIG_PATH).parent / "cs2" / "render_base.cfg").read_text(encoding="utf-8")
    base = base.replace("{stream}", r.stream_name).replace("{fps}", str(cfg.timing.fps)).replace("{settings}", settings)

    lines = [base.rstrip("\n"), "", f"// ---- {demo_id}: generated by cs2wm.pipeline.render ----"]
    lines.append(f"mirv_streams record name {console_quote(win_path(cfg.paths.raw, demo_id))}")
    lines.append("mirv_cmd clear")
    for entry in schedule:
        tick, action = entry["mirv_tick"], entry["action"]
        if action == "spec":
            arg = _spec_player_arg(entry["target"], r.spec_player_field, entry["segment_idx"])
            lines.append(f'mirv_cmd addAtTick {tick} "spec_mode {r.spec_mode}"')
            lines.append(f'mirv_cmd addAtTick {tick} "spec_player {arg}"')
        elif action in ("record_start", "record_end"):
            lines.append(f'mirv_cmd addAtTick {tick} "mirv_streams record {action.split("_")[1]}"')
        else:
            raise RenderError(f"unknown schedule action {action!r}")
    last_end = max(e["mirv_tick"] for e in schedule if e["action"] == "record_end")
    demo_end = plan["demo_end_tick"] + plan["tick_offset"]
    quit_tick = min(last_end + int(r.quit_delay_ticks), max(demo_end, last_end + 1))
    lines.append(f'mirv_cmd addAtTick {quit_tick} "quit"')
    return "\n".join(lines) + "\n"


def install_cfg(text: str, demo_id: str, cs2_dir: str | Path) -> Path:
    """Write the cfg to ``<cs2_dir>/game/csgo/cfg/render_<demo_id>.cfg`` (where ``+exec`` finds it).

    Args:
        text: Output of ``build_render_cfg``.
        demo_id: Demo id.
        cs2_dir: CS2 install dir (``paths.cs2_dir``).

    Returns:
        The installed path.
    """
    dest = Path(cs2_dir) / "game" / "csgo" / "cfg" / f"render_{demo_id}.cfg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------------------------------------------------
# Launch


def build_hlae_command(cfg: DictConfig, demo_path: Path, cfg_name: str) -> list[str]:
    """Build the HLAE custom-loader argv (#14 step 1).

    ``HLAE.exe -customLoader -noGui -autoStart -hookDllPath <dll> -programPath <cs2.exe> -cmdLine "<cs2 args>"``

    Args:
        cfg: A validated cs2wm config.
        demo_path: Demo to play back (absolute path; CONFIRM(#14): ``playdemo`` accepts one outside ``game/csgo``).
        cfg_name: Installed cfg name without extension, e.g. ``render_dm-0001``.

    Returns:
        The argv list.

    Raises:
        ValueError: If ``-insecure`` is missing from the command line.
    """
    height, width = (int(v) for v in cfg.resolution.capture)
    cs2_args = [
        *(str(o) for o in cfg.render.launch_options),
        "-w", str(width), "-h", str(height),
        "+playdemo", console_quote(win_path(Path(demo_path).absolute())),
        "+exec", cfg_name,
    ]  # fmt: skip
    cmd = [
        win_path(cfg.paths.hlae_dir, cfg.render.hlae_exe),
        "-customLoader", "-noGui", "-autoStart",
        "-hookDllPath", win_path(cfg.paths.hlae_dir, *str(cfg.render.hook_dll).split("/")),
        "-programPath", win_path(cfg.paths.cs2_dir, *str(cfg.generate.cs2_exe).split("/")),
        "-cmdLine", " ".join(cs2_args),
    ]  # fmt: skip
    proc_utils.require_offline(cmd)
    return cmd


def cs2_running(run: Callable[..., Any], exe_name: str) -> bool:
    """Whether a process named ``exe_name`` is running (``tasklist``; always False off Windows).

    Args:
        run: ``subprocess.run``-like.
        exe_name: Image name, e.g. ``cs2.exe``.

    Returns:
        True if ``tasklist`` lists the image.
    """
    if not _is_windows():
        return False
    out = run(
        ["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/NH"], capture_output=True, text=True, check=False, timeout=30
    )
    return exe_name.lower() in (out.stdout or "").lower()


def launch_hlae(
    cfg: DictConfig,
    demo_path: Path,
    cfg_name: str,
    timeout_s: float,
    hooks: Hooks | None = None,
    is_cs2_running: Callable[[], bool] | None = None,
) -> LaunchResult:
    """Launch CS2 through the HLAE custom loader and wait for HLAE and cs2.exe to exit.

    CONFIRM(#14): HLAE.exe may exit while cs2.exe keeps running (or the other way round), so the wait ends when
    HLAE has exited AND no cs2.exe image is left (polled with ``tasklist``). On timeout the HLAE process tree is
    killed with ``taskkill /T /F`` and so is every ``cs2.exe`` image (the game may not be a child of HLAE).

    HLAE can exit before ``cs2.exe`` shows up in ``tasklist``, so after HLAE exits the wait only ends once cs2.exe
    has been seen and is gone again, or when it was never seen within a grace window of 5 x ``render.poll_s``
    (capped at ``render.startup_s``).

    NOTE: the ``tasklist`` probe and ``taskkill /IM cs2.exe`` assume this stage owns the only CS2 on the machine
    (the supervisor serializes the stages); another CS2 would be waited for, and killed on a timeout.

    Args:
        cfg: A validated cs2wm config.
        demo_path: Demo to play back.
        cfg_name: Installed cfg name without extension.
        timeout_s: Kill everything after this many seconds.
        hooks: Clock, sleep, popen and run (default: real ones).
        is_cs2_running: Probe for a running cs2.exe (default: ``tasklist``).

    Returns:
        How the launch ended.

    Raises:
        ValueError: If ``-insecure`` is missing from the command line.
    """
    hooks = hooks or Hooks()
    exe = Path(str(cfg.generate.cs2_exe)).name
    probe = is_cs2_running or (lambda: cs2_running(hooks.run, exe))
    cmd = build_hlae_command(cfg, demo_path, cfg_name)
    proc_utils.require_offline(cmd)
    windows = _is_windows()
    log.info("launching HLAE: %s", " ".join(cmd))
    proc = hooks.popen(cmd, **proc_utils.spawn_kwargs(windows))
    start = hooks.clock()
    returncode: int | None = None
    exited_at = 0.0
    seen = False
    grace = min(5 * float(cfg.render.poll_s), float(cfg.render.startup_s))
    try:
        while True:
            rc = proc.poll()
            now = hooks.clock()
            if rc is not None:
                if returncode is None:
                    returncode, exited_at = rc, now
                if probe():
                    seen = True
                elif seen or now - exited_at >= grace:
                    return LaunchResult(returncode, False)
            if now - start >= timeout_s:
                log.warning("render timeout after %.0f s; killing HLAE and CS2", timeout_s)
                break
            hooks.sleep(float(cfg.render.poll_s))
    finally:
        # Reached on timeout and on any error; a no-op once HLAE exited. Then clear a cs2.exe that outlived it.
        proc_utils.kill_tree(proc, hooks.run, windows)
    if windows:
        hooks.run(["taskkill", "/T", "/F", "/IM", exe], capture_output=True, check=False, timeout=30)
    return LaunchResult(returncode, True)


# ---------------------------------------------------------------------------------------------------------------------
# Captures


def list_takes(raw: Path) -> list[Path]:
    """The ``takeNNNN`` directories under ``raw`` ordered by take number (= segment order)."""
    if not raw.is_dir():
        return []
    takes = [p for p in raw.iterdir() if p.is_dir() and _TAKE_RE.match(p.name)]
    return sorted(takes, key=lambda p: int(p.name[len("take") :]))  # numeric, so take10000 follows take9999


def _image_files(directory: Path) -> list[Path]:
    """Lossless frame images in ``directory`` ordered by the integer in the name."""
    files = [p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES]

    def key(p: Path) -> tuple[int, str]:
        digits = re.findall(r"\d+", p.stem)
        return (int(digits[-1]) if digits else -1, p.name)

    return sorted(files, key=key)


def capture_source(take: Path, cfg: DictConfig) -> Path | None:
    """Where a take holds its frames: the stream directory (TGA) or the video file (FFmpeg).

    CONFIRM(#14): ``<take>/<stream>/00000.tga`` for TGA; for FFmpeg a video file in ``<take>/<stream>/`` or named
    ``<take>/<stream>.<ext>``. Lossless codecs are expected (``render.ffmpeg_preset``).

    Args:
        take: A ``takeNNNN`` directory.
        cfg: A validated cs2wm config.

    Returns:
        The source path, or None if the take holds none.
    """
    stream = take / str(cfg.render.stream_name)
    if cfg.render.output_format == "tga":
        return stream if stream.is_dir() else None
    candidates = sorted(p for p in stream.iterdir() if p.suffix.lower() in _VIDEO_SUFFIXES) if stream.is_dir() else []
    candidates += sorted(p for p in take.glob(f"{stream.name}.*") if p.suffix.lower() in _VIDEO_SUFFIXES)
    return candidates[0] if candidates else None


def _read_tga(path: Path) -> np.ndarray:
    """Decode an uncompressed 24/32-bit TGA to ``uint8[H, W, 3]`` BGR (OpenCV has no TGA codec)."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size < 18:
        raise ValueError(f"{path}: truncated TGA header")
    id_len, cmap_type, img_type = int(data[0]), int(data[1]), int(data[2])
    width = int(data[12]) | int(data[13]) << 8
    height = int(data[14]) | int(data[15]) << 8
    bpp, desc = int(data[16]), int(data[17])
    if img_type != 2 or cmap_type != 0 or bpp not in (24, 32):
        raise ValueError(f"{path}: unsupported TGA (type {img_type}, {bpp} bpp); expected uncompressed 24/32-bit")
    channels = bpp // 8
    start = 18 + id_len
    body = data[start : start + width * height * channels]
    if body.size != width * height * channels:
        raise ValueError(f"{path}: truncated TGA pixel data")
    img = body.reshape(height, width, channels)
    if not desc & 0x20:  # bottom-left origin: rows are stored bottom-up
        img = img[::-1]
    return np.ascontiguousarray(img[:, :, :3])


def _read_image(path: Path) -> np.ndarray:
    """Read one capture frame as BGR ``uint8``."""
    if path.suffix.lower() == ".tga":
        return _read_tga(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"{path}: unreadable image")
    return img


def _iter_frames(source: Path) -> Iterator[np.ndarray]:
    """Yield the BGR frames of an image directory or a video file, in order."""
    if source.is_dir():
        for path in _image_files(source):
            yield _read_image(path)
        return
    cap = cv2.VideoCapture(str(source))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def count_frames(source: Path) -> int:
    """Number of frames in an image directory, or in a video (decoded, since container headers can lie)."""
    if source.is_dir():
        return len(_image_files(source))
    return sum(1 for _ in _iter_frames(source))


def _segments(plan: Mapping) -> list[Mapping]:
    return sorted(plan["segments"], key=lambda s: s["segment_idx"])


def verify_frame_counts(plan: Mapping, raw: Path, cfg: DictConfig) -> dict[int, tuple[int, int]]:
    """Compare captured frame counts with the plan.

    The i-th take belongs to the i-th segment in schedule order. The expected count is the segment's
    ``rendered_frames`` (``labeled_frames + 1``). CONFIRM(#14): the frame at ``record_end_tick`` is captured.

    Args:
        plan: Parsed ``render_plan.json``.
        raw: Raw capture directory of the demo (``paths.raw/<demo_id>``).
        cfg: A validated cs2wm config (output format, stream name).

    Returns:
        ``{segment_idx: (expected, actual)}`` for every segment; a missing take counts as 0 frames. Mismatches are
        the entries where the two numbers differ.
    """
    takes = list_takes(raw)
    counts: dict[int, tuple[int, int]] = {}
    for i, seg in enumerate(_segments(plan)):
        source = capture_source(takes[i], cfg) if i < len(takes) else None
        counts[seg["segment_idx"]] = (int(seg["rendered_frames"]), count_frames(source) if source else 0)
    for idx, (expected, actual) in counts.items():
        if expected != actual:
            log.warning("segment %d: expected %d frames, captured %d", idx, expected, actual)
    return counts


def downscale_frames(
    src: Path, dst_dir: Path, size: tuple[int, int] = (150, 280), png_compression: int = 1, delete_raw: bool = True
) -> int:
    """Downscale a capture to ``size`` BGR with ``cv2.INTER_AREA`` into ``dst_dir`` and delete the raw files.

    The frames are written into ``<dst_dir>.tmp`` and the directory is renamed to ``dst_dir`` when complete, so an
    interrupted run never leaves a partial ``dst_dir``. The raw capture is deleted only after that rename.

    Args:
        src: Image directory (TGA/PNG/BMP sequence) or video file.
        dst_dir: Output directory; receives ``<frame:06d>.png`` (``uint8[H, W, 3]`` BGR).
        size: ``(H, W)``.
        png_compression: ``cv2.IMWRITE_PNG_COMPRESSION`` level.
        delete_raw: Delete ``src`` after a successful downscale.

    Returns:
        Number of frames written.

    Raises:
        ValueError: If a frame cannot be decoded or encoded, or ``src`` holds no frames.
    """
    tmp = dst_dir.with_name(dst_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    height, width = size
    n = 0
    try:
        for n, frame in enumerate(_iter_frames(src), start=1):
            small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(tmp / f"{n - 1:06d}.png"), small, [cv2.IMWRITE_PNG_COMPRESSION, png_compression]):
                raise ValueError(f"could not write frame {n - 1} to {tmp}")
        if n == 0:
            raise ValueError(f"{src} holds no frames")
        shutil.rmtree(dst_dir, ignore_errors=True)
        os.replace(tmp, dst_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if delete_raw:
        if src.is_dir():
            shutil.rmtree(src, ignore_errors=True)
        else:
            src.unlink(missing_ok=True)
    return n


# ---------------------------------------------------------------------------------------------------------------------
# Runner


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _manifest(plan: Mapping, plan_sha: str, cfg: DictConfig, frames: dict[int, int], result: RenderResult) -> dict:
    """The ``render.json`` content that ``pack`` reads."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "demo_id": plan["demo_id"],
        "plan_sha256": plan_sha,
        "fps": int(cfg.timing.fps),
        "ticks_per_frame": plan["timing"]["ticks_per_frame"],
        "size": [int(v) for v in cfg.resolution.render],
        "image_format": "png",
        "wall_s": round(result.wall_s, 3),
        "realtime_factor": round(result.realtime_factor, 3),
        "segments": [
            {
                "segment_idx": seg["segment_idx"],
                "dir": f"seg{seg['segment_idx']:03d}",
                "frames": frames[seg["segment_idx"]],
                "start_tick": seg["record_start_tick"],
                "end_tick": seg["record_end_tick"],
                "target": seg["target"],
                "episodes": [
                    {"episode_id": e["episode_id"], "first_frame": e["first_frame"], "n_frames": e["n_frames"]}
                    for e in seg["episodes"]
                ],
            }
            for seg in _segments(plan)
        ],
    }


def _attempt(
    plan: Mapping,
    cfg: DictConfig,
    demo_path: Path,
    frames_dir: Path,
    timeout_s: float,
    hooks: Hooks,
    launch: Callable[..., LaunchResult],
) -> tuple[str | None, dict[int, int]]:
    """One launch plus verification and downscaling. Returns ``(failure reason or None, frames per segment)``."""
    demo_id = plan["demo_id"]
    raw = raw_dir(cfg, demo_id)
    shutil.rmtree(raw, ignore_errors=True)  # stale takes would shift the take -> segment mapping
    shutil.rmtree(frames_dir, ignore_errors=True)
    raw.mkdir(parents=True)
    try:
        launched = launch(cfg, demo_path, f"render_{demo_id}", timeout_s, hooks)
    except OSError as exc:
        return f"could not start HLAE: {exc!r}", {}
    if launched.timed_out:
        return f"timeout after {timeout_s:.0f} s (killed HLAE and CS2)", {}

    counts = verify_frame_counts(plan, raw, cfg)
    n_takes = len(list_takes(raw))
    problems = [f"segment {i}: expected {e}, got {a} frames" for i, (e, a) in counts.items() if e != a]
    if n_takes != len(counts):
        problems.append(f"{n_takes} takes captured for {len(counts)} segments")
    if counts and all(a == e - 1 for e, a in counts.values()) and n_takes == len(counts):
        return "every segment is exactly one frame short (end tick not captured? see CONFIRM(#14))", {}
    if problems:
        code = "" if launched.returncode in (0, None) else f" (HLAE exit code {launched.returncode})"
        return "frame-count mismatch" + code + ": " + "; ".join(problems), {}

    frames: dict[int, int] = {}
    try:
        frames_dir.mkdir(parents=True, exist_ok=True)
        size = (int(cfg.resolution.render[0]), int(cfg.resolution.render[1]))
        for take, seg in zip(list_takes(raw), _segments(plan), strict=True):
            source = capture_source(take, cfg)
            assert source is not None  # verify_frame_counts counted frames in it
            idx = seg["segment_idx"]
            frames[idx] = downscale_frames(source, frames_dir / f"seg{idx:03d}", size, int(cfg.render.png_compression))
    except (OSError, ValueError, cv2.error) as exc:
        return f"downscale failed: {exc!r}", {}
    return None, frames


def run(
    demo_path: Path,
    cfg: DictConfig,
    hooks: Hooks | None = None,
    launch: Callable[..., LaunchResult] = launch_hlae,
) -> RenderResult:
    """CLI stage: render one demo.

    A crash, timeout or frame-count mismatch is retried ``render.retries`` (1) times; after that the result is a
    failure with a reason. Problems retrying cannot fix (no plan, a target without the ``spec_player`` field) fail
    at once. ``-insecure`` problems raise ``ValueError``.

    Args:
        demo_path: Path to the ``.dem`` file.
        cfg: A validated cs2wm config.
        hooks: Clock, sleep, popen and run (default: real ones).
        launch: HLAE launcher (``launch_hlae``-like; replaced by a fake in tests).

    Returns:
        The result; ``out_dir`` is ``frames/<demo_id>/``, which holds ``render.json`` on success.
    """
    hooks = hooks or Hooks()
    demo_path = Path(demo_path)
    demo_id = demo_path.stem
    out_dir = Path(cfg.paths.frames) / demo_id
    result = RenderResult(demo_id=demo_id, ok=False, out_dir=out_dir)
    t0 = hooks.clock()
    shutil.rmtree(out_dir.with_name(out_dir.name + ".deleting"), ignore_errors=True)  # a pack that died mid-cleanup

    plan_path = Path(cfg.paths.plans) / demo_id / "render_plan.json"
    try:
        plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        plan = read_render_plan(plan_path)
        if plan["demo_id"] != demo_id:
            raise RenderError(f"{plan_path} is for demo {plan['demo_id']!r}")
        cfg_text = build_render_cfg(plan, cfg)
    except (OSError, ValueError, KeyError, RenderError) as exc:
        result.reason = f"bad render plan: {exc}"
        log.error("%s: %s", demo_id, result.reason)
        return result
    try:
        install_cfg(cfg_text, demo_id, cfg.paths.cs2_dir)
    except OSError as exc:
        result.reason = f"could not install the render cfg: {exc}"
        log.error("%s: %s", demo_id, result.reason)
        return result

    length_s = plan["demo_end_tick"] / cfg.timing.tickrate
    timeout_s = float(cfg.render.timeout_factor) * length_s + float(cfg.render.startup_s)
    frames: dict[int, int] = {}
    reason: str | None = None
    for attempt in range(1, int(cfg.render.retries) + 2):
        result.attempts = attempt
        reason, frames = _attempt(plan, cfg, demo_path, out_dir, timeout_s, hooks, launch)
        if reason is None:
            break
        log.warning("%s: render attempt %d failed: %s", demo_id, attempt, reason)
    result.wall_s = hooks.clock() - t0
    shutil.rmtree(raw_dir(cfg, demo_id), ignore_errors=True)  # whatever raw captures are left (failure, empty take)
    if reason is not None:
        shutil.rmtree(out_dir, ignore_errors=True)
        result.reason = reason
        return result

    result.ok, result.frames = True, frames
    rendered_s = sum(frames.values()) / cfg.timing.fps
    result.realtime_factor = rendered_s / result.wall_s if result.wall_s > 0 else 0.0
    _write_json_atomic(out_dir / "render.json", _manifest(plan, plan_sha, cfg, frames, result))
    log.info("%s: %d frames in %.0f s (%.1fx realtime)", demo_id, sum(frames.values()), result.wall_s,
             result.realtime_factor)  # fmt: skip
    return result


@dataclass(frozen=True)
class CfgOnly:
    """What ``cs2wm render --cfg-only`` wrote and would run.

    Attributes:
        installed: The cfg in ``<cs2_dir>/game/csgo/cfg``.
        copy: A copy next to the plan (``render_<demo_id>.cfg``).
        argv: The HLAE argv ``run`` would launch.
        powershell: The same command for pasting into PowerShell.
    """

    installed: Path
    copy: Path
    argv: list[str]
    powershell: str


def powershell_command(argv: list[str]) -> str:
    """Format ``argv`` as one PowerShell line.

    ``--%`` (stop-parsing) hands the rest of the line to HLAE.exe verbatim, so the quoting is the Windows
    command-line quoting that ``subprocess`` uses for the real launch (the inner ``\"`` of the ``-cmdLine`` value).

    Args:
        argv: Output of ``build_hlae_command``.

    Returns:
        The line, starting with ``&``.
    """
    exe = argv[0].replace("'", "''")
    return f"& '{exe}' --% {subprocess.list2cmdline(argv[1:])}"


def write_cfg_only(plan_path: Path, cfg: DictConfig, demo_path: Path | None = None) -> CfgOnly:
    """Write the render cfg for a plan and build the HLAE command without launching (for the manual #14 session).

    No demo file is needed: without ``demo_path`` the command points at ``paths.demos/<kind>/<plan demo_file>``.

    Args:
        plan_path: A ``render_plan.json``.
        cfg: A validated cs2wm config.
        demo_path: Demo to play back, if known.

    Returns:
        The written files and the command.

    Raises:
        RenderError: If the plan cannot be turned into a cfg.
        ValueError: If the plan schema is unsupported, or the command would not be offline-only.
    """
    plan = read_render_plan(plan_path)
    text = build_render_cfg(plan, cfg)
    demo_id = plan["demo_id"]
    if demo_path is None:
        demo_path = Path(cfg.paths.demos) / str((plan.get("meta") or {}).get("kind", "dm")) / plan["demo_file"]
    installed = install_cfg(text, demo_id, cfg.paths.cs2_dir)
    copy = plan_path.with_name(f"render_{demo_id}.cfg")
    copy.write_text(text, encoding="utf-8")
    argv = build_hlae_command(cfg, demo_path, f"render_{demo_id}")
    return CfgOnly(installed, copy, argv, powershell_command(argv))
