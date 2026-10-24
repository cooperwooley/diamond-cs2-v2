"""H2 volumetric-smoke eval (#33): LPIPS (AlexNet) and SSIM inside smoke masks vs on static geometry.

Claim under test: CS2's volumetric smokes are harder to model than static geometry. Computed on held-out smoke episodes
for v2 and v1 (and any other model given), for teacher-forced single-step predictions and 16-step autoregressive
rollouts, with bootstrap CIs.

Method
------
1. **Data.** ``--data`` is a directory of *full-resolution* (150x280) packed episodes (the ``hdf5_fullres`` subset): the
   masks and AlexNet LPIPS need that resolution, and the denoiser's 30x56 context is derived from the ground truth with
   ``pack.to_low_res``. Only episodes with ``kind == "smoke"`` of test demos (``--split-json``) are used. Without
   ``--split-json`` the data are not known to be held out: ``data.held_out`` is false and a warning is printed.
2. **Episode to demo.** An episode's ``demo_id`` attr names ``<--demos>/<demo_id>.dem``; frame ``k`` is at tick
   ``start_tick + k * ticks_per_frame``. The spectated player is the episode's ``user_id`` attr
   if it has one, else the player whose ``name`` equals its ``player`` attr. Each demo is parsed once: the tick table
   (``cs2wm.demo.load_ticks``: position, view angles, ``duck_amount``, health for every player) and the smoke events
   (``cs2wm.demo.smoke_events``, or ``<paths.events>/<demo_id>.parquet`` when that exists). A tick missing from the table
   is filled by the nearest one within ``qa.max_fill_ticks``, else the frame has no pose and is skipped.
3. **Masks** (``cs2wm.eval.smoke_masks``, which documents the camera conventions): the smoke mask is the union of the
   projected, dilated spheres of the smokes active at the frame's tick; the static mask is its complement minus the
   viewmodel box, on frames with no living enemy in view. Wall occlusion is ignored (a documented limitation).
   ``--n-validate`` overlay PNGs (GT, smoke tinted, static outlined) go to ``mask_validation/``.
4. **Contexts.** ``eval.h2.contexts`` windows of 4 ground-truth frames and ``--rollout-steps`` (16) target frames,
   drawn deterministically (``--seed``) from windows in which at least ``eval.h2.min_smoke_frames`` targets have a
   smoke of at least ``eval.h2.min_region_px`` pixels; windows of one episode do not overlap.
5. **Predictions**, for every model on the same contexts and the same noise (``h1.job_generator``): the *rollout*
   (``WorldModel.rollout``, 16 steps driven by the ground-truth actions) and the *single-step* predictions (one step from
   the ground-truth context ending just before each target frame). Step 1 of the single-step condition is step 1 of the
   rollout (same noise). The upsampler comes from ``--upsampler-ckpt`` (default: the ``v1`` model's checkpoint) for all
   models but v1, which uses its own, as in H1. Generated frames are quantised to uint8 before scoring.
6. **Metrics.** Per frame and region, the mean of the spatial LPIPS map (``lpips.LPIPS(net="alex", spatial=True)``) and of
   the SSIM map (11x11 Gaussian window, valid region only, so the 5 px border has no SSIM value) over the region's
   pixels. A frame counts for a region when the region has at least ``eval.h2.min_region_px`` pixels. Per (model,
   condition, region, metric): the per-context mean of the frame values, then the mean over contexts with a
   percentile-bootstrap 95% CI (``eval.h2.bootstrap`` resamples over *contexts*, seeded). The paired difference
   smoke - static uses the contexts that have both regions.

``summary.json`` (``schema_version`` 1; NaN is written as null)
---------------------------------------------------------------
``config``: ``hfov_deg``, ``smoke_radius_units``, ``mask_dilate_px``, ``max_los_units``, ``viewmodel_box``,
    ``enemy_max_dist_units``, ``enemy_margin_px``, ``min_region_px``, ``min_smoke_frames``, ``rollout_steps``,
    ``contexts`` (requested), ``bootstrap``, ``denoising_steps``, ``batch_size``, ``device``, ``frame_hw``.
``seed``; ``data``: ``dir``, ``split_json``, ``held_out``, ``demos_dir``, ``n_episodes`` (smoke episodes used),
    ``skipped`` (list of ``{episode, reason}``), ``contexts`` (list of ``{index, episode, demo_id, start}``).
``mask_stats``: over all frames of the used episodes: ``frames``, ``no_pose``, ``with_smoke``, ``static_ok``,
    ``enemy_excluded``, ``los_dropped_smokes`` (active smokes in the frustum that the line-of-sight filter left out of
    the mask, summed over frames; their pixels sit in the static region), ``frames_with_los_dropped`` and
    ``mean_smoke_fraction`` (of the frame, over frames with smoke).
``models``: name -> ``{checkpoint: {path, sha256_16}, upsampler: {path, sha256_16}}``.
``results``: model -> condition (``single`` | ``rollout``) -> ``{smoke: {...}, static: {...}, smoke_minus_static: {...}}``;
    ``smoke`` and ``static`` map ``lpips`` and ``ssim`` to ``{mean, ci_lo, ci_hi, n_contexts, n_frames, n_pixels}``
    (``n_pixels``: region pixels summed over the counted frames; for SSIM only those with an SSIM value);
    ``smoke_minus_static`` maps them to ``{mean, ci_lo, ci_hi, n_contexts}``. Positive LPIPS and negative SSIM
    differences mean smoke is harder.
``warnings``; ``limitations``; ``versions``; ``files``.

Other outputs in ``out_dir``: ``frames.csv`` (one row per model, condition, context and step), ``mask_validation/*.png``,
``triptychs/<model>_<condition>_c<ctx>_s<step>.png`` and ``<model>_triptychs.{pdf,png}`` (GT | generated | mask).
"""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from cs2wm import demo
from cs2wm.config import ticks_per_frame
from cs2wm.eval import smoke_masks as sm
from cs2wm.eval.h1_controllability import (
    V1_MODEL_NAME,
    AgentFactory,
    _bgr_uint8,
    _json_clean,
    _sha16,
    _versions,
    bootstrap_ci,
    job_generator,
    list_episodes,
    make_agent_factory,
)
from cs2wm.eval.upsampler_check import ssim_map
from cs2wm.pipeline.pack import read_episode, to_low_res
from cs2wm.play.world_model import WorldModel, frames_to_tensor
from cs2wm.timeline import frame_to_tick

SCHEMA_VERSION = 1
N_COND = 4  # context frames of the denoiser (num_steps_conditioning)
CONDITIONS = ("single", "rollout")
REGIONS = ("smoke", "static")
METRICS = ("lpips", "ssim")
# CONFIRM(#14): demoparser2 prop names (position, view angles, ducking, health, identity of every player).
TICK_PROPS = ("X", "Y", "Z", "yaw", "pitch", "duck_amount", "health", "name", "user_id")
LIMITATIONS = (
    (
        "Smoke masks are geometric approximations: each smoke is a sphere of eval.h2.smoke_radius_units around its "
        "detonation point, and occlusion by walls and other geometry is ignored, so a smoke behind a wall still masks "
        "pixels. The real volume is irregular and dissipates before it expires."
    ),
    (
        "The static mask keeps pixels outside the projected smokes and the viewmodel box on frames with no living "
        "player in view; it can still contain thin smoke edges beyond the dilation, effects, and dynamic objects "
        "such as grenades."
    ),
    "Contexts of one episode are correlated; the bootstrap resamples contexts, not episodes.",
)

# Maps (pred, target) in [-1, 1], float[b, 3, H, W], to a spatial LPIPS map float[b, H, W].
LpipsMapFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# --- LPIPS and masked statistics ---------------------------------------------------------------------------------


def make_lpips_map_fn(device: torch.device) -> LpipsMapFn:
    """Spatial AlexNet LPIPS from the ``lpips`` package (downloads the torchvision AlexNet weights on first use).

    Args:
        device: Device to run it on.

    Returns:
        A function ``(pred, target) -> float[b, H, W]`` taking frames in [-1, 1]; the per-layer maps are upsampled to
        the image size by the package.
    """
    import lpips

    model = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval().requires_grad_(False)

    def fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return model(pred.to(device), target.to(device))[:, 0].cpu()

    return fn


def score_maps(
    pred: torch.Tensor, target: torch.Tensor, lpips_map_fn: LpipsMapFn, device: torch.device, chunk: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """LPIPS and SSIM maps of generated frames against ground truth.

    Args:
        pred: ``float[n, 3, H, W]`` RGB in [-1, 1].
        target: ``float[n, 3, H, W]`` RGB in [-1, 1].
        lpips_map_fn: Spatial LPIPS.
        device: Device for SSIM.
        chunk: Frames per forward pass.

    Returns:
        ``(lpips, ssim)``, both ``float32[n, H, W]``. The SSIM map is NaN in the 5 px border its window cannot reach.
    """
    lp, ss = [], []
    for i in range(0, len(pred), chunk):
        p, t = pred[i : i + chunk], target[i : i + chunk]
        lp.append(lpips_map_fn(p, t).float().cpu())
        m = ssim_map(p.to(device).add(1).div(2), t.to(device).add(1).div(2)).cpu()
        ss.append(torch.nn.functional.pad(m, (5, 5, 5, 5), value=float("nan")))
    return torch.cat(lp).numpy(), torch.cat(ss).numpy()


def masked_mean(values: np.ndarray, mask: np.ndarray | None) -> tuple[float, int]:
    """Mean of a ``[H, W]`` map over a boolean mask, ignoring NaN entries.

    Args:
        values: ``float[H, W]`` map.
        mask: ``bool[H, W]`` or None.

    Returns:
        ``(mean, n)`` with ``n`` the pixels used; ``(nan, 0)`` if there are none.
    """
    if mask is None:
        return float("nan"), 0
    v = values[mask]
    v = v[np.isfinite(v)]
    return (float(v.mean()) if len(v) else float("nan")), len(v)


# --- demo tables and poses ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FrameGeom:
    """Camera and scene of one frame.

    Attributes:
        tick: Demo tick of the frame.
        eye: ``float[3]`` eye position.
        yaw: Source yaw in degrees.
        pitch: Source pitch in degrees (positive looks down).
        enemies: ``float[k, 3]`` origins of the other living players.
    """

    tick: int
    eye: np.ndarray
    yaw: float
    pitch: float
    enemies: np.ndarray


@dataclass(frozen=True, eq=False)
class DemoTables:
    """Everything H2 needs from one demo, parsed once.

    Attributes:
        smokes: Smoke table (``cs2wm.demo.SMOKE_COLUMNS``).
        ticks: Tick table of all players at the needed ticks.
    """

    smokes: pd.DataFrame
    ticks: pd.DataFrame


def _attr_str(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def read_attrs(path: Path) -> dict[str, Any]:
    """Read the attrs of a packed episode without its frames."""
    with h5py.File(path, "r") as f:
        return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in f.attrs.items()}


def frame_ticks(attrs: dict[str, Any], n_frames: int, cfg: DictConfig) -> np.ndarray:
    """Demo ticks at which the frames of an episode are shown: ``T_k = start_tick + k * ticks_per_frame``.

    ``render.tick_offset`` is not applied: it only shifts when the render commands are scheduled
    (``pipeline/segment.py`` ``mirv_tick = demo_tick + tick_offset``), so frame ``k`` still shows ``T_k``.
    """
    return frame_to_tick(np.arange(n_frames), int(attrs["start_tick"]), ticks_per_frame(cfg))


def load_demo_tables(
    demo_path: Path, ticks: np.ndarray, cfg: DictConfig, events_path: Path | None = None
) -> DemoTables:
    """Parse one demo: the tick table at ``ticks`` (plus the fill margin) and the smoke events.

    Args:
        demo_path: ``.dem`` file.
        ticks: Frame ticks needed.
        cfg: A validated config.
        events_path: A saved ``events/<demo_id>.parquet`` to use instead of parsing the smoke events.

    Returns:
        The tables.
    """
    fill = int(cfg.qa.max_fill_ticks)
    wanted = np.unique((ticks[:, None] + np.arange(-fill, fill + 1)[None]).ravel())
    props = list(TICK_PROPS)
    if cfg.eval.h2.view_offset_prop is not None:
        props.append(str(cfg.eval.h2.view_offset_prop))
    table = demo.load_ticks(demo_path, props, ticks=wanted.tolist())
    smokes = pd.read_parquet(events_path) if events_path is not None and events_path.exists() else None
    if smokes is None:
        smokes = demo.smoke_events(demo_path)
    return DemoTables(smokes, table)


def build_geoms(attrs: dict[str, Any], n_frames: int, tables: DemoTables, cfg: DictConfig) -> list[FrameGeom | None]:
    """Pose and enemy positions of every frame of an episode.

    Args:
        attrs: The episode's attrs (``start_tick``, ``user_id`` or ``player``).
        n_frames: Frames in the episode.
        tables: The demo's tables.
        cfg: A validated config.

    Returns:
        One entry per frame; None where the demo has no usable pose near the frame's tick.

    Raises:
        ValueError: If the spectated player cannot be identified in the tick table.
    """
    df = tables.ticks
    uid = attrs.get("user_id")
    if uid not in (None, "", -1) and "user_id" in df.columns:
        is_player = df["user_id"] == int(uid)
    elif attrs.get("player") and "name" in df.columns:
        is_player = df["name"] == _attr_str(attrs["player"])
    else:
        raise ValueError("cannot identify the spectated player: the episode has no user_id and no player name")
    own = df[is_player].drop_duplicates("tick").sort_values("tick")
    if own.empty:
        raise ValueError(f"spectated player {uid or attrs.get('player')!r} is not in the tick table")
    ticks = frame_ticks(attrs, n_frames, cfg)
    present = own["tick"].to_numpy(dtype=np.int64)
    pos = np.clip(np.searchsorted(present, ticks), 1, len(present) - 1) if len(present) > 1 else np.zeros(n_frames, int)
    left = np.maximum(pos - 1, 0)
    nearest = np.where(np.abs(present[left] - ticks) <= np.abs(present[pos] - ticks), left, pos)
    ok = np.abs(present[nearest] - ticks) <= int(cfg.qa.max_fill_ticks)

    others = df[~is_player]
    if "health" in others.columns:
        others = others[pd.to_numeric(others["health"], errors="coerce") > 0]
    others = others.dropna(subset=["X", "Y", "Z"])
    by_tick = {int(t): g[["X", "Y", "Z"]].to_numpy(np.float64) for t, g in others.groupby("tick")}
    offset_col = cfg.eval.h2.view_offset_prop
    geoms: list[FrameGeom | None] = []
    for k in range(n_frames):
        if not ok[k]:
            geoms.append(None)
            continue
        row = own.iloc[nearest[k]]
        origin = np.array([row["X"], row["Y"], row["Z"]], dtype=np.float64)
        yaw, pitch = float(row["yaw"]), float(row["pitch"])
        if not (np.isfinite(origin).all() and np.isfinite([yaw, pitch]).all()):
            geoms.append(None)
            continue
        offset = float(row[offset_col]) if offset_col is not None and offset_col in own.columns else None
        eye = sm.eye_position(origin, float(row.get("duck_amount", 0.0)), cfg, offset)
        tick = int(present[nearest[k]])
        geoms.append(FrameGeom(tick, eye, yaw, pitch, by_tick.get(tick, np.zeros((0, 3)))))
    return geoms


def frame_regions(
    geom: FrameGeom | None, smokes: pd.DataFrame, cfg: DictConfig, size: Sequence[int]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Smoke and static masks of one frame.

    Args:
        geom: The frame's pose, or None.
        smokes: The demo's smoke table.
        cfg: A validated config.
        size: Image ``(H, W)``.

    Returns:
        ``(smoke, static)``: ``(None, None)`` without a pose; ``static`` is None when an enemy is in view.
    """
    if geom is None:
        return None, None
    smoke = sm.smoke_mask_for_frame(geom.tick, smokes, geom.eye, geom.yaw, geom.pitch, cfg, size)
    static = sm.static_mask_for_frame(smoke, geom.enemies, geom.eye, geom.yaw, geom.pitch, cfg)
    return smoke, static


# --- episodes and contexts ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EpisodeInfo:
    """A held-out smoke episode with its demo tables.

    Attributes:
        path: Packed full-res episode.
        attrs: Its attrs.
        n_frames: Frames.
        tables: Tables of its demo (shared with the other episodes of the demo).
        geoms: Per-frame pose.
        smoke_px: ``int[n]`` smoke-mask pixels per frame (0 without a pose).
        counts: Frame counts by mask status (``mask_stats`` of ``summary.json``, before aggregation).
    """

    path: Path
    attrs: dict[str, Any]
    n_frames: int
    tables: DemoTables
    geoms: list[FrameGeom | None]
    smoke_px: np.ndarray
    counts: dict[str, float]

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def demo_id(self) -> str:
        return _attr_str(self.attrs.get("demo_id", self.path.stem))


@dataclass(frozen=True, eq=False)
class H2Context:
    """A held-out context: ``episode.frames[start:start + 4]`` and the ``rollout_steps`` target frames after them.

    Attributes:
        index: Position in the sampled list (seeds the noise).
        episode: The episode.
        start: Index of the first context frame.
    """

    index: int
    episode: EpisodeInfo
    start: int


def prepare_episodes(
    files: Sequence[Path],
    demos_dir: Path,
    events_dir: Path | None,
    cfg: DictConfig,
    loader: Callable[..., DemoTables] = load_demo_tables,
) -> tuple[list[EpisodeInfo], list[dict[str, str]]]:
    """Keep the smoke episodes whose demo is available and attach their poses.

    Args:
        files: Held-out packed episodes.
        demos_dir: Directory of ``<demo_id>.dem`` files.
        events_dir: Directory of saved ``<demo_id>.parquet`` smoke tables, or None.
        cfg: A validated config.
        loader: ``(demo_path, ticks, cfg, events_path) -> DemoTables``.

    Returns:
        ``(episodes, skipped)`` where ``skipped`` lists ``{episode, reason}`` for the dropped files.
    """
    by_demo: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    skipped: list[dict[str, str]] = []
    for path in files:
        attrs = read_attrs(path)
        if _attr_str(attrs.get("kind", "")) != "smoke":
            skipped.append({"episode": path.stem, "reason": f"kind is {attrs.get('kind')!r}, not 'smoke'"})
        elif not (Path(demos_dir) / f"{_attr_str(attrs.get('demo_id', path.stem))}.dem").exists():
            skipped.append({"episode": path.stem, "reason": f"demo {attrs.get('demo_id')!r} not found in {demos_dir}"})
        else:
            by_demo.setdefault(_attr_str(attrs.get("demo_id", path.stem)), []).append((path, attrs))
    episodes: list[EpisodeInfo] = []
    for demo_id, group in sorted(by_demo.items()):
        ticks = np.concatenate([frame_ticks(a, int(a["n_frames"]), cfg) for _, a in group])
        events = Path(events_dir) / f"{demo_id}.parquet" if events_dir is not None else None
        tables = loader(Path(demos_dir) / f"{demo_id}.dem", ticks, cfg, events)
        for path, attrs in group:
            n = int(attrs["n_frames"])
            try:
                geoms = build_geoms(attrs, n, tables, cfg)
            except ValueError as e:
                skipped.append({"episode": path.stem, "reason": str(e)})
                continue
            episodes.append(scan_episode(path, attrs, n, tables, geoms, cfg))
    return episodes, skipped


def scan_episode(
    path: Path, attrs: dict[str, Any], n: int, tables: DemoTables, geoms: list[FrameGeom | None], cfg: DictConfig
) -> EpisodeInfo:
    """Compute every frame's masks once to find the smoke pixels and count frames by mask status."""
    size = tuple(int(v) for v in cfg.resolution.render)
    min_px = int(cfg.eval.h2.min_region_px)
    smoke_px = np.zeros(n, dtype=np.int64)
    counts = {
        "frames": n,
        "no_pose": 0,
        "with_smoke": 0,
        "static_ok": 0,
        "enemy_excluded": 0,
        "los_dropped_smokes": 0,
        "frames_with_los_dropped": 0,
        "smoke_fraction_sum": 0.0,
    }
    for k, geom in enumerate(geoms):
        smoke, static = frame_regions(geom, tables.smokes, cfg, size)
        if smoke is None:
            counts["no_pose"] += 1
            continue
        dropped = sm.los_dropped_count(geom.tick, tables.smokes, geom.eye, geom.yaw, geom.pitch, cfg, size)
        counts["los_dropped_smokes"] += dropped
        counts["frames_with_los_dropped"] += dropped > 0
        smoke_px[k] = smoke.sum()
        if smoke_px[k] >= min_px:
            counts["with_smoke"] += 1
            counts["smoke_fraction_sum"] += float(smoke.mean())
        if static is None:
            counts["enemy_excluded"] += 1
        elif static.sum() >= min_px:
            counts["static_ok"] += 1
    return EpisodeInfo(path, attrs, n, tables, geoms, smoke_px, counts)


def mask_statistics(episodes: Sequence[EpisodeInfo]) -> dict[str, Any]:
    """Sum the frame counts by mask status over the episodes."""
    total = {k: sum(ep.counts[k] for ep in episodes) for k in episodes[0].counts}
    with_smoke = total.pop("smoke_fraction_sum")
    total["mean_smoke_fraction"] = with_smoke / total["with_smoke"] if total["with_smoke"] else float("nan")
    return {k: int(v) if k != "mean_smoke_fraction" else v for k, v in total.items()}


def select_contexts(episodes: Sequence[EpisodeInfo], n: int, steps: int, min_px: int, min_frames: int, seed: int):
    """Draw ``n`` contexts deterministically, spread over the episodes.

    A window ``[s, s + 4 + steps)`` qualifies when at least ``min_frames`` of its ``steps`` target frames have a smoke
    mask of ``min_px`` pixels. Episodes are visited in a seeded random order, cycling; each visit takes a random
    qualifying window that does not overlap one already taken from that episode.

    Args:
        episodes: Candidate episodes.
        n: Contexts wanted.
        steps: Target frames per context.
        min_px: Smoke pixels a target frame needs to count.
        min_frames: Counting target frames a window needs.
        seed: RNG seed.

    Returns:
        Up to ``n`` contexts sorted by (episode, start) and indexed in that order; fewer when the data run out.

    Raises:
        ValueError: If ``n < 1`` or no window qualifies.
    """
    if n < 1:
        raise ValueError(f"contexts must be at least 1, got {n}")
    span = N_COND + steps
    pools: dict[int, np.ndarray] = {}
    for i, ep in enumerate(episodes):
        if ep.n_frames < span:
            continue
        has = (ep.smoke_px >= min_px).astype(np.int64)
        cum = np.concatenate([[0], np.cumsum(has)])
        starts = np.arange(ep.n_frames - span + 1)
        count = cum[starts + span] - cum[starts + N_COND]
        if (good := starts[count >= min_frames]).size:
            pools[i] = good
    if not pools:
        raise ValueError(
            f"no window of {span} frames has {min_frames} target frames with a smoke of at least {min_px} px in view"
        )
    rng = np.random.default_rng(seed)
    order = [int(i) for i in rng.permutation(sorted(pools))]
    picks: list[tuple[int, int]] = []
    while len(picks) < n and pools:
        for i in order:
            if i not in pools:
                continue
            s = int(rng.choice(pools[i]))
            picks.append((i, s))
            left = pools[i][np.abs(pools[i] - s) >= span]
            if left.size:
                pools[i] = left
            else:
                del pools[i]
            if len(picks) == n:
                break
    return [H2Context(k, episodes[i], s) for k, (i, s) in enumerate(sorted(picks))]


class EpisodeReader:
    """Reads packed episodes, keeping the last one (contexts are processed episode by episode)."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._data: tuple[np.ndarray, np.ndarray] | None = None

    def __call__(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frames_bgr, actions)`` of the episode."""
        if self._path != path or self._data is None:
            frames, actions, _ = read_episode(path)
            self._path, self._data = path, (frames, actions)
        return self._data


# --- predictions and scoring -------------------------------------------------------------------------------------


def _noise_name(t: int) -> str:
    # Step 1 of the single-step condition is step 1 of the rollout: same context, action and noise.
    return "rollout" if t == 0 else f"single{t}"


FRAME_COLUMNS = (
    "model",
    "condition",
    "context",
    "episode",
    "demo_id",
    "start",
    "step",
    "frame",
    "tick",
    "status",
    "smoke_px",
    "static_px",
    "lpips_smoke",
    "lpips_static",
    "ssim_smoke",
    "ssim_static",
    "ssim_px_smoke",
    "ssim_px_static",
)


def evaluate_model(
    name: str,
    world: WorldModel,
    contexts: Sequence[H2Context],
    reader: EpisodeReader,
    cfg: DictConfig,
    lpips_map_fn: LpipsMapFn,
    seed: int,
    batch_size: int,
    steps: int,
    device: torch.device,
    example_keys: set[tuple[int, str, int]] = frozenset(),
    progress: Callable[[int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str, int], np.ndarray]]:
    """Predict and score every context for one model.

    Args:
        name: Model name (a column of the rows).
        world: The model's world model.
        contexts: Held-out contexts.
        reader: Episode reader.
        cfg: A validated config.
        lpips_map_fn: Spatial LPIPS.
        seed: Run seed (noise is seeded by it, the context index and the step).
        batch_size: Rollouts per batched forward pass.
        steps: Target frames per context.
        device: Device for the metrics.
        example_keys: ``(context, condition, step)`` of generated frames to keep for the triptychs.
        progress: Called with the number of contexts finished after each batch.

    Returns:
        ``(rows, examples)``: one row (``FRAME_COLUMNS``) per context, condition and step, and the kept generated
        frames as ``uint8[H, W, 3]`` BGR.
    """
    size = tuple(int(v) for v in cfg.resolution.render)
    low_size = tuple(int(v) for v in cfg.resolution.low_res)
    min_px = int(cfg.eval.h2.min_region_px)
    rows: list[dict[str, Any]] = []
    examples: dict[tuple[int, str, int], np.ndarray] = {}
    for b0 in range(0, len(contexts), batch_size):
        batch = contexts[b0 : b0 + batch_size]
        low, acts, gt = [], [], []
        for c in batch:
            frames, actions = reader(c.episode.path)
            s = c.start
            low.append(frames_to_tensor(to_low_res(frames[s : s + steps + N_COND - 1], low_size)))
            acts.append(torch.from_numpy(actions[s : s + steps + N_COND - 1]))
            gt.append(frames_to_tensor(frames[s + N_COND : s + N_COND + steps]))
        # rollout: one stream per context
        gen_streams = [job_generator(seed, c.index, "rollout") for c in batch]
        _, full = world.rollout(
            torch.stack([x[:N_COND] for x in low]),
            torch.stack([a[:N_COND] for a in acts]),
            torch.stack([a[N_COND - 1 : N_COND - 1 + steps] for a in acts]),
            gen_streams,
        )
        gens = {"rollout": [_bgr_uint8(full[i]) for i in range(len(batch))]}
        # single-step: one stream per (context, step)
        jobs = [(i, t) for i in range(len(batch)) for t in range(steps)]
        single: dict[tuple[int, int], np.ndarray] = {}
        for j0 in range(0, len(jobs), batch_size):
            part = jobs[j0 : j0 + batch_size]
            _, full = world.rollout(
                torch.stack([low[i][t : t + N_COND] for i, t in part]),
                torch.stack([acts[i][t : t + N_COND] for i, t in part]),
                torch.stack([acts[i][t + N_COND - 1][None] for i, t in part]),
                [job_generator(seed, batch[i].index, _noise_name(t)) for i, t in part],
            )
            for k, (i, t) in enumerate(part):
                single[(i, t)] = _bgr_uint8(full[k])[0]
        gens["single"] = [np.stack([single[(i, t)] for t in range(steps)]) for i in range(len(batch))]

        regions = {
            (i, t): frame_regions(c.episode.geoms[c.start + N_COND + t], c.episode.tables.smokes, cfg, size)
            for i, c in enumerate(batch)
            for t in range(steps)
        }
        for cond in CONDITIONS:
            pred = torch.cat([frames_to_tensor(gens[cond][i]) for i in range(len(batch))])
            lp, ss = score_maps(pred, torch.cat(gt), lpips_map_fn, device)
            for i, c in enumerate(batch):
                for t in range(steps):
                    n = i * steps + t
                    frame = c.start + N_COND + t
                    geom = c.episode.geoms[frame]
                    smoke, static = regions[(i, t)]
                    status = "no_pose" if geom is None else ("enemy" if static is None else "ok")
                    row: dict[str, Any] = {
                        "model": name,
                        "condition": cond,
                        "context": c.index,
                        "episode": c.episode.name,
                        "demo_id": c.episode.demo_id,
                        "start": c.start,
                        "step": t + 1,
                        "frame": frame,
                        "tick": geom.tick if geom is not None else "",
                        "status": status,
                        "smoke_px": 0 if smoke is None else int(smoke.sum()),
                        "static_px": 0 if static is None else int(static.sum()),
                    }
                    for region, mask in (("smoke", smoke), ("static", static)):
                        usable = mask if mask is not None and mask.sum() >= min_px else None
                        row[f"lpips_{region}"], _ = masked_mean(lp[n], usable)
                        row[f"ssim_{region}"], row[f"ssim_px_{region}"] = masked_mean(ss[n], usable)
                    rows.append(row)
                    if (c.index, cond, t + 1) in example_keys:
                        examples[(c.index, cond, t + 1)] = gens[cond][i][t]
        if progress is not None:
            progress(len(batch))
    return rows, examples


# --- statistics --------------------------------------------------------------------------------------------------


def _cell(values: np.ndarray, frames: pd.Series, pixels: pd.Series, n_boot: int, seed: int) -> dict[str, Any]:
    mean, lo, hi = bootstrap_ci(values, n_boot, seed)
    return {
        "mean": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_contexts": int(np.isfinite(values).sum()),
        "n_frames": int(frames.notna().sum()),
        "n_pixels": int(pixels[frames.notna()].sum()),
    }


def summarize_rows(rows: Sequence[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    """Aggregate per-frame rows into model -> condition -> region -> metric statistics.

    Args:
        rows: Rows with the ``FRAME_COLUMNS``.
        n_boot: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        The ``results`` block of ``summary.json``.
    """
    df = pd.DataFrame(rows, columns=list(FRAME_COLUMNS))
    out: dict[str, Any] = {}
    for (model, cond), g in df.groupby(["model", "condition"], sort=False):
        res: dict[str, Any] = {}
        per_ctx: dict[tuple[str, str], pd.Series] = {}
        for region in REGIONS:
            res[region] = {}
            for metric in METRICS:
                col = f"{metric}_{region}"
                ctx = g.groupby("context")[col].mean()
                per_ctx[(region, metric)] = ctx
                px = (
                    g["static_px" if region == "static" else "smoke_px"]
                    if metric == "lpips"
                    else g[f"ssim_px_{region}"]
                )
                res[region][metric] = _cell(ctx.to_numpy(np.float64), g[col], px, n_boot, seed)
        res["smoke_minus_static"] = {}
        for metric in METRICS:
            diff = (per_ctx[("smoke", metric)] - per_ctx[("static", metric)]).to_numpy(np.float64)
            mean, lo, hi = bootstrap_ci(diff, n_boot, seed)
            res["smoke_minus_static"][metric] = {
                "mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_contexts": int(np.isfinite(diff).sum()),
            }
        out.setdefault(str(model), {})[str(cond)] = res
    return out


# --- images ------------------------------------------------------------------------------------------------------

OVERLAY_SCALE = 4
SMOKE_TINT = (0, 0, 255)  # BGR red
STATIC_OUTLINE = (0, 255, 0)
BOX_COLOUR = (0, 255, 255)


def overlay_masks(
    frame_bgr: np.ndarray, smoke: np.ndarray | None, static: np.ndarray | None, box: np.ndarray, label: str
) -> np.ndarray:
    """GT frame with the smoke mask tinted red, the static mask outlined green and the viewmodel box in yellow.

    Args:
        frame_bgr: ``uint8[H, W, 3]`` ground-truth frame.
        smoke: ``bool[H, W]`` smoke mask, or None without a pose.
        static: ``bool[H, W]`` static mask, or None when an enemy is in view or without a pose.
        box: ``bool[H, W]`` viewmodel box.
        label: Text drawn at the top.

    Returns:
        The upscaled ``uint8`` image.
    """
    k = OVERLAY_SCALE
    img = cv2.resize(frame_bgr, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)

    def up(mask: np.ndarray) -> np.ndarray:
        return cv2.resize(mask.astype(np.uint8), None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)

    if smoke is not None and smoke.any():
        tinted = (0.55 * img + 0.45 * np.array(SMOKE_TINT, dtype=np.float64)).astype(np.uint8)
        img = np.where(up(smoke)[..., None] > 0, tinted, img)
    if static is not None:
        contours, _ = cv2.findContours(up(static), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, STATIC_OUTLINE, 1)
    elif smoke is not None:
        label += "  ENEMY IN VIEW: no static mask"
    contours, _ = cv2.findContours(up(box), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, BOX_COLOUR, 1)
    cv2.putText(img, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def mask_panel(smoke: np.ndarray | None, static: np.ndarray | None, size: Sequence[int]) -> np.ndarray:
    """``uint8[H, W, 3]`` BGR panel: smoke red, static green, everything else dark grey."""
    panel = np.full((*size, 3), 40, dtype=np.uint8)
    if static is not None:
        panel[static] = (0, 140, 0)
    if smoke is not None:
        panel[smoke] = (0, 0, 220)
    return panel


def pick_frames(candidates: Sequence[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    """``n`` entries of ``candidates`` spread evenly over the list (all of them if there are fewer)."""
    if len(candidates) <= n:
        return list(candidates)
    return [candidates[int(i)] for i in np.linspace(0, len(candidates) - 1, n).round()]


def target_frames(
    contexts: Sequence[H2Context], steps: int, min_px: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split the ``(context index, step)`` target frames into those with a smoke of ``min_px`` pixels and the rest."""
    smoky, other = [], []
    for c in contexts:
        for t in range(steps):
            (smoky if c.episode.smoke_px[c.start + N_COND + t] >= min_px else other).append((c.index, t + 1))
    return smoky, other


def write_mask_validation(
    contexts: Sequence[H2Context], reader: EpisodeReader, cfg: DictConfig, n: int, steps: int, out_dir: Path
) -> list[tuple[int, int]]:
    """Write overlay PNGs of ``n`` target frames, preferring frames with a smoke in view.

    Args:
        contexts: Held-out contexts.
        reader: Episode reader.
        cfg: A validated config.
        n: Overlays wanted.
        steps: Target frames per context.
        out_dir: Directory for the PNGs (created).

    Returns:
        The chosen ``(context index, step)`` pairs (steps are 1-based).
    """
    size = tuple(int(v) for v in cfg.resolution.render)
    min_px = int(cfg.eval.h2.min_region_px)
    smoky, other = target_frames(contexts, steps, min_px)
    chosen = pick_frames(smoky, n)
    if len(chosen) < n:
        chosen += pick_frames(other, n - len(chosen))
    by_index = {c.index: c for c in contexts}
    out_dir.mkdir(parents=True, exist_ok=True)
    box = sm.viewmodel_mask(cfg, size)
    for k, (ci, step) in enumerate(sorted(chosen)):
        c = by_index[ci]
        frame = c.start + N_COND + step - 1
        gt = reader(c.episode.path)[0][frame]
        smoke, static = frame_regions(c.episode.geoms[frame], c.episode.tables.smokes, cfg, size)
        label = f"{c.episode.name} f{frame} tick {c.episode.geoms[frame].tick if c.episode.geoms[frame] else '-'}"
        label += f" smoke {0 if smoke is None else int(smoke.sum())} px"
        cv2.imwrite(
            str(out_dir / f"{k:02d}_{c.episode.name}_f{frame:04d}.png"), overlay_masks(gt, smoke, static, box, label)
        )
    return sorted(chosen)


def write_triptychs(
    model: str,
    keys: Sequence[tuple[int, str, int]],
    examples: dict[tuple[int, str, int], np.ndarray],
    rows: Sequence[dict[str, Any]],
    contexts: Sequence[H2Context],
    reader: EpisodeReader,
    cfg: DictConfig,
    out_dir: Path,
) -> list[str]:
    """Write GT | generated | mask triptychs of a model as PNGs and one PDF/PNG grid.

    Args:
        model: Model name.
        keys: ``(context, condition, step)`` of the examples, in display order.
        examples: Generated frames by key (``uint8`` BGR).
        rows: The model's per-frame rows (for the scores in the titles).
        contexts: Held-out contexts.
        reader: Episode reader.
        cfg: A validated config.
        out_dir: Output directory (the PNGs go to ``triptychs/``).

    Returns:
        File names relative to ``out_dir``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = tuple(int(v) for v in cfg.resolution.render)
    by_index = {c.index: c for c in contexts}
    score = {(r["context"], r["condition"], r["step"]): r for r in rows if r["model"] == model}
    keys = [k for k in keys if k in examples]
    (out_dir / "triptychs").mkdir(parents=True, exist_ok=True)
    files = []
    fig, axes = plt.subplots(max(len(keys), 1), 3, figsize=(10, 2.0 * max(len(keys), 1)), squeeze=False)
    for r, key in enumerate(keys):
        ci, cond, step = key
        c = by_index[ci]
        frame = c.start + N_COND + step - 1
        gt = reader(c.episode.path)[0][frame]
        smoke, static = frame_regions(c.episode.geoms[frame], c.episode.tables.smokes, cfg, size)
        panels = [gt, examples[key], mask_panel(smoke, static, size)]
        png = f"triptychs/{model}_{cond}_c{ci}_s{step}.png"
        cv2.imwrite(
            str(out_dir / png), cv2.resize(np.hstack(panels), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        )
        files.append(png)
        s = score[key]
        caption = f"{cond} step {step}, {c.episode.name} f{frame}\nLPIPS smoke {s['lpips_smoke']:.3f} static {s['lpips_static']:.3f}"
        for col, (panel, title) in enumerate(
            zip(panels, ("ground truth", "generated", "mask (red smoke, green static)"))
        ):
            ax = axes[r][col]
            ax.imshow(panel[..., ::-1])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8)
        axes[r][0].set_ylabel(caption, fontsize=6)
    fig.suptitle(f"H2 examples: {model}", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{model}_triptychs.{ext}", dpi=150)
        files.append(f"{model}_triptychs.{ext}")
    plt.close(fig)
    return files


def write_frames_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write the per-frame rows (NaN as empty cells)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FRAME_COLUMNS)
        for r in rows:
            w.writerow(["" if isinstance(r[c], float) and not np.isfinite(r[c]) else r[c] for c in FRAME_COLUMNS])


# --- driver ------------------------------------------------------------------------------------------------------


def run_h2(
    models: dict[str, Path],
    cfg: DictConfig,
    out_dir: Path,
    seed: int = 0,
    data_dir: Path | None = None,
    split_json: Path | None = None,
    demos_dir: Path | None = None,
    upsampler_ckpt: Path | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 16,
    rollout_steps: int = 16,
    n_validate: int = 20,
    contexts: int | None = None,
    denoising_steps: int = 1,
    agent_factory: AgentFactory | None = None,
    lpips_map_fn: LpipsMapFn | None = None,
    events_dir: Path | None = None,
    v1_name: str = V1_MODEL_NAME,
) -> dict:
    """Run the H2 comparison.

    Args:
        models: Mapping from name (``v2``, ``v1``) to checkpoint path.
        cfg: A validated cs2wm config.
        out_dir: Output directory (``results/h2``).
        seed: RNG seed (context sampling, rollout noise, bootstrap).
        data_dir: Directory of held-out full-res (150x280) packed episodes.
        split_json: Optional ``split.json`` keeping only test demos.
        demos_dir: Directory of ``<demo_id>.dem`` files (default ``paths.demos``).
        upsampler_ckpt: v1 checkpoint providing the upsampler for models other than ``v1_name`` (default: the
            ``v1_name`` model's checkpoint).
        device: Torch device; defaults to cuda if available.
        batch_size: Rollouts per batched forward pass.
        rollout_steps: Frames per rollout (and single-step targets per context).
        n_validate: Mask overlay PNGs to write.
        contexts: Contexts to draw (default ``eval.h2.contexts``).
        denoising_steps: Sampler preset, 1 or 3.
        agent_factory: ``(name, ckpt) -> agent``; defaults to ``make_agent_factory``.
        lpips_map_fn: Spatial LPIPS; defaults to AlexNet from the ``lpips`` package.
        events_dir: Directory of saved smoke tables (default ``paths.events``).
        v1_name: Name of the model that uses its own upsampler (the v1 model).

    Returns:
        The summary dict that is written to ``summary.json``.

    Raises:
        ValueError: If ``data_dir`` is missing, ``models`` is empty, no smoke episode can be used, no context qualifies
            or the data are not full-resolution.
    """
    if data_dir is None:
        raise ValueError("data_dir (--data) is required")
    if not models:
        raise ValueError("no models given")
    h = cfg.eval.h2
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    demos_dir = Path(demos_dir) if demos_dir is not None else Path(cfg.paths.demos)
    events_dir = Path(events_dir) if events_dir is not None else Path(cfg.paths.events)
    n_ctx = int(contexts if contexts is not None else h.contexts)
    n_boot = int(h.bootstrap)
    size = tuple(int(v) for v in cfg.resolution.render)
    if rollout_steps < 1:
        raise ValueError(f"rollout_steps must be at least 1, got {rollout_steps}")
    if upsampler_ckpt is None and v1_name in models:
        upsampler_ckpt = models[v1_name]
    upsamplers = {name: Path(ckpt) if name == v1_name else upsampler_ckpt for name, ckpt in models.items()}
    if agent_factory is None and any(up is None for up in upsamplers.values()):
        raise ValueError(f"models without an upsampler: pass --upsampler-ckpt or a model named {v1_name!r}")
    for name, up in upsamplers.items():
        source = "its own checkpoint" if name == v1_name else "--upsampler-ckpt / the v1 model"
        print(f"[h2] {name}: denoiser {models[name]}, upsampler {up} ({source})")
    factory = agent_factory or make_agent_factory(upsampler_ckpt, dev, v1_name)

    files = list_episodes(data_dir, split_json)
    episodes, skipped = prepare_episodes(files, demos_dir, events_dir, cfg)
    if not episodes:
        raise ValueError(f"no usable smoke episode in {data_dir} ({len(skipped)} skipped: {skipped[:3]})")
    probe = read_attrs(episodes[0].path)
    if int(probe.get("height", 0)) != size[0] or int(probe.get("width", 0)) != size[1]:
        raise ValueError(
            f"{data_dir}: H2 needs {size[0]}x{size[1]} full-res episodes (hdf5_fullres), got "
            f"{probe.get('height')}x{probe.get('width')}"
        )
    warnings: list[str] = [f"skipped {s['episode']}: {s['reason']}" for s in skipped]
    if split_json is None:
        warnings.append("NOT HELD OUT: no --split-json was given, so the contexts may come from training demos")
    ctxs = select_contexts(episodes, n_ctx, rollout_steps, int(h.min_region_px), int(h.min_smoke_frames), seed)
    if len(ctxs) < n_ctx:
        warnings.append(f"only {len(ctxs)} non-overlapping contexts available, {n_ctx} requested")
    print(f"[h2] {len(episodes)} smoke episodes, {len(ctxs)} contexts x {rollout_steps} steps")
    stats = mask_statistics(episodes)

    reader = EpisodeReader()
    write_mask_validation(ctxs, reader, cfg, n_validate, rollout_steps, out_dir / "mask_validation")
    smoky = target_frames(ctxs, rollout_steps, int(h.min_region_px))[0]  # independent of n_validate
    example_pairs = pick_frames(smoky, int(h.examples))
    example_keys = [(ci, cond, step) for ci, step in example_pairs for cond in CONDITIONS]

    lpips_fn = lpips_map_fn or make_lpips_map_fn(dev)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "hfov_deg": float(cfg.eval.h1.hfov_deg),
            "frame_hw": list(size),
            "smoke_radius_units": float(h.smoke_radius_units),
            "mask_dilate_px": int(h.mask_dilate_px),
            "max_los_units": float(h.max_los_units),
            "viewmodel_box": [float(v) for v in h.viewmodel_box],
            "enemy_max_dist_units": float(h.enemy_max_dist_units),
            "enemy_margin_px": float(h.enemy_margin_px),
            "min_region_px": int(h.min_region_px),
            "min_smoke_frames": int(h.min_smoke_frames),
            "rollout_steps": rollout_steps,
            "contexts": n_ctx,
            "bootstrap": n_boot,
            "denoising_steps": denoising_steps,
            "batch_size": batch_size,
            "device": str(dev),
        },
        "seed": seed,
        "data": {
            "dir": str(data_dir),
            "split_json": str(split_json) if split_json else None,
            "held_out": split_json is not None,
            "demos_dir": str(demos_dir),
            "n_episodes": len(episodes),
            "skipped": skipped,
            "contexts": [
                {"index": c.index, "episode": c.episode.name, "demo_id": c.episode.demo_id, "start": c.start}
                for c in ctxs
            ],
        },
        "mask_stats": stats,
        "models": {},
    }
    rows: list[dict[str, Any]] = []
    plots: list[str] = []
    for name, ckpt in models.items():
        print(f"[h2] evaluating {name}")
        agent = factory(name, Path(ckpt))
        world = WorldModel(agent, denoising_steps, use_upsampler=True)
        if world.n_cond != N_COND:
            raise ValueError(f"{name}: num_steps_conditioning is {world.n_cond}, H2 contexts have {N_COND} frames")
        up = upsamplers[name]
        summary["models"][name] = {
            "checkpoint": {"path": str(ckpt), "sha256_16": _sha16(Path(ckpt))},
            "upsampler": {"path": str(up), "sha256_16": _sha16(up) if up else None},
        }
        done = 0

        def progress(n: int, name: str = name) -> None:
            nonlocal done
            done += n
            print(f"[h2] {name}: {done}/{len(ctxs)} contexts", end="\r", flush=True)

        model_rows, examples = evaluate_model(
            name, world, ctxs, reader, cfg, lpips_fn, seed, batch_size, rollout_steps, dev, set(example_keys), progress
        )
        print()
        del world, agent
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        rows += model_rows
        plots += write_triptychs(name, example_keys, examples, model_rows, ctxs, reader, cfg, out_dir)

    summary["results"] = summarize_rows(rows, n_boot, seed)
    for name, by_cond in summary["results"].items():
        for cond, res in by_cond.items():
            for region in REGIONS:
                if res[region]["lpips"]["n_contexts"] < 2:
                    warnings.append(
                        f"{name}/{cond}: fewer than 2 contexts have a {region} region; its CI is not meaningful"
                    )
    write_frames_csv(out_dir / "frames.csv", rows)
    summary["warnings"] = warnings
    summary["limitations"] = list(LIMITATIONS)
    summary["versions"] = _versions()
    summary["files"] = ["frames.csv", "mask_validation/"] + plots + ["summary.json"]
    (out_dir / "summary.json").write_text(json.dumps(_json_clean(summary), indent=2) + "\n")
    for w in warnings:
        print(f"[h2] WARNING: {w}", file=sys.stderr)
    print(format_summary(summary))
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    """One line per model, condition and metric: smoke, static and the paired difference."""
    lines = []
    for model, by_cond in summary["results"].items():
        for cond, res in by_cond.items():
            for metric in METRICS:
                parts = []
                for key in (*REGIONS, "smoke_minus_static"):
                    c = res[key][metric]
                    parts.append(f"{key} {c['mean']:.4f} [{c['ci_lo']:.4f}, {c['ci_hi']:.4f}]")
                lines.append(f"{model:>8} {cond:>7} {metric:>5}: " + "  ".join(parts))
    return "\n".join(lines)
