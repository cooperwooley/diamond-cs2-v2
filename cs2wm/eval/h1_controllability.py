"""H1 controllability eval (#32): scripted inputs -> ego-rotation estimate -> correlation with the commanded input.

Claim under test: when a mouse turn is commanded, the generated frames rotate by the commanded amount. Compares v2, the
v1 zero-action model (fed the same actions) and the shuffled-action control.

Method
------
1. **Estimator validation on ground truth.** For pairs of consecutive ground-truth frames ``(k, k + 1)`` of held-out
   episodes the image shift is estimated and correlated with the label-implied rotation (the decoded mouse bin centre,
   in degrees, of action ``k``, which covers ``k -> k + 1``). Pearson r and the slope (degrees per pixel, fitted on
   ``-dx`` for yaw and ``-dy`` for pitch) are reported per estimator (phase correlation, Farneback), axis and
   resolution (150x280 needs ``--fullres-data``; 30x56 comes from ``--data``). Yaw r below ``eval.h1.r_threshold`` (0.8)
   is flagged in ``warnings`` and on stderr. The fitted slope is the conversion used for the model rollouts
   (the theoretical pinhole slope, ``rotation.theoretical_slopes``, is reported next to it, and used if no fit exists).
2. **Contexts.** ``--contexts`` windows of 4 ground-truth frames and actions are drawn deterministically (``--seed``)
   from the held-out episodes of ``--data``.
3. **Scripted suites** (``ScriptedSuite``): yaw at +-{2, 4, 10, 20, 30, 60} bins, pitch at +-{2, 4, 10, 20}, W, S, A, D
   and no-op, each held for ``eval.h1.rollout_frames`` frames. The actions are built with ``cs2wm.actions.encode`` from
   the configured bins.
4. **Rollouts.** Every (context, script) pair is rolled out for each model with ``WorldModel.rollout`` (batched; the
   same sampler and upsampler conditioning as play). Noise is seeded per (context, script) and shared by all models.
   v2 and shuffled take the upsampler from ``--upsampler-ckpt`` (default: the ``v1`` model's checkpoint); v1 uses its
   own. The shift is estimated between consecutive *generated* frames (``T - 1`` pairs per rollout; the context
   frame is not used because the first generated frame is not an upsampler output of the same kind), after the
   upsampler at 150x280, and at 30x56 on the low-res frames as a robustness check. Frames are quantised to uint8 first.
5. **Metrics** per model and resolution (all scripted frames pooled): Pearson r and least-squares slope of estimated vs
   commanded d-yaw and d-pitch, sign accuracy over non-zero commands, per-script mean +- 95% bootstrap CI (percentile
   bootstrap over *contexts* of the per-context mean, since frames of one rollout are correlated), and the drift under
   no-op (mean absolute estimated d-yaw/d-pitch per frame). For W/S the sign of the Farneback flow divergence.

Sign convention: positive d-yaw = turn right, positive d-pitch = look down (``cs2wm.actions``); see ``rotation``.

``summary.json`` (``schema_version`` 1)
---------------------------------------
``config``: ``hfov_deg``, ``capture_hw``, ``frame_hw``, ``low_res_hw``, ``rollout_frames``, ``estimator``,
    ``denoising_steps``, ``contexts``, ``bootstrap``, ``r_threshold``, ``gt_max_abs_deg``, ``gt_pairs``,
    ``batch_size``, ``device``, ``deg_per_unit``.
``seed``; ``data``: ``dir``, ``split_json``, ``fullres_dir``, ``n_episodes``, ``contexts`` (list of
    ``{index, episode, start}``).
``models``: name -> ``{checkpoint: {path, sha256_16}, upsampler: {path, sha256_16}}``.
``scripts``: list of ``{name, kind, commanded_yaw_deg, commanded_pitch_deg}``; ``kind`` is ``yaw|pitch|key|noop``.
``estimator_validation``: ``threshold_r``, ``max_abs_deg``, ``passed`` (selected estimator, yaw, every resolution
    that has ground truth; null if none has), ``resolutions``: ``"150x280"|"30x56"`` -> ``{available, source,
    n_pairs, estimators: {phasecorr|flow: {yaw|pitch: {n, r, slope_deg_per_px, intercept_deg, n_all, r_all,
    theoretical_slope_deg_per_px, passed}}}}``; ``available`` false (with ``reason``) if there is no ground truth.
``conversion``: ``"fullres"|"lowres"`` -> ``{yaw_deg_per_px, pitch_deg_per_px, yaw_source, pitch_source}`` where
    the source is ``fitted`` or ``theory``; ``est_yaw = yaw_deg_per_px * -dx``, ``est_pitch = pitch_deg_per_px * -dy``.
``results``: model -> ``{"fullres"|"lowres": {estimator, yaw|pitch: {n, pearson_r, slope, intercept, sign_accuracy,
    n_nonzero}, per_bin: [{script, kind, commanded_yaw_deg, commanded_pitch_deg, n_contexts, yaw|pitch: {mean,
    ci_lo, ci_hi}}], noop_drift: {mean_abs_dyaw_deg|mean_abs_dpitch_deg: {mean, ci_lo, ci_hi}}}, ws_divergence:
    {w|s: {mean, ci_lo, ci_hi, sign_accuracy}}}``. ``slope`` is 1 for a perfectly calibrated model; NaN becomes null.
``warnings``; ``versions``; ``files``.

Other outputs in ``out_dir``: ``<model>_frames.csv`` (one row per generated frame pair: context, episode, start, script,
kind, frame, commanded and estimated values at both resolutions, divergence), ``<model>_scatter.{pdf,png}`` and
``per_bin_curves.{pdf,png}``.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from scipy import stats

from cs2wm.actions import ActionSpec, decode, encode
from cs2wm.eval import rotation
from cs2wm.pipeline.pack import read_episode, sha256_file
from cs2wm.play.agent import build_play_agent
from cs2wm.play.world_model import WorldModel, frames_to_tensor

SCHEMA_VERSION = 1
V1_MODEL_NAME = "v1"  # takes its upsampler from its own checkpoint
RESOLUTIONS = {"fullres": "150x280", "lowres": "30x56"}
FLOW_BORDER_FRAC = 0.1  # fraction of the shorter side excluded at each edge for the Farneback median

AgentFactory = Callable[[str, Path], Any]
Estimator = Callable[[np.ndarray, np.ndarray], tuple[float, float]]


# --- scripted suite ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScriptedSuite:
    """Scripted input suite.

    Attributes:
        yaw_units: Yaw bins, each run as ``+u`` and ``-u`` (Pearce & Zhu units; must be mouse-x bin centres).
        pitch_units: Pitch bins, each run as ``+u`` and ``-u`` (must be mouse-y bin centres).
        keys: Movement keys to test, one script each.
        rollout_frames: Frames each script is held for.
        noop: Whether to add the no-op script.
    """

    yaw_units: tuple[int, ...] = (2, 4, 10, 20, 30, 60)
    pitch_units: tuple[int, ...] = (2, 4, 10, 20)
    keys: tuple[str, ...] = ("w", "s", "a", "d")
    rollout_frames: int = 32
    noop: bool = True

    @classmethod
    def from_config(cls, cfg: DictConfig) -> ScriptedSuite:
        """Build the suite from ``eval.h1``."""
        h = cfg.eval.h1
        return cls(
            yaw_units=tuple(int(u) for u in h.yaw_units),
            pitch_units=tuple(int(u) for u in h.pitch_units),
            keys=tuple(h["keys"]),
            rollout_frames=int(h.rollout_frames),
        )

    def scripts(self, spec: ActionSpec) -> list[Script]:
        """Build the scripts: yaw, pitch, keys, no-op, in that order.

        Args:
            spec: Action layout (from the config's bins).

        Returns:
            The scripts. ``commanded_*_deg`` are read back from the encoded action (the bin centre), not assumed.

        Raises:
            ValueError: If a scripted unit is not a configured bin centre.
        """
        out: list[Script] = []
        for kind, units, centres in (
            ("yaw", self.yaw_units, spec.mouse_x_units),
            ("pitch", self.pitch_units, spec.mouse_y_units),
        ):
            for u in units:
                if float(u) not in centres or -float(u) not in centres:
                    raise ValueError(f"scripted {kind} unit {u} is not a bin centre of {list(centres)}")
                for signed in (u, -u):
                    deg = signed * spec.deg_per_unit
                    out.append(
                        Script.build(
                            f"{kind}{signed:+d}",
                            kind,
                            [],
                            deg if kind == "yaw" else 0.0,
                            deg if kind == "pitch" else 0.0,
                            spec,
                        )
                    )
        for key in self.keys:
            out.append(Script.build(f"key_{key}", "key", [key], 0.0, 0.0, spec))
        if self.noop:
            out.append(Script.build("noop", "noop", [], 0.0, 0.0, spec))
        return out


@dataclass(frozen=True, eq=False)
class Script:
    """One scripted action held for the whole rollout.

    Attributes:
        name: ``yaw+10``, ``pitch-4``, ``key_w``, ``noop``.
        kind: ``yaw``, ``pitch``, ``key`` or ``noop``.
        action: ``float32[51]`` action vector.
        commanded_yaw_deg: Commanded d-yaw per frame in degrees (decoded bin centre, +right).
        commanded_pitch_deg: Commanded d-pitch per frame in degrees (decoded bin centre, +down).
    """

    name: str
    kind: str
    action: np.ndarray
    commanded_yaw_deg: float
    commanded_pitch_deg: float

    @classmethod
    def build(
        cls, name: str, kind: str, keys: Sequence[str], yaw_deg: float, pitch_deg: float, spec: ActionSpec
    ) -> Script:
        """Encode the action and read the commanded rotation back from it (the bin centre, not the request)."""
        action = encode(keys, False, False, yaw_deg, pitch_deg, spec)
        d = decode(action, spec)
        return cls(name, kind, action, d.mouse_x_deg, d.mouse_y_deg)


# --- held-out data -------------------------------------------------------------------------------------------------


def _demo_id(path: Path) -> str:
    with h5py.File(path, "r") as f:
        value = f.attrs.get("demo_id")
    if value is None:
        return path.stem
    return value.decode() if isinstance(value, bytes) else str(value)


def list_episodes(data_dir: Path, split_json: Path | None = None) -> list[Path]:
    """List the packed episodes to evaluate on, sorted.

    Args:
        data_dir: Directory of packed ``.hdf5`` episodes.
        split_json: Optional ``split.json`` (``scripts/hdf5_to_diamond.py``): a mapping ``demo_id -> "train"|"test"``
            (or ``{"test": [demo ids], ...}``). Only episodes of test demos are kept; the episode's ``demo_id`` attr is
            used (its file stem if the attr is missing).

    Returns:
        The episode paths.

    Raises:
        FileNotFoundError: If there are no episodes.
        ValueError: If the split leaves no test episodes.
    """
    files = sorted(Path(data_dir).glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"{data_dir}: no .hdf5 episodes found")
    if split_json is not None:
        split = json.loads(Path(split_json).read_text())
        if all(isinstance(v, list) for v in split.values()):
            test_demos = set(split.get("test", []))
        else:
            test_demos = {demo for demo, name in split.items() if name == "test"}
        files = [p for p in files if _demo_id(p) in test_demos]
        if not files:
            raise ValueError(f"{split_json}: no episode in {data_dir} belongs to a test demo")
    return files


@dataclass(frozen=True, eq=False)
class Context:
    """A held-out context: ``n_cond`` ground-truth frames and the actions taken at them.

    Attributes:
        index: Position in the sampled list.
        episode: Episode file stem.
        start: Index of the first frame in the episode.
        frames: ``float32[n, 3, h, w]`` RGB in [-1, 1].
        actions: ``float32[n, 51]``.
    """

    index: int
    episode: str
    start: int
    frames: torch.Tensor
    actions: np.ndarray


def sample_contexts(files: Sequence[Path], n: int, n_cond: int, seed: int) -> list[Context]:
    """Draw ``n`` contexts deterministically, spread over the episodes.

    Episodes with at least ``n_cond`` frames are visited in a seeded random order (cycling if ``n`` exceeds their
    number), each contributing a window starting at a seeded random frame.

    Args:
        files: Packed episodes.
        n: Contexts wanted.
        n_cond: Frames per context (the denoiser's ``num_steps_conditioning``).
        seed: RNG seed.

    Returns:
        ``n`` contexts.

    Raises:
        ValueError: If ``n < 1`` or no episode has ``n_cond`` frames.
    """
    if n < 1:
        raise ValueError(f"contexts must be at least 1, got {n}")
    lengths = []
    for p in files:
        with h5py.File(p, "r") as f:
            lengths.append(int(f.attrs["n_frames"]))
    eligible = [i for i, m in enumerate(lengths) if m >= n_cond]
    if not eligible:
        raise ValueError(f"no episode has at least {n_cond} frames")
    rng = np.random.default_rng(seed)
    order: list[int] = []
    while len(order) < n:
        order += [int(i) for i in rng.permutation(eligible)]
    picks = [(i, int(rng.integers(0, lengths[i] - n_cond + 1))) for i in order[:n]]
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    contexts = []
    for k, (i, start) in enumerate(picks):
        if i not in cache:
            frames, actions, _ = read_episode(files[i])
            cache = {i: (frames, actions)}  # picks revisit an episode only after a full cycle: keep one
        frames, actions = cache[i]
        window = slice(start, start + n_cond)
        contexts.append(Context(k, files[i].stem, start, frames_to_tensor(frames[window]), actions[window]))
    return contexts


# --- estimators and ground-truth validation ------------------------------------------------------------------------


def phasecorr_estimator(prev: np.ndarray, curr: np.ndarray) -> tuple[float, float]:
    """Phase-correlation ``(dx, dy)``."""
    s = rotation.estimate_shift_phasecorr(prev, curr)
    return s.dx, s.dy


def flow_estimator(prev: np.ndarray, curr: np.ndarray) -> tuple[float, float]:
    """Farneback median-flow ``(dx, dy)`` with ``FLOW_BORDER_FRAC`` of the image excluded at each edge."""
    s = rotation.estimate_shift_flow(prev, curr, border=int(FLOW_BORDER_FRAC * min(prev.shape[:2])))
    return s.dx, s.dy


ESTIMATORS: dict[str, Estimator] = {"phasecorr": phasecorr_estimator, "flow": flow_estimator}


def collect_gt_shifts(files: Sequence[Path], n_pairs: int, seed: int, spec: ActionSpec) -> dict[str, Any]:
    """Estimate the shift of consecutive ground-truth frames and pair it with the label.

    Pairs ``(k, k + 1)`` are drawn at random, an equal share from every episode; the label is the decoded bin centre
    of action ``k`` (action ``k`` covers frames ``k -> k + 1``).

    Args:
        files: Packed episodes (any frame resolution).
        n_pairs: Pairs wanted (fewer if the episodes are short).
        seed: RNG seed.
        spec: Action layout, to decode the labels.

    Returns:
        ``{"label_yaw", "label_pitch", "shifts": {estimator: (dx, dy)}, "hw"}`` with float64 arrays of length ``n``.

    Raises:
        ValueError: If there is no episode with two frames.
    """
    rng = np.random.default_rng(seed)
    per_file = -(-n_pairs // max(len(files), 1))
    label_yaw, label_pitch = [], []
    shifts: dict[str, list[tuple[float, float]]] = {name: [] for name in ESTIMATORS}
    hw: tuple[int, int] | None = None
    for path in files:
        frames, actions, _ = read_episode(path)
        if len(frames) < 2 or len(label_yaw) >= n_pairs:
            continue
        hw = frames.shape[1:3]
        ks = rng.choice(len(frames) - 1, size=min(per_file, len(frames) - 1, n_pairs - len(label_yaw)), replace=False)
        for k in sorted(int(k) for k in ks):
            d = decode(actions[k], spec)
            label_yaw.append(d.mouse_x_deg)
            label_pitch.append(d.mouse_y_deg)
            for name, fn in ESTIMATORS.items():
                shifts[name].append(fn(frames[k], frames[k + 1]))
    if hw is None:
        raise ValueError("no episode has at least two frames")
    return {
        "label_yaw": np.array(label_yaw),
        "label_pitch": np.array(label_pitch),
        "shifts": {name: np.array(v).reshape(-1, 2).T for name, v in shifts.items()},
        "hw": (int(hw[0]), int(hw[1])),
    }


def fit_axis(label_deg: np.ndarray, px: np.ndarray, max_abs_deg: float) -> dict[str, Any]:
    """Pearson r and least-squares slope of the label (degrees) on the rotation-signed pixel shift.

    Args:
        label_deg: Label-implied rotation per frame in degrees.
        px: Pixel shift signed so that a positive rotation is positive (``-dx`` for yaw, ``-dy`` for pitch).
        max_abs_deg: ``n``, ``r``, ``slope_deg_per_px`` and ``intercept_deg`` use pairs with ``|label| <= max_abs_deg``
            (the range the scripts cover); ``n_all`` and ``r_all`` use all pairs.

    Returns:
        ``{n, r, slope_deg_per_px, intercept_deg, n_all, r_all}``; NaN where undefined (constant input).
    """
    keep = np.abs(label_deg) <= max_abs_deg
    out: dict[str, Any] = {"n": int(keep.sum()), "n_all": len(label_deg)}
    nan = float("nan")
    x, y = px[keep], label_deg[keep]
    if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        fit = stats.linregress(x, y)
        out.update(r=float(fit.rvalue), slope_deg_per_px=float(fit.slope), intercept_deg=float(fit.intercept))
    else:
        out.update(r=nan, slope_deg_per_px=nan, intercept_deg=nan)
    constant = len(px) < 3 or np.ptp(px) == 0 or np.ptp(label_deg) == 0
    out["r_all"] = nan if constant else float(stats.pearsonr(px, label_deg)[0])
    return out


def validate_estimator(
    low_files: Sequence[Path],
    full_files: Sequence[Path] | None,
    cfg: DictConfig,
    seed: int,
) -> dict[str, Any]:
    """Validate the shift estimators against the labels of ground-truth episodes, at both resolutions.

    Args:
        low_files: Packed 30x56 episodes (``--data``).
        full_files: Full-res 150x280 twins (``--fullres-data``), or None to skip that resolution.
        cfg: A validated cs2wm config.
        seed: RNG seed.

    Returns:
        The ``estimator_validation`` section of the summary (see the module docstring).
    """
    h = cfg.eval.h1
    spec = ActionSpec.from_config(cfg)
    capture = tuple(int(v) for v in cfg.resolution.capture)
    threshold = float(h.r_threshold)
    resolutions: dict[str, Any] = {}
    for key, files in (("fullres", full_files), ("lowres", low_files)):
        name = RESOLUTIONS[key]
        if not files:
            resolutions[name] = {
                "available": False,
                "reason": "no ground-truth episodes at this resolution (--fullres-data)",
            }
            continue
        gt = collect_gt_shifts(files, int(h.gt_pairs), seed, spec)
        height, width = gt["hw"]
        theory = rotation.theoretical_slopes(width, height, float(h.hfov_deg), capture)
        estimators: dict[str, Any] = {}
        for est, (dx, dy) in gt["shifts"].items():
            axes = {}
            for axis, label, px, th in (
                ("yaw", gt["label_yaw"], -dx, theory[0]),
                ("pitch", gt["label_pitch"], -dy, theory[1]),
            ):
                fit = fit_axis(label, px, float(h.gt_max_abs_deg))
                fit["theoretical_slope_deg_per_px"] = th
                fit["passed"] = bool(fit["r"] >= threshold)
                axes[axis] = fit
            estimators[est] = axes
        resolutions[name] = {
            "available": True,
            "source": {"dir": str(Path(files[0]).parent), "n_episodes": len(files)},
            "n_pairs": len(gt["label_yaw"]),
            "estimators": estimators,
        }
    selected = str(h.estimator)
    gated = [r["estimators"][selected]["yaw"]["passed"] for r in resolutions.values() if r["available"]]
    return {
        "threshold_r": threshold,
        "max_abs_deg": float(h.gt_max_abs_deg),
        "estimator": selected,
        "passed": bool(all(gated)) if gated else None,
        "resolutions": resolutions,
    }


def build_conversion(validation: dict[str, Any], cfg: DictConfig) -> tuple[dict[str, Any], list[str]]:
    """Pick the pixel-to-degree conversion for the rollouts: the ground-truth fit where there is one.

    Args:
        validation: Output of ``validate_estimator``.
        cfg: A validated cs2wm config.

    Returns:
        ``(conversion, warnings)``; the conversion maps ``fullres``/``lowres`` to degrees per pixel per axis and the
        source (``fitted`` or ``theory``). A missing, non-finite or non-positive fit falls back to theory, with a
        warning.
    """
    h = cfg.eval.h1
    capture = tuple(int(v) for v in cfg.resolution.capture)
    sizes = {"fullres": tuple(cfg.resolution.render), "lowres": tuple(cfg.resolution.low_res)}
    conversion: dict[str, Any] = {}
    warnings: list[str] = []
    for key, name in RESOLUTIONS.items():
        theory = rotation.theoretical_slopes(sizes[key][1], sizes[key][0], float(h.hfov_deg), capture)
        res = validation["resolutions"][name]
        entry: dict[str, Any] = {}
        for i, axis in enumerate(("yaw", "pitch")):
            slope, source = theory[i], "theory"
            if res["available"]:
                fitted = res["estimators"][validation["estimator"]][axis]["slope_deg_per_px"]
                if np.isfinite(fitted) and fitted > 0:
                    slope, source = float(fitted), "fitted"
                else:
                    warnings.append(f"{name} {axis}: ground-truth fit unusable ({fitted}); using the theoretical slope")
            entry[f"{axis}_deg_per_px"] = slope
            entry[f"{axis}_source"] = source
        conversion[key] = entry
    return conversion, warnings


# --- rollouts ------------------------------------------------------------------------------------------------------


def job_generator(seed: int, context: int, script: str) -> torch.Generator:
    """The noise generator of one (context, script) rollout: the same for every model, different per run seed."""
    state = np.random.SeedSequence([seed, context, zlib.crc32(script.encode())]).generate_state(2, dtype=np.uint32)
    return torch.Generator().manual_seed(int(state[0]) << 32 | int(state[1]))


def _bgr_uint8(x: torch.Tensor) -> np.ndarray:
    """``float[t, 3, H, W]`` RGB in [-1, 1] to ``uint8[t, H, W, 3]`` BGR (clamped before the cast)."""
    rgb = x.add(1).div(2).mul(255).round().clamp(0, 255).byte()
    return np.ascontiguousarray(rgb.flip(1).permute(0, 2, 3, 1).numpy())


def rollout_shifts(
    world: WorldModel,
    contexts: Sequence[Context],
    scripts: Sequence[Script],
    rollout_frames: int,
    seed: int,
    batch_size: int,
    estimator: Estimator,
    progress: Callable[[int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Roll out every (context, script) pair and estimate the shift between consecutive generated frames.

    Args:
        world: The model's world model.
        contexts: Held-out contexts.
        scripts: Scripted suite.
        rollout_frames: Frames per rollout.
        seed: Run seed (noise is seeded by it, the context index and the script name).
        batch_size: Rollouts per batched forward pass.
        estimator: Shift estimator.
        progress: Called with the number of rollouts finished after each batch.

    Returns:
        Arrays of length ``len(contexts) * len(scripts) * (rollout_frames - 1)``: ``context``, ``script``, ``frame``
        (1..T-1, the later frame of the pair), ``dx``, ``dy`` (150x280), ``dx_lo``, ``dy_lo`` (30x56) and
        ``divergence`` (Farneback, full-res, W/S scripts only, else NaN).
    """
    jobs = [(ci, si) for si in range(len(scripts)) for ci in range(len(contexts))]
    cols: dict[str, list] = {k: [] for k in ("context", "script", "frame", "dx", "dy", "dx_lo", "dy_lo", "divergence")}
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        frames = torch.stack([contexts[ci].frames for ci, _ in batch])
        ctx_actions = torch.from_numpy(np.stack([contexts[ci].actions for ci, _ in batch]))
        actions = torch.from_numpy(np.stack([np.tile(scripts[si].action, (rollout_frames, 1)) for _, si in batch]))
        gens = [job_generator(seed, ci, scripts[si].name) for ci, si in batch]
        low, full = world.rollout(frames, ctx_actions, actions, gens)
        for b, (ci, si) in enumerate(batch):
            hi, lo = _bgr_uint8(full[b]), _bgr_uint8(low[b])
            forward_backward = scripts[si].name in ("key_w", "key_s")
            for t in range(1, rollout_frames):
                dx, dy = estimator(hi[t - 1], hi[t])
                dx_lo, dy_lo = estimator(lo[t - 1], lo[t])
                div = np.nan
                if forward_backward:
                    div = rotation.flow_divergence(hi[t - 1], hi[t], border=int(FLOW_BORDER_FRAC * min(hi.shape[1:3])))
                for k, v in zip(cols, (ci, si, t, dx, dy, dx_lo, dy_lo, div)):
                    cols[k].append(v)
        if progress is not None:
            progress(len(batch))
    out = {
        k: np.asarray(v, dtype=np.float64 if k not in ("context", "script", "frame") else np.int64)
        for k, v in cols.items()
    }
    return out


# --- statistics ----------------------------------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap confidence interval of the mean.

    Args:
        values: 1-D sample (NaNs are dropped).
        n_boot: Resamples.
        seed: RNG seed; the result is a deterministic function of ``(values, n_boot, seed)``.
        level: Confidence level.

    Returns:
        ``(mean, lo, hi)``; all NaN for an empty sample.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(v.mean()), float(lo), float(hi)


def axis_metrics(commanded: np.ndarray, estimated: np.ndarray) -> dict[str, Any]:
    """Pearson r, regression slope and sign accuracy of estimated vs commanded rotation.

    Args:
        commanded: Commanded rotation per frame, degrees.
        estimated: Estimated rotation per frame, degrees.

    Returns:
        ``{n, pearson_r, slope, intercept, sign_accuracy, n_nonzero}``. r/slope are NaN if either input is constant;
        the sign accuracy (share of non-zero commands whose estimate has the same sign; an estimate of exactly 0
        counts as wrong) is NaN if every command is zero.
    """
    c, e = np.asarray(commanded, dtype=np.float64), np.asarray(estimated, dtype=np.float64)
    out: dict[str, Any] = {"n": len(c)}
    if len(c) >= 3 and np.ptp(c) > 0 and np.ptp(e) > 0:
        fit = stats.linregress(c, e)
        out.update(pearson_r=float(fit.rvalue), slope=float(fit.slope), intercept=float(fit.intercept))
    else:
        out.update(pearson_r=float("nan"), slope=float("nan"), intercept=float("nan"))
    nz = c != 0
    out["n_nonzero"] = int(nz.sum())
    out["sign_accuracy"] = float((np.sign(e[nz]) == np.sign(c[nz])).mean()) if nz.any() else float("nan")
    return out


def _ci_dict(values: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    mean, lo, hi = bootstrap_ci(values, n_boot, seed)
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi}


def model_metrics(
    rec: dict[str, np.ndarray],
    est: dict[str, np.ndarray],
    scripts: Sequence[Script],
    n_contexts: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Controllability metrics of one model at one resolution.

    Args:
        rec: Output of ``rollout_shifts`` (for ``context`` and ``script``).
        est: ``{"yaw", "pitch"}`` estimated rotation in degrees per row of ``rec``.
        scripts: The scripted suite.
        n_contexts: Number of contexts.
        n_boot: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        ``{yaw, pitch, per_bin, noop_drift}`` (see the module docstring); scripts are pooled for r/slope/sign.
    """
    cmd = {
        "yaw": np.array([s.commanded_yaw_deg for s in scripts])[rec["script"]],
        "pitch": np.array([s.commanded_pitch_deg for s in scripts])[rec["script"]],
    }
    out: dict[str, Any] = {axis: axis_metrics(cmd[axis], est[axis]) for axis in ("yaw", "pitch")}
    per_bin = []
    for si, script in enumerate(scripts):
        sel = rec["script"] == si
        entry: dict[str, Any] = {
            "script": script.name,
            "kind": script.kind,
            "commanded_yaw_deg": script.commanded_yaw_deg,
            "commanded_pitch_deg": script.commanded_pitch_deg,
            "n_contexts": n_contexts,
        }
        for axis in ("yaw", "pitch"):
            per_ctx = _per_context_mean(rec["context"][sel], est[axis][sel], n_contexts)
            entry[axis] = _ci_dict(per_ctx, n_boot, seed + 1000 * si + (axis == "pitch"))
        per_bin.append(entry)
    out["per_bin"] = per_bin
    noop = [i for i, s in enumerate(scripts) if s.kind == "noop"]
    if noop:
        sel = rec["script"] == noop[0]
        out["noop_drift"] = {
            f"mean_abs_d{axis}_deg": _ci_dict(
                _per_context_mean(rec["context"][sel], np.abs(est[axis][sel]), n_contexts), n_boot, seed + 7 + j
            )
            for j, axis in enumerate(("yaw", "pitch"))
        }
    return out


def _per_context_mean(context: np.ndarray, values: np.ndarray, n_contexts: int) -> np.ndarray:
    """Mean of ``values`` per context index (NaN for contexts without rows)."""
    sums = np.bincount(context, weights=values, minlength=n_contexts)
    counts = np.bincount(context, minlength=n_contexts)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts


def divergence_metrics(
    rec: dict[str, np.ndarray], scripts: Sequence[Script], n_contexts: int, n_boot: int, seed: int
) -> dict:
    """Flow-divergence check for W (expect > 0) and S (expect < 0).

    Returns:
        ``{w|s: {mean, ci_lo, ci_hi, sign_accuracy}}`` over the per-context mean divergence (1/frame); the sign
        accuracy is over frames. Keys absent from the suite are omitted.
    """
    out: dict[str, Any] = {}
    for key, expected in (("w", 1.0), ("s", -1.0)):
        idx = [i for i, s in enumerate(scripts) if s.name == f"key_{key}"]
        if not idx:
            continue
        sel = rec["script"] == idx[0]
        d = rec["divergence"][sel]
        entry = _ci_dict(_per_context_mean(rec["context"][sel], d, n_contexts), n_boot, seed + 31 + idx[0])
        entry["sign_accuracy"] = float((np.sign(d) == expected).mean())
        out[key] = entry
    return out


# --- outputs -------------------------------------------------------------------------------------------------------


def _json_clean(obj: Any) -> Any:
    """Make an object JSON-safe: NaN/inf become null, numpy scalars Python numbers, tuples lists."""
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_frames_csv(
    path: Path,
    rec: dict[str, np.ndarray],
    est: dict[str, dict[str, np.ndarray]],
    scripts: Sequence[Script],
    contexts: Sequence[Context],
) -> None:
    """Write one row per generated frame pair.

    Columns: ``context, episode, start, script, kind, frame, cmd_yaw_deg, cmd_pitch_deg, dx_px, dy_px, est_yaw_deg,
    est_pitch_deg, dx_px_lowres, dy_px_lowres, est_yaw_deg_lowres, est_pitch_deg_lowres, divergence`` (``frame`` is
    the later frame of the pair; empty cells are not computed).
    """

    def num(x: float) -> str:
        return "" if not np.isfinite(x) else repr(float(x))

    pixel_cols = {
        "dx_px": "dx",
        "dy_px": "dy",
        "dx_px_lowres": "dx_lo",
        "dy_px_lowres": "dy_lo",
        "divergence": "divergence",
    }
    est_cols = {
        "est_yaw_deg": ("fullres", "yaw"),
        "est_pitch_deg": ("fullres", "pitch"),
        "est_yaw_deg_lowres": ("lowres", "yaw"),
        "est_pitch_deg_lowres": ("lowres", "pitch"),
    }
    header = ["context", "episode", "start", "script", "kind", "frame", "cmd_yaw_deg", "cmd_pitch_deg"]
    header += [*pixel_cols, *est_cols]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(rec["frame"])):
            c, s = contexts[rec["context"][i]], scripts[rec["script"][i]]
            row = [
                c.index,
                c.episode,
                c.start,
                s.name,
                s.kind,
                rec["frame"][i],
                s.commanded_yaw_deg,
                s.commanded_pitch_deg,
            ]
            row += [num(rec[k][i]) for k in pixel_cols.values()]
            row += [num(est[res][axis][i]) for res, axis in est_cols.values()]
            w.writerow(row)


def _save_fig(fig: Any, out_dir: Path, stem: str) -> list[str]:
    paths = []
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        paths.append(f"{stem}.{ext}")
    return paths


def plot_scatter(
    model: str, rec: dict, est: dict, scripts: Sequence[Script], metrics: dict, out_dir: Path
) -> list[str]:
    """Commanded vs estimated rotation per frame (150x280), yaw and pitch panels, with the identity line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    rng = np.random.default_rng(0)
    for ax, axis in zip(axes, ("yaw", "pitch")):
        cmd = np.array([getattr(s, f"commanded_{axis}_deg") for s in scripts])[rec["script"]]
        e = est["fullres"][axis]
        ax.scatter(
            cmd + rng.normal(0, 0.01, len(cmd)), e, s=3, alpha=0.15, color="#4477aa", linewidths=0, rasterized=True
        )
        lim = max(0.5, float(np.nanmax(np.abs(np.r_[cmd, e]))) * 1.05) if len(cmd) else 1.0
        ax.plot([-lim, lim], [-lim, lim], color="0.4", lw=1, ls="--", label="ideal")
        m = metrics[axis]
        ax.set_title(f"{model}: {axis}  (r = {m['pearson_r']:.2f}, slope = {m['slope']:.2f})", fontsize=10)
        ax.set_xlabel(f"commanded d-{axis} (deg/frame)")
        ax.set_ylabel(f"estimated d-{axis} (deg/frame)")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths = _save_fig(fig, out_dir, f"{model}_scatter")
    plt.close(fig)
    return paths


def plot_per_bin(results: dict[str, Any], out_dir: Path) -> list[str]:
    """Per-bin mean estimated rotation with 95% bootstrap CI against the commanded value, all models (150x280)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    lim = {"yaw": 0.1, "pitch": 0.1}
    for ax, axis in zip(axes, ("yaw", "pitch")):
        for color, (model, res) in zip(colors, results.items()):
            # Scripts of this axis plus no-op (commanded 0), sorted by commanded value.
            rows = [b for b in res["fullres"]["per_bin"] if b["kind"] in (axis, "noop") and b[axis]["mean"] is not None]
            rows.sort(key=lambda b: b[f"commanded_{axis}_deg"])
            x = np.array([b[f"commanded_{axis}_deg"] for b in rows])
            y = np.array([b[axis]["mean"] for b in rows])
            lo = np.array([b[axis]["ci_lo"] for b in rows])
            hi = np.array([b[axis]["ci_hi"] for b in rows])
            ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", ms=3, lw=1, capsize=2, color=color, label=model)
            if len(x):
                lim[axis] = max(lim[axis], float(np.abs(x).max()), float(np.abs(np.r_[lo, hi]).max()))
        lim[axis] *= 1.1
        ax.plot([-lim[axis], lim[axis]], [-lim[axis], lim[axis]], color="0.4", lw=1, ls="--", label="ideal")
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_xlabel(f"commanded d-{axis} (deg/frame)")
        ax.set_ylabel(f"estimated d-{axis} (deg/frame)")
        ax.set_title(f"per-bin mean, 95% bootstrap CI: {axis}", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths = _save_fig(fig, out_dir, "per_bin_curves")
    plt.close(fig)
    return paths


# --- driver --------------------------------------------------------------------------------------------------------


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> dict[str, Any]:
    import matplotlib
    import scipy

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "git_sha": _git_sha(),
    }


def make_agent_factory(upsampler_ckpt: Path | None, device: torch.device) -> AgentFactory:
    """The agent loader for the real run: ``build_play_agent`` with the right upsampler.

    The model named ``v1`` uses its own checkpoint for the upsampler; every other model uses ``upsampler_ckpt``.

    Args:
        upsampler_ckpt: The v1 checkpoint providing the upsampler for v2 and shuffled.
        device: Device for the agents.

    Returns:
        ``factory(name, ckpt) -> Agent``.
    """

    def factory(name: str, ckpt: Path) -> Any:
        up = ckpt if name == V1_MODEL_NAME else upsampler_ckpt
        if up is None:
            raise ValueError(f"model {name!r} needs --upsampler-ckpt (the v1 checkpoint)")
        return build_play_agent(ckpt, up, device)

    return factory


def _estimated_degrees(rec: dict[str, np.ndarray], conversion: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Convert pixel shifts to degrees with the chosen slopes (``est = slope * -shift``)."""
    full, low = conversion["fullres"], conversion["lowres"]
    return {
        "fullres": {"yaw": -full["yaw_deg_per_px"] * rec["dx"], "pitch": -full["pitch_deg_per_px"] * rec["dy"]},
        "lowres": {"yaw": -low["yaw_deg_per_px"] * rec["dx_lo"], "pitch": -low["pitch_deg_per_px"] * rec["dy_lo"]},
    }


def run_h1(
    models: dict[str, Path],
    cfg: DictConfig,
    out_dir: Path,
    contexts: int = 50,
    seed: int = 0,
    data_dir: Path | None = None,
    split_json: Path | None = None,
    fullres_dir: Path | None = None,
    upsampler_ckpt: Path | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 16,
    denoising_steps: int = 1,
    suite: ScriptedSuite | None = None,
    agent_factory: AgentFactory | None = None,
) -> dict:
    """Roll out every scripted input for every model and compute the controllability metrics.

    Args:
        models: Mapping from name (``v2``, ``v1``, ``shuffled``) to checkpoint path.
        cfg: A validated config.
        out_dir: Output directory (``results/h1``).
        contexts: Number of held-out contexts.
        seed: RNG seed (context sampling, ground-truth pairs, rollout noise, bootstrap).
        data_dir: Directory of held-out packed 30x56 episodes.
        split_json: Optional ``split.json`` keeping only test demos.
        fullres_dir: Optional directory of 150x280 full-res twins (``hdf5_fullres``) for the ground-truth validation
            at full resolution; without it the full-res conversion is the theoretical one.
        upsampler_ckpt: v1 checkpoint providing the upsampler for non-``v1`` models (default: the ``v1`` model's).
        device: Torch device; defaults to cuda if available.
        batch_size: Rollouts per batched forward pass.
        denoising_steps: Sampler preset, 1 or 3.
        suite: Scripted suite; defaults to ``ScriptedSuite.from_config(cfg)``.
        agent_factory: ``(name, ckpt) -> agent``; defaults to ``make_agent_factory``.

    Returns:
        The summary dict that is written to ``summary.json``.

    Raises:
        ValueError: If ``data_dir`` is missing, ``models`` is empty or the suite is invalid.
    """
    if data_dir is None:
        raise ValueError("data_dir (--data) is required")
    if not models:
        raise ValueError("no models given")
    h = cfg.eval.h1
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = ActionSpec.from_config(cfg)
    suite = suite or ScriptedSuite.from_config(cfg)
    scripts = suite.scripts(spec)
    if suite.rollout_frames < 2:
        raise ValueError("rollout_frames must be at least 2 (one frame pair)")
    n_boot = int(h.bootstrap)
    estimator = ESTIMATORS[str(h.estimator)]
    if upsampler_ckpt is None and V1_MODEL_NAME in models:
        upsampler_ckpt = models[V1_MODEL_NAME]
    factory = agent_factory or make_agent_factory(upsampler_ckpt, dev)

    files = list_episodes(data_dir, split_json)
    full_files = list_episodes(fullres_dir, split_json) if fullres_dir is not None else None

    validation = validate_estimator(files, full_files, cfg, seed)
    conversion, warnings = build_conversion(validation, cfg)
    for name, res in validation["resolutions"].items():
        if not res["available"]:
            warnings.append(f"estimator not validated at {name}: {res['reason']}")
            continue
        fit = res["estimators"][validation["estimator"]]["yaw"]
        if not fit["passed"]:
            warnings.append(
                f"ESTIMATOR VALIDATION FAILED at {name}: {validation['estimator']} yaw r = {fit['r']:.3f} < "
                f"{validation['threshold_r']} on ground truth; switch estimator or resolution"
            )
    ctxs = sample_contexts(files, contexts, 4, seed)

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "hfov_deg": float(h.hfov_deg),
            "capture_hw": list(cfg.resolution.capture),
            "frame_hw": list(cfg.resolution.render),
            "low_res_hw": list(cfg.resolution.low_res),
            "deg_per_unit": spec.deg_per_unit,
            "rollout_frames": suite.rollout_frames,
            "estimator": str(h.estimator),
            "denoising_steps": denoising_steps,
            "contexts": contexts,
            "bootstrap": n_boot,
            "r_threshold": float(h.r_threshold),
            "gt_max_abs_deg": float(h.gt_max_abs_deg),
            "gt_pairs": int(h.gt_pairs),
            "batch_size": batch_size,
            "device": str(dev),
        },
        "seed": seed,
        "data": {
            "dir": str(data_dir),
            "split_json": str(split_json) if split_json else None,
            "fullres_dir": str(fullres_dir) if fullres_dir else None,
            "n_episodes": len(files),
            "contexts": [{"index": c.index, "episode": c.episode, "start": c.start} for c in ctxs],
        },
        "models": {},
        "scripts": [
            {
                "name": s.name,
                "kind": s.kind,
                "commanded_yaw_deg": s.commanded_yaw_deg,
                "commanded_pitch_deg": s.commanded_pitch_deg,
            }
            for s in scripts
        ],
        "estimator_validation": validation,
        "conversion": conversion,
        "results": {},
    }
    plots: list[str] = []
    for name, ckpt in models.items():
        print(
            f"[h1] rolling out {name} ({len(ctxs)} contexts x {len(scripts)} scripts x {suite.rollout_frames} frames)"
        )
        agent = factory(name, Path(ckpt))
        world = WorldModel(agent, denoising_steps, use_upsampler=True)
        up = Path(ckpt) if name == V1_MODEL_NAME else upsampler_ckpt
        summary["models"][name] = {
            "checkpoint": {"path": str(ckpt), "sha256_16": _sha16(Path(ckpt))},
            "upsampler": {"path": str(up), "sha256_16": _sha16(up) if up else None},
        }
        done = 0

        def progress(n: int, name: str = name) -> None:
            nonlocal done
            done += n
            print(f"[h1] {name}: {done}/{len(ctxs) * len(scripts)} rollouts", end="\r", flush=True)

        rec = rollout_shifts(world, ctxs, scripts, suite.rollout_frames, seed, batch_size, estimator, progress)
        print()
        del world, agent
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        est = _estimated_degrees(rec, conversion)
        res = {
            key: {"estimator": str(h.estimator), **model_metrics(rec, est[key], scripts, len(ctxs), n_boot, seed)}
            for key in RESOLUTIONS
        }
        res["ws_divergence"] = divergence_metrics(rec, scripts, len(ctxs), n_boot, seed)
        summary["results"][name] = res
        write_frames_csv(out_dir / f"{name}_frames.csv", rec, est, scripts, ctxs)
        plots += plot_scatter(name, rec, est, scripts, res["fullres"], out_dir)
    plots += plot_per_bin(summary["results"], out_dir)

    summary["warnings"] = warnings
    summary["versions"] = _versions()
    summary["files"] = [f"{m}_frames.csv" for m in models] + plots + ["summary.json"]
    (out_dir / "summary.json").write_text(json.dumps(_json_clean(summary), indent=2) + "\n")
    for w in warnings:
        print(f"[h1] WARNING: {w}", file=sys.stderr)
    print(format_summary(summary))
    return summary


def _sha16(path: Path | None) -> str | None:
    return sha256_file(path)[:16] if path is not None and Path(path).exists() else None


def format_summary(summary: dict[str, Any]) -> str:
    """Printable per-model table of r, slope and sign accuracy at both resolutions."""
    lines = [f"{'model':<10}{'res':<9}{'axis':<7}{'r':>8}{'slope':>8}{'sign':>8}"]
    for model, res in summary["results"].items():
        for key, label in RESOLUTIONS.items():
            for axis in ("yaw", "pitch"):
                m = res[key][axis]
                lines.append(
                    f"{model:<10}{label:<9}{axis:<7}{m['pearson_r']:>8.3f}{m['slope']:>8.3f}{m['sign_accuracy']:>8.3f}"
                )
    return "\n".join(lines)
