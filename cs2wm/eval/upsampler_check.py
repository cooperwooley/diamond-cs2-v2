"""v1 upsampler compatibility check on v2 low-res frames (#27).

v2 reuses the v1 upsampler unchanged, so check that it still works on v2 frames (HLAE capture, no HUD) against a
bicubic baseline. v1 frames came from YouTube video with a HUD.

Method
------
1. **Sampling frames.** For each dataset (v2: the ``--fullres`` directory of pack's JPEG full-res episodes; v1: the
   optional ``--v1-fullres`` directory of v1 test full-res HDF5 files) ``n`` frame indices ``t >= 1`` are drawn
   deterministically from ``--seed``, round-robin over the episodes so they are spread across them.
2. **Inputs.** The ground truth is full-res frame ``t`` (150x280). The low-res input is made from it with
   ``pack.to_low_res``, the exact v1 recipe plus v2's clamp (#12), for both datasets alike. The conditioning previous
   full-res frame is ground-truth frame ``t - 1`` (teacher-forced, a single step).
3. **Predictions.** *Upsampler*: ``world_model.upsample_step``, the very step play uses, so the conditioning is
   ``[previous full-res, bicubic(new low-res)]`` with ``s_cond = 0`` and the sampler preset of ``--denoising-steps``.
   *Bicubic*: ``F.interpolate(low_res, (150, 280), mode="bicubic", align_corners=False)``, clamped.
4. **Metrics,** per frame, then mean and std over frames. All on RGB in [0, 1]: PSNR (peak 1), SSIM (11x11 Gaussian
   window, sigma 1.5, K1 = 0.01, K2 = 0.03, per channel then averaged, no padding) and LPIPS (AlexNet, which wants
   [-1, 1], so the frames are rescaled for it). Predictions are not quantised to uint8 first.
5. **Verdict.** Pass iff v2 LPIPS (upsampler) <= 1.15 x v1 LPIPS (upsampler) AND v2 LPIPS (upsampler) < v2 LPIPS
   (bicubic). The first criterion is one-sided: a v2 score better than v1's is not an incompatibility. Without
   ``--v1-fullres`` the first criterion cannot be evaluated and the verdict is incomplete, not a pass.

Caveats (also under ``notes`` in the JSON):

* v2 ground truth, and so the previous-frame conditioning, is JPEG-decoded (the 10% JPEG subset, quality 95) while v1
  is raw. That adds compression error to v2 only and biases the comparison against v2, i.e. conservatively.
* Scores are teacher-forced (previous frame = GT t-1) with ``s_cond = 0``. In play the previous frame is the
  upsampler's own output, so these scores are optimistic relative to play.

Outputs: ``upsampler_check.json`` and ``upsampler_check_grid.png`` in ``out_dir``, plus a printed summary table.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from cs2wm.actions import ACTION_DIM
from cs2wm.pipeline.pack import sha256_file, to_low_res
from cs2wm.play.agent import DEFAULT_AGENT_CONFIG_PATH, _import_agent
from cs2wm.play.world_model import frames_to_tensor, make_upsampler_sampler, upsample_step

LPIPS_TOLERANCE = 0.15  # v2 LPIPS may exceed v1's by at most this fraction
METRICS = ("psnr", "ssim", "lpips")
METHODS = ("upsampler", "bicubic")
GRID_FRAMES = 4  # rows per dataset in the grid
TILE_LABELS = ("low-res (nearest)", "bicubic", "upsampler", "ground truth")
_SSIM_WINDOW = 11
_SSIM_SIGMA = 1.5
_SSIM_K1, _SSIM_K2 = 0.01, 0.03

NOTES = (
    (
        "v2 ground truth and the previous-frame conditioning are JPEG-decoded (10% JPEG subset) while v1 is raw: "
        "this biases the comparison against v2 (conservative)."
    ),
    (
        "Scores are teacher-forced (previous frame = GT t-1, s_cond=0); in play the previous frame is generated, "
        "so they are optimistic relative to play."
    ),
)

LpipsFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR for images in [0, 1] (peak 1).

    Args:
        pred: ``float[b, 3, H, W]``.
        target: ``float[b, 3, H, W]``.

    Returns:
        ``float[b]`` in dB; identical images give 100 dB (the MSE is floored at 1e-10) instead of infinity.
    """
    mse = (pred - target).square().flatten(1).mean(1).clamp_min(1e-10)
    return -10 * torch.log10(mse)


def _gaussian_window(channels: int, device: torch.device) -> torch.Tensor:
    x = torch.arange(_SSIM_WINDOW, dtype=torch.float32, device=device) - (_SSIM_WINDOW - 1) / 2
    g = torch.exp(-(x**2) / (2 * _SSIM_SIGMA**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, _SSIM_WINDOW, _SSIM_WINDOW).contiguous()


def ssim_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-position SSIM map for images in [0, 1] (data range 1), averaged over channels.

    11x11 Gaussian window with sigma 1.5, K1 = 0.01, K2 = 0.03, computed per channel over the valid region (no
    padding). Entry ``(i, j)`` belongs to the window centred on pixel ``(i + 5, j + 5)``.

    Args:
        pred: ``float[b, c, H, W]`` with ``H, W >= 11``.
        target: ``float[b, c, H, W]``.

    Returns:
        ``float[b, H - 10, W - 10]``.
    """
    pred, target = pred.float(), target.float()
    win = _gaussian_window(pred.shape[1], pred.device)

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, win, groups=x.shape[1])

    c1, c2 = _SSIM_K1**2, _SSIM_K2**2
    mu_p, mu_t = blur(pred), blur(target)
    var_p = blur(pred * pred) - mu_p**2
    var_t = blur(target * target) - mu_t**2
    cov = blur(pred * target) - mu_p * mu_t
    num = (2 * mu_p * mu_t + c1) * (2 * cov + c2)
    den = (mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2)
    return (num / den).mean(1)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image SSIM for images in [0, 1] (data range 1): the mean of ``ssim_map``.

    Args:
        pred: ``float[b, c, H, W]`` with ``H, W >= 11``.
        target: ``float[b, c, H, W]``.

    Returns:
        ``float[b]``; 1 for identical images.
    """
    return ssim_map(pred, target).flatten(1).mean(1)


def make_lpips_fn(device: torch.device) -> LpipsFn:
    """AlexNet LPIPS from the ``lpips`` package (downloads the torchvision AlexNet weights on first use).

    Args:
        device: Device to run it on.

    Returns:
        A function ``(pred, target) -> float[b]`` taking frames in [-1, 1].
    """
    import lpips

    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)

    def fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return model(pred, target).flatten()

    return fn


def sample_frame_indices(lengths: np.ndarray, n: int, seed: int) -> list[tuple[int, int]]:
    """Draw ``(episode index, t)`` pairs with ``t >= 1``, spread across the episodes.

    Episodes are visited round-robin in a random order, each yielding its next frame from its own random permutation,
    so no frame repeats and every episode contributes within one of the others until it runs out.

    Args:
        lengths: Frames per episode.
        n: Frames wanted.
        seed: RNG seed.

    Returns:
        Up to ``n`` pairs (fewer if the dataset has fewer valid frames), in a fixed order for a given seed.

    Raises:
        ValueError: If no episode has two or more frames.
    """
    rng = np.random.default_rng(seed)
    pools = {int(e): list(rng.permutation(np.arange(1, int(m)))) for e, m in enumerate(lengths) if m >= 2}
    if not pools:
        raise ValueError("no episode has at least 2 frames, so no (t-1, t) pair can be formed")
    order = [int(e) for e in rng.permutation(list(pools))]
    out: list[tuple[int, int]] = []
    while len(out) < n and any(pools.values()):
        for e in order:
            if pools[e] and len(out) < n:
                out.append((e, int(pools[e].pop())))
    return out


def _to_bgr_uint8(x: torch.Tensor) -> np.ndarray:
    """``float[b, 3, H, W]`` RGB in [-1, 1] to ``uint8[b, H, W, 3]`` BGR (clamped before the cast)."""
    rgb = x.add(1).div(2).mul(255).round().clamp(0, 255).byte()
    return np.ascontiguousarray(rgb.flip(1).permute(0, 2, 3, 1).cpu().numpy())


def _unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] to [0, 1], clamped."""
    return x.add(1).div(2).clamp(0, 1)


def _to_rgb_uint8(x01: torch.Tensor) -> np.ndarray:
    """``float[3, H, W]`` in [0, 1] to ``uint8[H, W, 3]`` RGB."""
    return x01.mul(255).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()


def load_upsampler_agent(ckpt: Path, device: torch.device, agent_cfg: DictConfig | None = None) -> Any:
    """Build an eval-mode agent whose upsampler is loaded from ``ckpt`` (its denoiser stays unused).

    Args:
        ckpt: v1 agent checkpoint.
        device: Device to put the agent on.
        agent_cfg: Agent config with an upsampler; defaults to ``config/agent/cs2.yaml``.

    Returns:
        The agent in eval mode.

    Raises:
        ValueError: If the config has no upsampler.
    """
    if agent_cfg is None:
        agent_cfg = OmegaConf.load(DEFAULT_AGENT_CONFIG_PATH)
    if agent_cfg.get("upsampler") is None:
        raise ValueError("agent_cfg must define an upsampler")
    agent = _import_agent()(instantiate(agent_cfg, num_actions=ACTION_DIM))
    agent.load(ckpt, load_denoiser=False, load_upsampler=True)
    return agent.to(device).eval().requires_grad_(False)


def _load_pair(dataset: Any, episode: int, t: int) -> torch.Tensor:
    """Frames ``t - 1`` and ``t`` of an episode as ``float[2, 3, H, W]`` RGB in [-1, 1]."""
    from data import SegmentId

    return dataset[SegmentId(episode, t - 1, t + 1)].obs


@torch.no_grad()
def evaluate_dataset(
    directory: Path,
    sampler: Any,
    factor: int,
    lpips_fn: LpipsFn,
    n: int,
    seed: int,
    batch_size: int,
    device: torch.device,
    grid_frames: int = GRID_FRAMES,
) -> dict[str, Any]:
    """Score the upsampler and bicubic on frames sampled from one full-res dataset.

    Args:
        directory: Directory of full-res HDF5 episodes (v2 JPEG or v1 raw; ``CS2Hdf5Dataset`` reads both).
        sampler: Upsampler ``DiffusionSampler``.
        factor: The upsampler's upsampling factor.
        lpips_fn: LPIPS on frames in [-1, 1].
        n: Frames to evaluate.
        seed: Seeds both the frame sampling and the sampler noise.
        batch_size: Frames per forward pass.
        device: Device for the models and metrics.
        grid_frames: How many frames to keep as images for the grid.

    Returns:
        ``{"scores": {method: {metric: float[n] ndarray}}, "images": [rows]}`` where a row is the four grid tiles
        ``uint8[H, W, 3]`` RGB (low-res nearest-upscaled, bicubic, upsampler, ground truth).

    Raises:
        ValueError: If the directory has no usable frames.
    """
    from data import CS2Hdf5Dataset

    dataset = CS2Hdf5Dataset(directory)
    if dataset.num_episodes == 0:
        raise ValueError(f"{directory}: no .hdf5 files found")
    items = sample_frame_indices(dataset.lengths, n, seed)
    torch.manual_seed(seed)
    scores: dict[str, dict[str, list[np.ndarray]]] = {m: {k: [] for k in METRICS} for m in METHODS}
    images: list[list[np.ndarray]] = []
    for start in range(0, len(items), batch_size):
        pairs = torch.stack([_load_pair(dataset, e, t) for e, t in items[start : start + batch_size]]).to(device)
        prev, gt = pairs[:, 0], pairs[:, 1]
        # The low-res input goes through the one v1 recipe (plus clamp), the same for every dataset.
        low = frames_to_tensor(to_low_res(_to_bgr_uint8(gt))).to(device)
        preds = {
            "upsampler": upsample_step(sampler, prev, low, factor).clamp(-1, 1),
            "bicubic": F.interpolate(low, size=gt.shape[-2:], mode="bicubic", align_corners=False).clamp(-1, 1),
        }
        gt01 = _unit(gt)
        for method, pred in preds.items():
            pred01 = _unit(pred)
            scores[method]["psnr"].append(psnr(pred01, gt01).cpu().numpy())
            scores[method]["ssim"].append(ssim(pred01, gt01).cpu().numpy())
            scores[method]["lpips"].append(lpips_fn(pred, gt).float().cpu().numpy())
        for i in range(len(gt)):
            if len(images) >= grid_frames:
                break
            nearest = F.interpolate(low[i : i + 1], size=gt.shape[-2:], mode="nearest")[0]
            tiles = (nearest, preds["bicubic"][i], preds["upsampler"][i], gt[i])
            images.append([_to_rgb_uint8(_unit(x)) for x in tiles])
    return {"scores": {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in scores.items()}, "images": images}


def summarize(scores: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """Mean and (population) std of each metric per method, plus the frame count."""
    out: dict[str, Any] = {}
    for method, metrics in scores.items():
        out[method] = {k: {"mean": float(v.mean()), "std": float(v.std())} for k, v in metrics.items()}
        out["n"] = len(next(iter(metrics.values())))
    return out


def judge(v2: dict[str, Any], v1: dict[str, Any] | None) -> dict[str, Any]:
    """Apply the pass criteria to the summaries of ``summarize``.

    Args:
        v2: v2 summary.
        v1: v1 summary, or None if no v1 data was given.

    Returns:
        ``{"criteria": ..., "passed": bool | None, "summary": str}``. ``passed`` is None when the v1 reference is
        missing (the verdict is incomplete); it is False as soon as one evaluable criterion fails.
    """
    v2_up, v2_bi = v2["upsampler"]["lpips"]["mean"], v2["bicubic"]["lpips"]["mean"]
    beats = {"v2_upsampler_lpips": v2_up, "v2_bicubic_lpips": v2_bi, "passed": bool(v2_up < v2_bi)}
    compat: dict[str, Any] = {"threshold_ratio": 1 + LPIPS_TOLERANCE, "passed": None}
    if v1 is not None:
        v1_up = v1["upsampler"]["lpips"]["mean"]
        ratio = v2_up / v1_up if v1_up > 0 else (1.0 if v2_up == 0 else float("inf"))
        compat.update(v2_upsampler_lpips=v2_up, v1_upsampler_lpips=v1_up, ratio=ratio)
        compat["passed"] = bool(ratio <= 1 + LPIPS_TOLERANCE)
    named = (("v2 LPIPS more than 15% above v1", compat), ("upsampler not better than bicubic on v2", beats))
    failed = [name for name, c in named if c["passed"] is False]
    passed = False if failed else (None if compat["passed"] is None else True)
    if failed:
        summary = "FAIL: " + "; ".join(failed)
    elif passed is None:
        summary = "INCOMPLETE: no --v1-fullres, so v2 vs v1 LPIPS was not evaluated (v2 upsampler beats bicubic)"
    else:
        summary = "PASS: v2 LPIPS within 15% of v1 and better than bicubic on v2"
    return {"criteria": {"v2_vs_v1_lpips": compat, "v2_beats_bicubic": beats}, "passed": passed, "summary": summary}


def format_table(results: dict[str, dict[str, Any]]) -> str:
    """Printable summary table: one row per dataset and method."""
    lines = [f"{'dataset':<8}{'method':<11}{'n':>5}" + "".join(f"{m.upper():>16}" for m in METRICS)]
    for name, summ in results.items():
        for method in METHODS:
            cells = "".join(
                f"{summ[method][m]['mean']:>10.4f}+-{summ[method][m]['std']:<4.2f}".rjust(16) for m in METRICS
            )
            lines.append(f"{name:<8}{method:<11}{summ['n']:>5}{cells}")
    return "\n".join(lines)


def write_grid(path: Path, rows_by_dataset: dict[str, list[list[np.ndarray]]]) -> bool:
    """Write the side-by-side grid: low-res (nearest), bicubic, upsampler, ground truth per frame.

    Args:
        path: Output PNG.
        rows_by_dataset: Dataset name to rows of four ``uint8[H, W, 3]`` RGB tiles (any number of rows per dataset).

    Returns:
        True if the PNG was written; False (nothing written) if there are no rows at all.

    Raises:
        OSError: If the PNG cannot be written.
    """
    rows_by_dataset = {k: v for k, v in rows_by_dataset.items() if v}
    if not rows_by_dataset:
        return False
    font, band = cv2.FONT_HERSHEY_SIMPLEX, 26

    def label(text: str, width: int) -> np.ndarray:
        img = np.full((band, width, 3), 30, np.uint8)
        cv2.putText(img, text, (6, band - 8), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    tile_w = next(iter(next(iter(rows_by_dataset.values()))))[0].shape[1]
    parts = [np.concatenate([label(t, tile_w) for t in TILE_LABELS], axis=1)]
    for name, rows in rows_by_dataset.items():
        parts.append(label(f"dataset: {name}", tile_w * len(TILE_LABELS)))
        parts += [np.concatenate(row, axis=1) for row in rows]
    grid = np.concatenate(parts, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.ascontiguousarray(grid[..., ::-1])):
        raise OSError(f"could not write {path}")
    return True


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


def run_upsampler_check(
    ckpt: Path,
    fullres_dir: Path,
    n: int = 256,
    out_dir: Path = Path("results"),
    v1_fullres_dir: Path | None = None,
    seed: int = 0,
    device: str | torch.device | None = None,
    batch_size: int = 16,
    denoising_steps: int = 1,
    lpips_fn: LpipsFn | None = None,
    agent_cfg: DictConfig | None = None,
) -> dict:
    """Compare PSNR, SSIM and LPIPS of upsampled low-res frames against full-res ground truth.

    Args:
        ckpt: v1 agent checkpoint holding the upsampler.
        fullres_dir: v2 ``hdf5_fullres`` directory.
        n: Frames to evaluate per dataset.
        out_dir: Where ``upsampler_check.json`` and ``upsampler_check_grid.png`` are written.
        v1_fullres_dir: Directory of v1 test full-res HDF5 files; without it only v2 is scored and the verdict is
            incomplete.
        seed: Seeds the frame sampling and the sampler noise.
        device: Torch device; defaults to cuda if available.
        batch_size: Frames per forward pass.
        denoising_steps: Sampler preset for the upsampler, 1 (``fast.yaml``, play's default) or 3.
        lpips_fn: LPIPS on frames in [-1, 1]; defaults to AlexNet LPIPS (needs to download weights once).
        agent_cfg: Agent config with an upsampler; defaults to ``config/agent/cs2.yaml``.

    Returns:
        The report dict that was written to JSON: per-dataset metrics (``v2``, and ``v1`` if given), the verdict
        under ``verdict``, the ``grid`` status, ``notes`` and run provenance.

    Raises:
        ValueError: If ``n < 1``, or a dataset has no usable frames.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    agent = load_upsampler_agent(ckpt, dev, agent_cfg)
    factor = int(agent.upsampler.cfg.upsampling_factor)
    sampler = make_upsampler_sampler(agent.upsampler, denoising_steps)
    lpips_fn = lpips_fn if lpips_fn is not None else make_lpips_fn(dev)

    dirs = {"v2": fullres_dir, **({"v1": v1_fullres_dir} if v1_fullres_dir is not None else {})}
    results: dict[str, dict[str, Any]] = {}
    grid_rows: dict[str, list[list[np.ndarray]]] = {}
    for name, directory in dirs.items():
        res = evaluate_dataset(Path(directory), sampler, factor, lpips_fn, n, seed, batch_size, dev)
        results[name] = summarize(res["scores"])
        grid_rows[name] = res["images"]
    verdict = judge(results["v2"], results.get("v1"))

    report = {
        **results,
        "verdict": verdict,
        "n_requested": n,
        "denoising_steps": denoising_steps,
        "checkpoint": {"path": str(ckpt), "sha256_16": sha256_file(Path(ckpt))[:16]},
        "seed": seed,
        "device": str(dev),
        "torch_version": torch.__version__,
        "git_sha": _git_sha(),
        "notes": list(NOTES),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    written = write_grid(out_dir / "upsampler_check_grid.png", grid_rows)
    report["grid"] = {"written": written, "rows": {k: len(v) for k, v in grid_rows.items()}}
    (out_dir / "upsampler_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(format_table(results))
    print(verdict["summary"])
    return report
