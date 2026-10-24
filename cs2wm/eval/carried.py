"""The carried-over v1 evals in one run (#34): loss curves, single/multi-step rollouts and FID at 30x56, v1 vs v2.

``cs2wm eval carried`` calls ``run_carried``, which ties together the ports of v1's ``scripts/evaluate.py``,
``eval_single_step.py`` and ``compute_fid.py`` (``loss_curve``, ``rollouts``, ``fid``; cooperwooley/diamond-cs2@a0b9e24,
see ATTRIBUTION.md).

Method
------
1. **Loss curves** (``loss_curve``): v1 and v2 test loss (and train loss) against epoch on the same axes.
2. **Windows.** ``--contexts`` windows (for the rollout figures and metrics) and enough windows for the FID
   (``ceil(n_fid / frames)``, whichever is more) of ``n_cond + frames`` consecutive ground-truth frames and actions are
   drawn from the v2 test set (``--data``) with ``--seed``; the first ``--contexts`` of them are the rollout contexts.
   The same windows are used for every model. v1's own test set (``--v1-data``) gets its own windows the same way.
3. **Rollouts.** Every model is rolled out ``--frames`` (64) steps from every window with ``WorldModel.rollout``
   (denoiser only; noise per window shared by all models). ``--denoising-steps`` defaults to 3, the step count of v1's
   scripts; the sampler's ``s_cond`` is the preset's 0.005 (v1's scripts used 0; ``--s-cond`` overrides it), so this
   re-evaluates v1 and v2 through the same code rather than reproducing v1's original numbers. Actions: v1 was trained
   on zero actions only, so the model named ``--v1-name`` is driven with zeros by default (``--v1-actions
   zeros|gt|both``; ``both`` gives two labelled rows, ``v1 (zero actions)`` and ``v1 (GT actions)``); every other model
   gets the ground-truth actions of the held-out episode. ``--zero-actions`` forces zeros for all models. Single-step
   grids, rollout strips and per-step MSE/PSNR come from the first ``--contexts`` windows (``rollouts``).
4. **FID** (``fid``): the generated frames of all windows (first ``--n-fid`` in window-major order) against the
   ground-truth frames at the same positions, at 30x56, pytorch-fid Inception at 2048 features, for ``--fid-seeds``
   seeds (windows and noise are redrawn per seed; mean and std are reported). Rows: v1 on the v1 test set, v2 on the v2
   test set, v1 on the v2 test set (domain-shift reference); in general every model on the v2 test set, plus the v1
   model on ``--v1-data``.

``summary.json`` (``schema_version`` 1)
---------------------------------------
``config``: ``frames``, ``contexts``, ``n_fid``, ``denoising_steps``, ``zero_actions``, ``v1_actions``, ``v1_name``,
    ``fid_seeds`` (list), ``batch_size``, ``device``, ``n_cond``, ``single_step_samples``, ``strip_contexts``,
    ``sampler`` (``num_steps_denoising``, ``sigma_min``, ``sigma_max``, ``rho``, ``order``, ``s_churn``, ``s_cond``).
``seed``; ``data``: dataset name (``v2_test``, ``v1_test``) -> ``{dir, format, n_episodes, n_windows, windows: [{index,
    episode, start}]}`` (the first seed's windows).
``models``: name -> ``{checkpoint: {path, sha256_16}}``.
``loss_curve``: ``x_axis`` ("epoch"), ``v1``: ``{steps_per_epoch, test: {epochs, denoiser, upsampler}, train_early:
    {epochs, loss, n_steps, mean_epoch, mean_loss}}``, ``v2``: ``{epochs, train_loss, test_loss, n_lines,
    n_duplicates}`` or null, ``figure``, ``notes``.
``rollouts``: ``{dataset, contexts, frames, steps_shown, metrics: {label: {mse: [T], psnr: [T], single_step: {mse,
    psnr}, action_source}, hold_last_context: ...}, files, windows}``; ``mse``/``psnr`` are per rollout step, averaged
    over contexts; ``label`` is the model name, or ``name (zero actions)`` / ``name (GT actions)`` for a model shown
    with both.
``fid``: ``{protocol, rows: [{label, model, dataset, action_source, n_generated, n_real, n_windows, seeds, fid,
    fid_std, fid_per_seed}]}``; ``fid`` is the mean over ``seeds`` and ``fid_std`` the sample std (null for one
    seed). ``protocol`` states the frame counts, resolution, feature size, which frames are real, the sampler, the
    caveats and ``deviations_from_v1``.
``warnings``; ``versions``; ``files``.

Other outputs in ``out_dir``: ``loss_curve.{pdf,png}``, ``single_step_grid.png``, ``rollout_strip_NN.png``,
``rollout_metrics.{pdf,png}``.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs2wm.eval import fid as fid_mod
from cs2wm.eval import loss_curve, rollouts
from cs2wm.pipeline.pack import sha256_file
from cs2wm.play.agent import build_denoiser_agent
from cs2wm.play.world_model import WorldModel, load_sampler_cfg

SCHEMA_VERSION = 1
V1_MODEL_NAME = "v1"
N_COND = 4  # the denoiser's num_steps_conditioning (v1 and v2)
V2_SET, V1_SET = "v2_test", "v1_test"
AgentFactory = Callable[[str, Path], Any]


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
    import pytorch_fid
    import scipy

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pytorch_fid": pytorch_fid.__version__,
        "git_sha": _git_sha(),
    }


def _sha16(path: Path) -> str | None:
    return sha256_file(path)[:16] if Path(path).exists() else None


def _dataset_summary(source: rollouts.EpisodeSource, windows: list[rollouts.Window]) -> dict[str, Any]:
    return {
        "dir": str(source.directory),
        "format": source.kind,
        "n_episodes": len(source),
        "n_windows": len(windows),
        "windows": [{"index": w.index, "episode": w.episode, "start": w.start} for w in windows],
    }


def _fid_features(gen: torch.Tensor, windows: list[rollouts.Window], n_fid: int) -> tuple[np.ndarray, np.ndarray]:
    """The first ``n_fid`` generated frames (window-major) and the ground truth at the same positions, as uint8."""

    def flat(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, *x.shape[2:])[:n_fid]

    return fid_mod.to_uint8(flat(gen)), fid_mod.to_uint8(flat(rollouts.ground_truth(windows, N_COND)))


def action_sources(name: str, v1_name: str, v1_actions: str, zero_actions: bool) -> list[str]:
    """The action sources a model is evaluated with.

    Args:
        name: Model name.
        v1_name: Name of the v1 model (trained on zero actions only).
        v1_actions: ``zeros``, ``gt`` or ``both``, for the v1 model.
        zero_actions: Force zeros for every model.

    Returns:
        A list of ``"gt"`` / ``"zeros"``.
    """
    if zero_actions:
        return ["zeros"]
    if name != v1_name:
        return ["gt"]
    return {"zeros": ["zeros"], "gt": ["gt"], "both": ["zeros", "gt"]}[v1_actions]


def _label(name: str, source: str, n_sources: int) -> str:
    """Row label: the model name, qualified by the action source only when a model appears with both."""
    if n_sources == 1:
        return name
    return f"{name} ({'zero' if source == 'zeros' else 'GT'} actions)"


def _sampler_record(world: WorldModel) -> dict[str, Any]:
    c = world.sampler.cfg
    keys = ("num_steps_denoising", "sigma_min", "sigma_max", "rho", "order", "s_churn", "s_cond")
    return {k: getattr(c, k) for k in keys}


def deviations_from_v1(n_frames: int, n_fid: int, n_windows: int, s_cond: float) -> list[str]:
    """What differs from v1's scripts, for the report to cite."""
    return [
        (
            f"v1's FID used 256 frames of one rollout from test episode 0 with zero actions; here {n_fid} frames from "
            f"{n_windows} windows of {n_frames} contiguous frames drawn over the held-out episodes"
        ),
        "generated frames are rounded to uint8, v1 truncated them",
        "no PNG round trip (lossless for uint8)",
        (
            f"sampler s_cond is {s_cond} (the world_model_env preset); v1's scripts used the default 0, so the numbers "
            "are not a reproduction of v1's originals, but v1 and v2 go through the same code"
        ),
        (
            "action source: v1 (trained on zero actions only) is driven with zeros by default, other models with the "
            "ground-truth actions of the held-out episode; see each row's action_source"
        ),
        (
            "FID over a few dozen 64-frame clips of contiguous frames is noisy (frames within a clip are correlated); see "
            "fid_std when --fid-seeds > 1"
        ),
    ]


def run_carried(
    models: dict[str, Path],
    data_dir: Path,
    out_dir: Path,
    v2_metrics: Path | None = None,
    v1_results: Path = Path("results/v1"),
    v1_data_dir: Path | None = None,
    split_json: Path | None = None,
    n_frames: int = 64,
    contexts: int = 16,
    n_fid: int = 2048,
    seed: int = 0,
    device: str | torch.device | None = None,
    batch_size: int = 16,
    denoising_steps: int = 3,
    zero_actions: bool = False,
    v1_actions: str = "zeros",
    v1_name: str = V1_MODEL_NAME,
    s_cond: float | None = None,
    fid_seeds: int = 1,
    single_step_samples: int = 12,
    strip_contexts: int = 3,
    agent_factory: AgentFactory | None = None,
    extractor: fid_mod.FeatureExtractor | None = None,
) -> dict[str, Any]:
    """Run the carried-over evals and write ``summary.json`` and the figures.

    Args:
        models: Name -> agent checkpoint (``v1``, ``v2``). Every model is rolled out on the v2 test set; the model
            named ``v1_name`` also on ``v1_data_dir``.
        data_dir: v2 test set: packed 30x56 HDF5 episodes.
        out_dir: Output directory (``results/carried``).
        v2_metrics: v2 ``metrics.jsonl``; without it the loss curve shows v1 alone.
        v1_results: Directory with v1's loss JSONs.
        v1_data_dir: v1 test set (DIAMOND ``low_res/test`` directory or packed HDF5); without it the v1-on-v1 FID row
            is skipped.
        split_json: Optional ``split.json`` keeping only test demos of ``data_dir``.
        n_frames: Rollout length.
        contexts: Windows for the rollout figures and metrics.
        n_fid: Frames per side for the FID.
        seed: RNG seed (window sampling and rollout noise).
        device: Torch device; defaults to cuda if available.
        batch_size: Rollouts per batched forward pass.
        denoising_steps: Sampler preset, 1 or 3 (v1's scripts used 3 steps).
        zero_actions: Force zero actions for every model (v1-style); overrides ``v1_actions``.
        v1_actions: Actions driving the ``v1_name`` model: ``zeros`` (it was trained on zeros only), ``gt`` or ``both``
            (two labelled rows). Other models always get the ground-truth actions.
        v1_name: Name of the v1 model in ``models``.
        s_cond: Sampler conditioning noise; None keeps the preset's (0.005; v1's scripts used 0).
        fid_seeds: Number of seeds (``seed``, ``seed + 1``, ...) to compute the FID with; mean and std are reported.
        single_step_samples: Rows of the single-step grid.
        strip_contexts: Windows shown as rollout strips.
        agent_factory: ``(name, ckpt) -> agent``; defaults to ``build_denoiser_agent`` on ``device``.
        extractor: FID feature extractor; defaults to pytorch-fid's Inception (downloads weights on first use).

    Returns:
        The summary dict that is written to ``summary.json``.

    Raises:
        ValueError: If ``models`` is empty, ``contexts``, ``n_frames`` or ``fid_seeds`` is below 1, ``v1_actions`` is
            unknown, or a model's context length is not 4.
    """
    if not models:
        raise ValueError("no models given")
    if contexts < 1 or n_frames < 1 or fid_seeds < 1:
        raise ValueError("contexts, n_frames and fid_seeds must be at least 1")
    if v1_actions not in ("zeros", "gt", "both"):
        raise ValueError(f"v1_actions must be zeros, gt or both, got {v1_actions!r}")
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    factory = agent_factory or (lambda name, ckpt: build_denoiser_agent(ckpt, dev))
    warnings: list[str] = []

    print("[carried] loss curves")
    loss = loss_curve.run_loss_curve(v2_metrics, v1_results, out_dir)
    if v2_metrics is None:
        warnings.append("no --v2-metrics: the loss curve shows v1 alone")

    n_windows = max(contexts, math.ceil(n_fid / n_frames))
    length = N_COND + n_frames
    seeds = [seed + i for i in range(fid_seeds)]
    sources = {V2_SET: rollouts.EpisodeSource(data_dir, split_json)}
    if v1_data_dir is not None:
        sources[V1_SET] = rollouts.EpisodeSource(v1_data_dir)
    elif v1_name in models:
        warnings.append("no --v1-data: the v1-on-v1-test FID row is skipped")
    # (dataset, seed) -> windows; the first seed's windows are the rollout contexts.
    windows = {
        (ds, sd): rollouts.sample_windows(src, n_windows, length, sd) for ds, src in sources.items() for sd in seeds
    }

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "frames": n_frames,
            "contexts": contexts,
            "n_fid": n_fid,
            "denoising_steps": denoising_steps,
            "zero_actions": zero_actions,
            "v1_actions": v1_actions,
            "v1_name": v1_name,
            "fid_seeds": seeds,
            "batch_size": batch_size,
            "device": str(dev),
            "n_cond": N_COND,
            "single_step_samples": single_step_samples,
            "strip_contexts": strip_contexts,
        },
        "seed": seed,
        "data": {ds: _dataset_summary(sources[ds], windows[(ds, seed)]) for ds in sources},
        "models": {},
        "loss_curve": loss,
    }

    gen: dict[tuple[str, str, int], torch.Tensor] = {}  # (label, dataset, seed) -> rollouts
    row_info: dict[str, tuple[str, str]] = {}  # label -> (model, action source)
    for name, ckpt in models.items():
        summary["models"][name] = {"checkpoint": {"path": str(ckpt), "sha256_16": _sha16(Path(ckpt))}}
        agent = factory(name, Path(ckpt))
        world = WorldModel(agent, denoising_steps, use_upsampler=False)
        if s_cond is not None:
            from models.diffusion import DiffusionSampler

            world.sampler = DiffusionSampler(agent.denoiser, load_sampler_cfg(denoising_steps, s_cond))
        if world.n_cond != N_COND:
            raise ValueError(f"{name}: num_steps_conditioning is {world.n_cond}, expected {N_COND}")
        summary["config"]["sampler"] = _sampler_record(world)
        srcs = action_sources(name, v1_name, v1_actions, zero_actions)
        for source in srcs:
            label = _label(name, source, len(srcs))
            row_info[label] = (name, source)
            for ds in sources:
                if ds == V1_SET and name != v1_name:
                    continue
                for sd in seeds:
                    done = 0

                    def progress(n: int, label: str = label, ds: str = ds) -> None:
                        nonlocal done
                        done += n
                        print(f"[carried] {label} on {ds}: {done}/{n_windows} rollouts", end="\r", flush=True)

                    gen[(label, ds, sd)] = rollouts.generate_rollouts(
                        world, windows[(ds, sd)], sd, batch_size, source == "zeros", progress
                    )
                    print()
        del world, agent
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    preds = {label: gen[(label, V2_SET, seed)][:contexts] for label in row_info}
    res = rollouts.run_rollouts(
        preds,
        windows[(V2_SET, seed)][:contexts],
        N_COND,
        out_dir,
        single_step_samples,
        strip_contexts,
        {label: info[1] for label, info in row_info.items()},
    )
    summary["rollouts"] = {"dataset": V2_SET, "contexts": contexts, "frames": n_frames, **res}

    print("[carried] FID")
    extract = extractor or fid_mod.make_inception_extractor(dev)
    rows = []
    for label, (name, source) in row_info.items():
        for ds in (V1_SET, V2_SET):  # the v1-on-v1 row first, as in the issue
            if (label, ds, seed) not in gen:
                continue
            print(f"[carried] FID {label} on {ds}")
            per_seed = []
            for sd in seeds:
                g, r = _fid_features(gen[(label, ds, sd)], windows[(ds, sd)], n_fid)
                per_seed.append(fid_mod.compute_fid(g, r, extract))
            rows.append(
                {
                    "label": label,
                    "model": name,
                    "dataset": ds,
                    "action_source": source,
                    "n_generated": len(g),
                    "n_real": len(r),
                    "n_windows": len(windows[(ds, seed)]),
                    "seeds": seeds,
                    "fid": float(np.mean(per_seed)),
                    "fid_std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None,
                    "fid_per_seed": per_seed,
                }
            )
    sampler = summary["config"]["sampler"]
    summary["fid"] = {
        "protocol": {
            "resolution": "30x56",
            "n_per_side": n_fid,
            "features": f"pytorch-fid Inception-v3, {fid_mod.INCEPTION_DIMS}-dim pool3 (default), inputs resized to "
            "299x299 inside the network",
            "generated": f"{n_frames}-step autoregressive rollouts from ground-truth {N_COND}-frame contexts, "
            f"{denoising_steps} denoising step(s); the actions are per row (action_source: 'gt' = the held-out "
            "episode's actions, 'zeros' = all-zero actions)",
            "real": "the ground-truth frames at the generated frames' positions (the frames following each context)",
            "sampler": sampler,
            "caveat": "n equals the feature size, so the covariances are rank-deficient; compare only across equal n",
            "deviations_from_v1": deviations_from_v1(n_frames, n_fid, n_windows, sampler["s_cond"]),
        },
        "rows": rows,
    }
    summary["warnings"] = warnings
    summary["versions"] = _versions()
    summary["files"] = loss["figure"] + res["files"] + ["summary.json"]
    (out_dir / "summary.json").write_text(json.dumps(_json_clean(summary), indent=2) + "\n")
    for w in warnings:
        print(f"[carried] WARNING: {w}", file=sys.stderr)
    print(format_summary(summary))
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    """A short text table of the FID rows and the mean rollout PSNR per model."""
    lines = ["carried evals", f"  {'model':<26}{'dataset':<10}{'n':>6}{'FID':>10}"]
    for r in summary["fid"]["rows"]:
        lines.append(f"  {r['label']:<26}{r['dataset']:<10}{r['n_generated']:>6}{r['fid']:>10.2f}")
    lines.append(
        "  mean rollout PSNR (dB): "
        + ", ".join(f"{m} {np.mean(v['psnr']):.2f}" for m, v in summary["rollouts"]["metrics"].items())
    )
    return "\n".join(lines)
