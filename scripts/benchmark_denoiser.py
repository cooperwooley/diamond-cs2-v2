#!/usr/bin/env python3
"""Micro-benchmark of denoiser training throughput for one hardware profile (#6, used by #8).

For each requested batch size it runs ``--steps`` optimizer steps on the data in ``--data`` and prints exactly one
line per configuration::

    BENCH {"hw": "h100", "device": "<GPU name>", "batch_size": 128, "precision": "bf16", "compile": false,
           "steps": 200, "ms_per_step": ..., "samples_per_s": ..., "peak_mem_gb": ...,
           "est_hours_150_epochs": ..., "oom": false}

``est_hours_150_epochs`` assumes 400 steps per epoch at an effective batch of 128 (v1 parity). On OOM the line is
still printed, with ``"oom": true`` and null timings. ``scripts/mill/benchmark.sbatch`` greps these lines.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EPOCHS = 150
STEPS_PER_EPOCH = 400
EFFECTIVE_BATCH = 128
TINY_MODEL = [
    "agent.denoiser.inner_model.cond_channels=16",
    "agent.denoiser.inner_model.depths=[1,1,1,1]",
    "agent.denoiser.inner_model.channels=[8,8,8,8]",
    "agent.denoiser.inner_model.attn_depths=[0,0,0,0]",
]
N_CACHED_BATCHES = 4


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hw", required=True, choices=["h100", "v100", "rtx3060"], help="config/hw profile")
    p.add_argument("--steps", type=int, default=200, help="optimizer steps per configuration")
    p.add_argument("--warmup", type=int, default=5, help="untimed optimizer steps before timing")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--data", required=True, help="low_res dataset directory")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default=None, help="default: the hw profile's")
    p.add_argument("--compile", action="store_true", help="torch.compile the denoiser")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--tiny", action="store_true", help="shrink the model to a few thousand parameters (CPU tests)")
    p.add_argument(
        "--override", nargs="*", default=[], help="extra Hydra overrides, e.g. denoiser.training.max_grad_norm=null"
    )
    return p


def estimate_hours(ms_per_step: float, batch_size: int) -> float:
    """Hours for 150 epochs of 400 steps at an effective batch of 128, at this config's per-sample speed."""
    return EPOCHS * STEPS_PER_EPOCH * EFFECTIVE_BATCH * (ms_per_step / batch_size) / 1000 / 3600


def compose_cfg(args: argparse.Namespace):
    """Compose config/trainer.yaml with the hw profile and a denoiser-only agent."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)  # as in src/main.py
    overrides = [
        f"hw={args.hw}",
        "agent=cs2_denoiser_only",
        f"env.path_data_low_res={args.data}",
        "env.path_data_full_res=null",
        *(TINY_MODEL if args.tiny else []),
        *args.override,
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "config")):
        return compose(config_name="trainer", overrides=overrides)


def bench_one(cfg, args: argparse.Namespace, device, precision: str, batch_size: int) -> dict:
    """Time `args.steps` optimizer steps of batch `batch_size`, raising torch.cuda.OutOfMemoryError on OOM."""
    import torch
    from hydra.utils import instantiate
    from torch.utils.data import DataLoader

    from agent import Agent
    from data import BatchSampler, Dataset, collate_segments_to_batch
    from utils import autocast, configure_opt

    c = cfg.denoiser.training
    seq_length = cfg.agent.denoiser.inner_model.num_steps_conditioning + 1 + c.num_autoregressive_steps

    dataset = Dataset(Path(cfg.static_dataset.path) / "train", None, "train_dataset", False)
    dataset.load_from_default_path()
    loader = DataLoader(
        dataset,
        batch_sampler=BatchSampler(dataset, 0, 1, batch_size, seq_length, None),
        collate_fn=collate_segments_to_batch,
    )
    # Sampling is excluded from the timing: a few distinct batches are drawn up front and cycled.
    batches = [b.to(device) for _, b in zip(range(N_CACHED_BATCHES), loader)]

    agent = Agent(instantiate(cfg.agent, num_actions=cfg.env.num_actions)).to(device)
    agent.setup_training(instantiate(cfg.denoiser.sigma_distribution), None)
    model = agent.denoiser
    if args.compile or cfg.training.compile_wm:
        model.compile()
    opt = configure_opt(model, **cfg.denoiser.optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and device.type == "cuda")
    model.train()

    def step(i: int) -> None:
        with autocast(precision, device.type):
            loss, _ = model(batches[i % len(batches)])
        scaler.scale(loss).backward()
        if c.max_grad_norm is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(args.warmup):
        step(i)
    sync()
    t0 = time.perf_counter()
    for i in range(args.steps):
        step(i)
    sync()
    ms_per_step = (time.perf_counter() - t0) / args.steps * 1000
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
    return {
        "ms_per_step": ms_per_step,
        "samples_per_s": batch_size / ms_per_step * 1000,
        "peak_mem_gb": peak,
        "est_hours_150_epochs": estimate_hours(ms_per_step, batch_size),
        "oom": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT / "src"))  # v1 code uses flat imports (agent, data, utils, ...)
    import torch

    cfg = compose_cfg(args)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    precision = args.precision or cfg.training.precision
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = cfg.training.tf32
        torch.backends.cudnn.allow_tf32 = cfg.training.tf32

    for batch_size in args.batch_sizes:
        row = {
            "hw": args.hw,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "batch_size": batch_size,
            "precision": precision,
            "compile": bool(args.compile or cfg.training.compile_wm),
            "steps": args.steps,
        }
        try:
            row.update(bench_one(cfg, args, device, precision, batch_size))
        except torch.cuda.OutOfMemoryError:
            row.update(ms_per_step=None, samples_per_s=None, peak_mem_gb=None, est_hours_150_epochs=None, oom=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("BENCH " + json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
