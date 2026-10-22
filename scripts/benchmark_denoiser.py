"""Micro-benchmark of denoiser training throughput for one hardware profile (#6, used by #8).

For each requested batch size it runs ``--steps`` optimizer steps on the data in ``--data`` and prints exactly one
line per configuration::

    BENCH {"hw": "h100", "batch_size": 128, "precision": "bf16", "compile": false, "ms_per_step": ...,
           "samples_per_s": ..., "peak_mem_gb": ..., "est_hours_150_epochs": ..., "oom": false}

``est_hours_150_epochs`` assumes 400 steps per epoch at an effective batch of 128 (v1 parity). On OOM the line is
still printed, with ``"oom": true`` and null timings. ``scripts/mill/benchmark.sbatch`` greps these lines.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hw", required=True, choices=["h100", "v100", "rtx3060"], help="config/hw profile")
    p.add_argument("--steps", type=int, default=200, help="optimizer steps per configuration")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--data", required=True, help="low_res dataset directory")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default=None, help="default: the hw profile's")
    p.add_argument("--compile", action="store_true", help="torch.compile the denoiser")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    build_parser().parse_args(argv)
    # TODO(#6): build the denoiser from config/hw/<hw>.yaml, time --steps steps per batch size, print BENCH lines.
    raise NotImplementedError("TODO(#6): benchmark_denoiser")


if __name__ == "__main__":
    raise SystemExit(main())
