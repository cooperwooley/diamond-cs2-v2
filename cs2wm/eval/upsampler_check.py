"""v1 upsampler compatibility check on v2 low-res frames (#27).

The v1 upsampler is reused unchanged, so check that it works on v2 frames (HLAE capture, no HUD) against a bicubic
baseline. Pass: v2 LPIPS within 15% of the v1 LPIPS and better than bicubic on v2.
"""

from __future__ import annotations

from pathlib import Path


def run_upsampler_check(ckpt: Path, fullres_dir: Path, n: int = 256, out_dir: Path = Path("results")) -> dict:
    """Compare PSNR, SSIM and LPIPS of upsampled low-res frames against full-res ground truth.

    Args:
        ckpt: v1 agent checkpoint holding the upsampler.
        fullres_dir: v2 ``hdf5_fullres`` directory.
        n: Frames to evaluate per dataset.
        out_dir: Where ``upsampler_check.json`` and the side-by-side grid PNG are written.

    Returns:
        The metrics dict (v2 and v1 test, upsampler and bicubic) plus the pass/fail verdict.
    """
    # TODO(#27): load only the upsampler (src Agent.load, #5); evaluate on v2 fullres and the v1 test set.
    raise NotImplementedError("TODO(#27): run_upsampler_check")
