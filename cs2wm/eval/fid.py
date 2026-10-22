"""FID at 30x56 (#34). Ports v1 ``scripts/compute_fid.py`` (diamond-cs2 @ a0b9e24); see ATTRIBUTION.md.

2,048 generated vs real frames with pytorch-fid at its default Inception feature size, for v1 on the v1 test set, v2
on the v2 test set, and v1 on the v2 test set (domain-shift reference).
"""

from __future__ import annotations

from pathlib import Path


def compute_fid(ckpt: Path, data: Path, n_frames: int = 2048, seed: int = 0) -> float:
    """Compute FID between generated and real 30x56 frames.

    Args:
        ckpt: Agent checkpoint.
        data: Test split directory.
        n_frames: Number of frames per side.
        seed: RNG seed.

    Returns:
        The FID score.
    """
    # TODO(#34): port v1 scripts/compute_fid.py.
    raise NotImplementedError("TODO(#34): compute_fid")
