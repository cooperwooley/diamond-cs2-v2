"""FID at 30x56 (#34). Ports v1 ``scripts/compute_fid.py`` (cooperwooley/diamond-cs2@a0b9e24), which is written against
DIAMOND's (Alonso et al., MIT) agent and sampler; see ATTRIBUTION.md.

v1's protocol: roll the denoiser out autoregressively, write the generated frames and the matching ground-truth frames
of the test episode as 30x56 PNGs, and run ``pytorch_fid.calculate_fid_given_paths`` with the default Inception feature
size (``dims=2048``, pool3). Inception's own bilinear resize to 299x299 (``resize_input=True``, inputs in [0, 1]
normalised to [-1, 1] inside the network) does the upscaling.

Here the PNG round trip is skipped (it is lossless for uint8 images): frames go from uint8 straight through the same
Inception model and the same statistics (mean, ``np.cov``) and Frechet distance (``calculate_frechet_distance``) as
pytorch-fid. The generation side (which frames, how many) is in ``rollouts.py`` and ``carried.py``.

Caveat: with 2,048 frames per side and 2,048 features the covariances are rank-deficient (rank <= n - 1). pytorch-fid
regularises the matrix square root for that, but the value is a little biased and only comparable to other FIDs
computed with the same n.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_fid.fid_score import calculate_frechet_distance

INCEPTION_DIMS = 2048  # pytorch-fid's default feature size
INCEPTION_BATCH = 64  # v1 compute_fid.py default

# uint8 RGB ``[n, h, w, 3]`` -> float features ``[n, d]``.
FeatureExtractor = Callable[[np.ndarray], np.ndarray]


def to_uint8(frames: torch.Tensor) -> np.ndarray:
    """Quantise frames in [-1, 1] to displayable uint8.

    v1 truncated (``astype(uint8)``) after scaling; rounding is used here so that a ground-truth frame survives the
    round trip through [-1, 1] exactly.

    Args:
        frames: ``float[n, 3, h, w]`` RGB; clamped to [-1, 1] first.

    Returns:
        ``uint8[n, h, w, 3]``.
    """
    x = frames.detach().float().cpu().clamp(-1, 1).add(1).div(2).mul(255).round().byte()
    return np.ascontiguousarray(x.permute(0, 2, 3, 1).numpy())


def make_inception_extractor(device: str | torch.device = "cpu", batch_size: int = INCEPTION_BATCH) -> FeatureExtractor:
    """The pytorch-fid Inception-v3 extractor at its default 2048-dim pool3 features.

    Downloads the FID Inception weights on first use.

    Args:
        device: Device for the network.
        batch_size: Images per forward pass.

    Returns:
        ``extract(images_uint8_rgb) -> float[n, 2048]``.
    """
    from pytorch_fid.inception import InceptionV3

    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[INCEPTION_DIMS]]).to(device).eval()

    @torch.no_grad()
    def extract(images: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(images[i : i + batch_size]).permute(0, 3, 1, 2).float().div(255).to(device)
            pred = model(x)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
            out.append(pred.squeeze(3).squeeze(2).cpu().numpy())
        return np.concatenate(out)

    return extract


def compute_fid(generated: np.ndarray, real: np.ndarray, extractor: FeatureExtractor) -> float:
    """Frechet distance between the feature statistics of two image sets.

    Args:
        generated: ``uint8[n, h, w, 3]`` RGB generated frames.
        real: ``uint8[m, h, w, 3]`` RGB real frames.
        extractor: Maps images to features (``make_inception_extractor`` for the real thing; tests inject a fake).

    Returns:
        The FID.

    Raises:
        ValueError: If a set has fewer than 2 images (no covariance) or the feature sizes differ.
    """
    stats = []
    for images in (generated, real):
        if len(images) < 2:
            raise ValueError(f"FID needs at least 2 images per set, got {len(images)}")
        feats = np.asarray(extractor(images), dtype=np.float64)
        stats.append((feats.mean(axis=0), np.cov(feats, rowvar=False)))
    (mu1, sigma1), (mu2, sigma2) = stats
    if mu1.shape != mu2.shape:
        raise ValueError(f"feature sizes differ: {mu1.shape} vs {mu2.shape}")
    return float(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))
