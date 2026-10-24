"""One-frame-at-a-time world model for play and the FPS benchmark (#28).

The src/ copy of DIAMOND has no ``WorldModelEnv``, so this is the small piece of it that play needs. It keeps the last
``num_steps_conditioning`` low-res frames and actions, samples the next 30x56 frame with ``DiffusionSampler`` and, if an
upsampler is present, upsamples it the way DIAMOND does: the upsampler is conditioned on the previous full-res frame and
the bicubic-upsampled new low-res frame (the ``cat`` order of ``Denoiser.forward``), and its output becomes the next
"previous full-res frame".

Frames are float tensors in [-1, 1], RGB, ``[3, H, W]``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from cs2wm.actions import ActionSpec, encode
from cs2wm.pipeline.pack import read_episode

if TYPE_CHECKING:
    from agent import Agent

_REPO_ROOT = Path(__file__).resolve().parents[2]
WORLD_MODEL_ENV_DIR = _REPO_ROOT / "config" / "world_model_env"
# denoising_steps -> sampler preset, as in DIAMOND's world_model_env configs.
SAMPLER_PRESETS = {1: "fast.yaml", 3: "higher_quality.yaml"}


def load_sampler_cfg(denoising_steps: int, s_cond: float | None = None) -> Any:
    """Load a ``DiffusionSamplerConfig`` from ``config/world_model_env``.

    Args:
        denoising_steps: 1 (``fast.yaml``) or 3 (``higher_quality.yaml``).
        s_cond: Overrides the preset's conditioning noise (the upsampler samples with 0).

    Returns:
        The sampler config.

    Raises:
        ValueError: If there is no preset for ``denoising_steps``.
    """
    from models.diffusion import DiffusionSamplerConfig

    if denoising_steps not in SAMPLER_PRESETS:
        raise ValueError(f"denoising_steps must be one of {sorted(SAMPLER_PRESETS)}, got {denoising_steps}")
    raw = OmegaConf.to_container(OmegaConf.load(WORLD_MODEL_ENV_DIR / SAMPLER_PRESETS[denoising_steps]))
    raw.pop("_target_")
    # The presets spell infinity with a custom OmegaConf resolver that only src/main.py registers.
    if isinstance(raw.get("s_tmax"), str):
        raw["s_tmax"] = float("inf")
    if s_cond is not None:
        raw["s_cond"] = s_cond
    return DiffusionSamplerConfig(**raw)


def bicubic_up(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Bicubic upscale exactly as ``Denoiser.forward`` builds the upsampler's low-res conditioning.

    Args:
        x: ``float[b, 3, h, w]``.
        factor: Integer upsampling factor.

    Returns:
        ``float[b, 3, h * factor, w * factor]``. Not clamped: ringing can leave the input range, as in training.
    """
    return F.interpolate(x, scale_factor=factor, mode="bicubic")


def make_upsampler_sampler(upsampler: Any, denoising_steps: int = 1) -> Any:
    """Build the sampler play uses for the upsampler (preset ``denoising_steps``, ``s_cond=0``).

    Args:
        upsampler: ``agent.upsampler``.
        denoising_steps: 1 or 3.

    Returns:
        A ``DiffusionSampler``.
    """
    from models.diffusion import DiffusionSampler

    return DiffusionSampler(upsampler, load_sampler_cfg(denoising_steps, s_cond=0.0))


# Draws standard normal noise of shape ``[b, *shape]``; see ``sample_from_noise``.
RandnFn = Callable[..., torch.Tensor]


def sample_from_noise(
    sampler: Any, prev_obs: torch.Tensor, prev_act: torch.Tensor | None, randn: RandnFn
) -> torch.Tensor:
    """``DiffusionSampler.sample`` with the noise supplied by the caller.

    The stock sampler draws all of its noise from the global torch RNG, so a sample depends on its position in the
    batch. This copy of its loop takes the noise from ``randn`` instead, which lets an evaluation give every sample
    its own reproducible noise and batch them freely. The maths is the sampler's (Euler or Heun, optional churn and
    conditioning noise); the draw order is: initial noise, then per step the churn noise (if any) and the
    conditioning noise (if ``s_cond > 0``).

    Args:
        sampler: A ``DiffusionSampler``.
        prev_obs: ``float[b, t, c, h, w]`` conditioning frames.
        prev_act: ``float[b, t, 51]`` conditioning actions, or None (the upsampler).
        randn: Called as ``randn(*shape)`` and returns ``float[b, *shape]`` standard normal noise on the right device.

    Returns:
        ``float[b, c, h, w]`` sample, like ``DiffusionSampler.sample(...)[0]``.
    """
    cfg, sigmas, denoiser = sampler.cfg, sampler.sigmas, sampler.denoiser
    b, t, c, h, w = prev_obs.shape
    prev_obs = prev_obs.reshape(b, t * c, h, w)
    s_in = torch.ones(b, device=prev_obs.device)
    gamma_ = min(cfg.s_churn / (len(sigmas) - 1), 2**0.5 - 1)
    x = randn(c, h, w)
    for sigma, next_sigma in itertools.pairwise(sigmas):
        gamma = gamma_ if cfg.s_tmin <= sigma <= cfg.s_tmax else 0
        sigma_hat = sigma * (gamma + 1)
        if gamma > 0:
            x = x + randn(c, h, w) * cfg.s_noise * (sigma_hat**2 - sigma**2) ** 0.5
        if cfg.s_cond > 0:
            sigma_cond = torch.full((b,), fill_value=cfg.s_cond, device=prev_obs.device)
            # Denoiser.apply_noise with sigma_offset_noise=0.
            prev_obs = prev_obs + randn(t * c, h, w) * sigma_cond.view(b, 1, 1, 1)
        else:
            sigma_cond = None
        denoised = denoiser.denoise(x, sigma, sigma_cond, prev_obs, prev_act)
        d = (x - denoised) / sigma_hat
        dt = next_sigma - sigma_hat
        if cfg.order == 1 or next_sigma == 0:
            x = x + d * dt
        else:
            x_2 = x + d * dt
            denoised_2 = denoiser.denoise(x_2, next_sigma * s_in, sigma_cond, prev_obs, prev_act)
            d_2 = (x_2 - denoised_2) / next_sigma
            x = x + (d + d_2) / 2 * dt
    return x


def upsample_step(
    sampler: Any, prev_full: torch.Tensor, low: torch.Tensor, factor: int, randn: RandnFn | None = None
) -> torch.Tensor:
    """Sample the next full-res frames from the previous full-res frames and the new low-res frames.

    The conditioning is ``[previous full-res, bicubic(new low-res)]`` (the ``cat`` order of ``Denoiser.forward``).
    Play, the upsampler compatibility check (#27) and the H1 eval (#32) all go through this, so they condition
    identically.

    Args:
        sampler: Upsampler ``DiffusionSampler`` from ``make_upsampler_sampler``.
        prev_full: ``float[b, 3, H, W]`` in [-1, 1].
        low: ``float[b, 3, h, w]`` in [-1, 1].
        factor: The upsampler's ``upsampling_factor``.
        randn: Per-sample noise source for ``sample_from_noise``; None uses the global torch RNG.

    Returns:
        ``float[b, 3, H, W]``; unclamped sampler output (about [-1, 1]).
    """
    cond = torch.stack((prev_full, bicubic_up(low, factor)), dim=1)
    if randn is None:
        full, _ = sampler.sample(cond, None)
    else:
        full = sample_from_noise(sampler, cond, None, randn)
    return full.float()


def frames_to_tensor(frames_bgr: np.ndarray) -> torch.Tensor:
    """Convert packed BGR uint8 frames to RGB floats in [-1, 1].

    Args:
        frames_bgr: ``uint8[n, H, W, 3]`` as returned by ``pack.read_episode``.

    Returns:
        ``float32[n, 3, H, W]``.
    """
    rgb = np.ascontiguousarray(frames_bgr[..., ::-1])
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div(255).mul(2).sub(1)


def tensor_to_rgb_uint8(frame: torch.Tensor) -> np.ndarray:
    """Convert a [-1, 1] frame to displayable RGB.

    Args:
        frame: ``float[3, H, W]``.

    Returns:
        ``uint8[H, W, 3]``. Clamped first because a float to uint8 cast wraps instead of saturating.
    """
    x = frame.detach().float().cpu().add(1).div(2).mul(255).round().clamp(0, 255).byte()
    return x.permute(1, 2, 0).numpy()


def load_context(path: Path, n: int) -> tuple[torch.Tensor, np.ndarray]:
    """Load the first ``n`` frames and actions of a packed episode.

    Args:
        path: Packed HDF5 episode (held-out test split).
        n: Context length (the denoiser's ``num_steps_conditioning``).

    Returns:
        ``(frames, actions)``: ``float32[n, 3, h, w]`` RGB in [-1, 1] and ``float32[n, 51]``.

    Raises:
        ValueError: If the episode has fewer than ``n`` frames.
    """
    frames, actions, _ = read_episode(path)
    if len(frames) < n:
        raise ValueError(f"{path}: need at least {n} frames for the context, got {len(frames)}")
    return frames_to_tensor(frames[:n]), actions[:n]


def blank_context(n: int, low_res: Sequence[int], spec: ActionSpec) -> tuple[torch.Tensor, np.ndarray]:
    """A mid-grey context with idle actions, used when no ``--context`` episode is given.

    Args:
        n: Context length.
        low_res: ``(h, w)`` of the low-res frames.
        spec: Action layout.

    Returns:
        ``(frames, actions)`` shaped like ``load_context``.
    """
    h, w = low_res
    idle = encode([], False, False, 0.0, 0.0, spec)
    return torch.zeros(n, 3, h, w), np.tile(idle, (n, 1))


class WorldModel:
    """Rolling-context next-frame sampler.

    Args:
        agent: Eval-mode agent; ``agent.upsampler`` is used only if ``use_upsampler``.
        denoising_steps: 1 or 3.
        use_upsampler: Upsample with the agent's upsampler; otherwise show a bicubic upscale.
        bf16: Run sampling under bf16 autocast (CUDA only).
        compile: ``torch.compile`` the inner U-Nets in place (the first steps are slow while it compiles).
        upsampling_factor: Used for the bicubic fallback; read from the upsampler config when it exists.
    """

    def __init__(
        self,
        agent: Agent,
        denoising_steps: int = 1,
        use_upsampler: bool = True,
        bf16: bool = False,
        compile: bool = False,
        upsampling_factor: int = 5,
    ) -> None:
        from models.diffusion import DiffusionSampler

        if use_upsampler and agent.upsampler is None:
            raise ValueError("use_upsampler=True but the agent has no upsampler")
        self.device = agent.device
        if bf16 and self.device.type != "cuda":
            raise ValueError("bf16 autocast is only supported on CUDA")
        self.bf16 = bf16
        if compile:
            agent.denoiser.inner_model.compile()
            if use_upsampler:
                agent.upsampler.inner_model.compile()
        self.n_cond = int(agent.denoiser.cfg.inner_model.num_steps_conditioning)
        self.sampler = DiffusionSampler(agent.denoiser, load_sampler_cfg(denoising_steps))
        self.up_sampler = None
        self.factor = upsampling_factor
        if use_upsampler:
            self.factor = int(agent.upsampler.cfg.upsampling_factor)
            self.up_sampler = make_upsampler_sampler(agent.upsampler, denoising_steps)
        self._ctx_obs: torch.Tensor | None = None
        self._ctx_act: torch.Tensor | None = None
        self._init: tuple[torch.Tensor, torch.Tensor] | None = None
        self.low_res: torch.Tensor  # latest low-res frame [3, h, w]
        self.full_res: torch.Tensor  # latest displayed frame [3, H, W]

    @property
    def uses_upsampler(self) -> bool:
        """Whether frames are upsampled by the model (False: bicubic)."""
        return self.up_sampler is not None

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.bf16 else nullcontext()

    def _bicubic(self, x: torch.Tensor) -> torch.Tensor:
        return bicubic_up(x, self.factor)

    def reset(self, frames: torch.Tensor, actions: np.ndarray | torch.Tensor) -> None:
        """Set the context and the displayed frame.

        The upsampler also needs a previous full-res frame, which the context does not have (packed episodes hold only
        low-res frames). The first one is a bicubic upscale of the last context frame; it is replaced by the
        upsampler's own output after one step, so the error does not accumulate. A full-res twin of the episode
        could provide a better first frame, but is not used.

        Args:
            frames: ``float[n_cond, 3, h, w]`` RGB in [-1, 1]; the last ``n_cond`` are kept.
            actions: ``[n_cond, 51]`` actions aligned with ``frames``. The last one is overwritten by the first
                ``step`` (it is the action taken at the last frame).

        Raises:
            ValueError: If fewer than ``n_cond`` frames or actions are given.
        """
        act = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
        if len(frames) < self.n_cond or len(act) < self.n_cond:
            raise ValueError(f"need {self.n_cond} context frames and actions, got {len(frames)} and {len(act)}")
        obs = frames[-self.n_cond :].to(self.device, torch.float32)
        act = act[-self.n_cond :].to(self.device)
        self._init = (obs.clone(), act.clone())
        self._ctx_obs, self._ctx_act = obs, act
        self.low_res = obs[-1]
        self.full_res = self._bicubic(obs[-1:])[0].clamp(-1, 1)

    def restart(self) -> None:
        """Return to the context given to the last ``reset`` (Backspace)."""
        if self._init is None:
            raise RuntimeError("restart() before reset()")
        self.reset(*self._init)

    def _advance(
        self,
        ctx_obs: torch.Tensor,
        ctx_act: torch.Tensor,
        full_res: torch.Tensor,
        action: torch.Tensor,
        randn_low: RandnFn | None = None,
        randn_up: RandnFn | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step for a batch of ``b`` independent streams (``step`` is the ``b = 1`` case).

        Args:
            ctx_obs: ``float[b, n_cond, 3, h, w]``.
            ctx_act: ``float[b, n_cond, 51]``; the last slot is replaced by ``action``.
            full_res: ``float[b, 3, H, W]`` the previous displayed frames.
            action: ``float[b, 51]``.
            randn_low: Noise source for the denoiser sampler, or None for the global RNG.
            randn_up: Noise source for the upsampler sampler, or None for the global RNG.

        Returns:
            ``(ctx_obs, ctx_act, low, full)``: the rolled contexts and the new ``float[b, 3, h, w]`` and
            ``float[b, 3, H, W]`` frames.
        """
        ctx_act = torch.cat((ctx_act[:, :-1], action[:, None]), dim=1)
        with self._autocast():
            if randn_low is None:
                low, _ = self.sampler.sample(ctx_obs, ctx_act)
            else:
                low = sample_from_noise(self.sampler, ctx_obs, ctx_act, randn_low)
            low = low.float()
            if self.up_sampler is not None:
                full = upsample_step(self.up_sampler, full_res, low, self.factor, randn_up)
        if self.up_sampler is None:
            full = self._bicubic(low).clamp(-1, 1)
        ctx_obs = torch.cat((ctx_obs[:, 1:], low[:, None]), dim=1)
        # The slot for the next frame's action; replaced by the next step's action before it is read.
        ctx_act = torch.cat((ctx_act[:, 1:], ctx_act[:, -1:]), dim=1)
        return ctx_obs, ctx_act, low, full

    @torch.no_grad()
    def step(self, action: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Advance one frame.

        Args:
            action: ``float[51]`` action taken at the current frame.

        Returns:
            The new displayed frame ``float[3, H, W]`` in [-1, 1] (upsampled or bicubic).

        Raises:
            RuntimeError: If called before ``reset``.
        """
        if self._ctx_obs is None or self._ctx_act is None:
            raise RuntimeError("step() before reset()")
        act = torch.as_tensor(np.asarray(action), dtype=torch.float32, device=self.device)
        obs, act_ctx, low, full = self._advance(
            self._ctx_obs[None], self._ctx_act[None], self.full_res[None], act[None]
        )
        self._ctx_obs, self._ctx_act = obs[0], act_ctx[0]
        self.low_res, self.full_res = low[0], full[0]
        return self.full_res

    @torch.no_grad()
    def rollout(
        self,
        frames: torch.Tensor,
        ctx_actions: torch.Tensor,
        actions: torch.Tensor,
        generators: Sequence[torch.Generator],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll ``b`` independent contexts forward in one batch, each with its own reproducible noise.

        Per stream this is ``reset`` followed by ``step`` for every row of ``actions`` (same context handling, same
        ``[prev full-res, bicubic(new low-res)]`` upsampler conditioning), but the streams share the forward passes.
        It leaves the single-stream state of the model untouched. Sample ``i`` draws all its noise from
        ``generators[i]`` (on the CPU, then moved to the device), in a fixed order, so its frames do not depend on
        the other streams in the batch or on the model: the same generator seed gives the same noise for any model.

        Args:
            frames: ``float[b, >=n_cond, 3, h, w]`` context frames in [-1, 1]; the last ``n_cond`` are kept.
            ctx_actions: ``float[b, >=n_cond, 51]`` context actions (the last is overwritten by the first action).
            actions: ``float[b, T, 51]`` the action held at each rollout step.
            generators: ``b`` CPU generators, one per stream.

        Returns:
            ``(low, full)``: ``float[b, T, 3, h, w]`` and ``float[b, T, 3, H, W]`` (the displayed frames; bicubic
            if the model has no upsampler), on the CPU.

        Raises:
            ValueError: On inconsistent batch sizes or too short a context.
        """
        b = len(frames)
        if not (len(ctx_actions) == len(actions) == len(generators) == b):
            raise ValueError("frames, ctx_actions, actions and generators must have the same batch size")
        if frames.shape[1] < self.n_cond or ctx_actions.shape[1] < self.n_cond:
            raise ValueError(f"need {self.n_cond} context frames and actions")
        obs = frames[:, -self.n_cond :].to(self.device, torch.float32)
        act = ctx_actions[:, -self.n_cond :].to(self.device, torch.float32)
        full = self._bicubic(obs[:, -1]).clamp(-1, 1)

        def randn(*shape: int) -> torch.Tensor:
            return torch.stack([torch.randn(shape, generator=g) for g in generators]).to(self.device)

        lows, fulls = [], []
        for t in range(actions.shape[1]):
            obs, act, low, full = self._advance(obs, act, full, actions[:, t].to(self.device), randn, randn)
            lows.append(low.cpu())
            fulls.append(full.cpu())
        return torch.stack(lows, dim=1), torch.stack(fulls, dim=1)
