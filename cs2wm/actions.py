"""The 51-dim action space shared by label extraction and live play (#9).

Layout (identical to v1, DIAMOND CS:GO and Pearce & Zhu): keys[11] + clicks[2] + mouse_x one-hot[23] +
mouse_y one-hot[15]. Mouse deltas are binned in degrees; see ``actions.mouse`` in ``config/cs2wm.yaml``.
Sign convention: +x is turn right, +y is look down.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from omegaconf import DictConfig

N_KEYS = 11
N_CLICKS = 2
ACTION_DIM = 51


@dataclass(frozen=True)
class ActionSpec:
    """Layout and mouse bins of the action vector.

    Attributes:
        keys: Key names in vector order (``w, a, s, d, space, ctrl, shift, 1, 2, 3, r``).
        clicks: Click names (``lmb, rmb``).
        mouse_x_units: Mouse-x bin centres in Pearce & Zhu units.
        mouse_y_units: Mouse-y bin centres in Pearce & Zhu units.
        deg_per_unit: Degrees per unit (0.055).
    """

    keys: tuple[str, ...]
    clicks: tuple[str, ...]
    mouse_x_units: tuple[float, ...]
    mouse_y_units: tuple[float, ...]
    deg_per_unit: float

    @property
    def dim(self) -> int:
        """Total vector length (51)."""
        raise NotImplementedError("TODO(#9): ActionSpec.dim")

    @property
    def idx_lmb(self) -> int:
        """Index of the left-click entry."""
        raise NotImplementedError("TODO(#9): ActionSpec.idx_lmb")

    @property
    def idx_rmb(self) -> int:
        """Index of the right-click entry."""
        raise NotImplementedError("TODO(#9): ActionSpec.idx_rmb")

    @property
    def mouse_x_slice(self) -> slice:
        """Slice of the mouse-x one-hot block."""
        raise NotImplementedError("TODO(#9): ActionSpec.mouse_x_slice")

    @property
    def mouse_y_slice(self) -> slice:
        """Slice of the mouse-y one-hot block."""
        raise NotImplementedError("TODO(#9): ActionSpec.mouse_y_slice")

    @property
    def mouse_x_centers_deg(self) -> np.ndarray:
        """Mouse-x bin centres in degrees (``units * deg_per_unit``)."""
        raise NotImplementedError("TODO(#9): ActionSpec.mouse_x_centers_deg")

    @property
    def mouse_y_centers_deg(self) -> np.ndarray:
        """Mouse-y bin centres in degrees (``units * deg_per_unit``)."""
        raise NotImplementedError("TODO(#9): ActionSpec.mouse_y_centers_deg")

    @classmethod
    def from_config(cls, cfg: DictConfig) -> ActionSpec:
        """Build the spec from ``cfg.actions``.

        Args:
            cfg: A validated cs2wm config.

        Returns:
            The action spec.
        """
        raise NotImplementedError("TODO(#9): ActionSpec.from_config")

    def key_index(self, name: str) -> int:
        """Return the vector index of a key name.

        Args:
            name: One of ``keys``.

        Returns:
            Index into the action vector.

        Raises:
            KeyError: If the key is unknown.
        """
        raise NotImplementedError("TODO(#9): ActionSpec.key_index")


@dataclass(frozen=True)
class DecodedAction:
    """Human-readable view of an action vector.

    Attributes:
        keys: Names of the pressed keys.
        lmb: Left click.
        rmb: Right click.
        mouse_x_deg: Centre of the active mouse-x bin, degrees (+right).
        mouse_y_deg: Centre of the active mouse-y bin, degrees (+down).
    """

    keys: tuple[str, ...]
    lmb: bool
    rmb: bool
    mouse_x_deg: float
    mouse_y_deg: float


def yaw_pitch_delta_to_mouse_deg(d_yaw: float, d_pitch: float) -> tuple[float, float]:
    """Convert Source view-angle deltas to mouse degrees.

    Args:
        d_yaw: Change in Source yaw (degrees, may wrap across +-180).
        d_pitch: Change in Source pitch (degrees).

    Returns:
        ``(x, y)`` in degrees with +x right and +y down: ``x = -wrap180(d_yaw)``, ``y = +d_pitch``.
    """
    raise NotImplementedError("TODO(#9): yaw_pitch_delta_to_mouse_deg")


def wrap180(angle_deg: ArrayLike) -> np.ndarray:
    """Wrap angles to [-180, 180).

    Args:
        angle_deg: Angle(s) in degrees.

    Returns:
        Wrapped angle(s), same shape.
    """
    raise NotImplementedError("TODO(#9): wrap180")


def counts_to_deg(counts: float, sensitivity: float, m_yaw: float = 0.022) -> float:
    """Convert raw mouse counts to degrees: ``counts * m_yaw * sensitivity``.

    Args:
        counts: Mouse counts accumulated over one frame.
        sensitivity: In-game sensitivity (``play.sensitivity``).
        m_yaw: Degrees per count at sensitivity 1.

    Returns:
        Rotation in degrees.
    """
    raise NotImplementedError("TODO(#9): counts_to_deg")


def bin_index(value_deg: ArrayLike, centers_deg: Sequence[float]) -> np.ndarray:
    """Index of the nearest bin centre. Clipped at the extremes; ties go to the lower index.

    Args:
        value_deg: Value(s) in degrees.
        centers_deg: Sorted bin centres in degrees.

    Returns:
        Integer bin index per value.
    """
    raise NotImplementedError("TODO(#9): bin_index")


def encode(
    keys: Iterable[str],
    lmb: bool,
    rmb: bool,
    mouse_x_deg: float,
    mouse_y_deg: float,
    spec: ActionSpec,
) -> np.ndarray:
    """Encode one action.

    Args:
        keys: Names of pressed keys.
        lmb: Left click.
        rmb: Right click.
        mouse_x_deg: Mouse-x delta in degrees (+right).
        mouse_y_deg: Mouse-y delta in degrees (+down).
        spec: Action layout.

    Returns:
        ``float32[51]`` vector.

    Raises:
        KeyError: If a key name is unknown.
    """
    raise NotImplementedError("TODO(#9): encode")


def encode_batch(
    key_matrix: np.ndarray,
    lmb: np.ndarray,
    rmb: np.ndarray,
    mouse_x_deg: np.ndarray,
    mouse_y_deg: np.ndarray,
    spec: ActionSpec,
) -> np.ndarray:
    """Vectorised ``encode`` for n frames.

    Args:
        key_matrix: ``bool[n, 11]`` pressed keys in spec order.
        lmb: ``bool[n]`` left click.
        rmb: ``bool[n]`` right click.
        mouse_x_deg: ``float[n]`` mouse-x degrees.
        mouse_y_deg: ``float[n]`` mouse-y degrees.
        spec: Action layout.

    Returns:
        ``float32[n, 51]`` array.
    """
    raise NotImplementedError("TODO(#9): encode_batch")


def decode(vec: np.ndarray, spec: ActionSpec) -> DecodedAction:
    """Decode an action vector.

    Args:
        vec: ``float[51]`` action.
        spec: Action layout.

    Returns:
        The decoded action.

    Raises:
        ValueError: If ``vec`` does not have length 51.
    """
    raise NotImplementedError("TODO(#9): decode")


def slot_for_weapon(weapon_name: str, cfg: DictConfig) -> str:
    """Map ``active_weapon_name`` to a slot key via ``actions.slots`` (unmatched names map to ``"3"``).

    Args:
        weapon_name: Weapon string from the demo (matched lower-cased, by substring).
        cfg: A validated cs2wm config.

    Returns:
        ``"1"``, ``"2"`` or ``"3"``.
    """
    raise NotImplementedError("TODO(#9): slot_for_weapon")
