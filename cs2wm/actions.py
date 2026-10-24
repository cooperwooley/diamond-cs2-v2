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
        return len(self.keys) + len(self.clicks) + len(self.mouse_x_units) + len(self.mouse_y_units)

    @property
    def idx_lmb(self) -> int:
        """Index of the left-click entry."""
        return len(self.keys)

    @property
    def idx_rmb(self) -> int:
        """Index of the right-click entry."""
        return len(self.keys) + 1

    @property
    def mouse_x_slice(self) -> slice:
        """Slice of the mouse-x one-hot block."""
        start = len(self.keys) + len(self.clicks)
        return slice(start, start + len(self.mouse_x_units))

    @property
    def mouse_y_slice(self) -> slice:
        """Slice of the mouse-y one-hot block."""
        start = self.mouse_x_slice.stop
        return slice(start, start + len(self.mouse_y_units))

    @property
    def mouse_x_centers_deg(self) -> np.ndarray:
        """Mouse-x bin centres in degrees (``units * deg_per_unit``)."""
        return np.asarray(self.mouse_x_units, dtype=np.float64) * self.deg_per_unit

    @property
    def mouse_y_centers_deg(self) -> np.ndarray:
        """Mouse-y bin centres in degrees (``units * deg_per_unit``)."""
        return np.asarray(self.mouse_y_units, dtype=np.float64) * self.deg_per_unit

    @classmethod
    def from_config(cls, cfg: DictConfig) -> ActionSpec:
        """Build the spec from ``cfg.actions``.

        Args:
            cfg: A validated cs2wm config.

        Returns:
            The action spec.
        """
        a = cfg.actions
        return cls(
            keys=tuple(a["keys"]),
            clicks=tuple(a.clicks),
            mouse_x_units=tuple(float(u) for u in a.mouse.x_units),
            mouse_y_units=tuple(float(u) for u in a.mouse.y_units),
            deg_per_unit=float(a.mouse.deg_per_unit),
        )

    def key_index(self, name: str) -> int:
        """Return the vector index of a key name.

        Args:
            name: One of ``keys``.

        Returns:
            Index into the action vector.

        Raises:
            KeyError: If the key is unknown.
        """
        try:
            return self.keys.index(name)
        except ValueError:
            raise KeyError(f"unknown key {name!r}; valid keys: {list(self.keys)}") from None


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
        d_yaw: Change in Source yaw (degrees, may wrap across +-180); scalar or array.
        d_pitch: Change in Source pitch (degrees); scalar or array.

    Returns:
        ``(x, y)`` in degrees with +x right and +y down: ``x = -wrap180(d_yaw)``, ``y = +d_pitch``. Arrays in,
        arrays out.
    """
    # Source yaw grows counter-clockwise (turning left), so a right turn is a negative yaw delta.
    x = -wrap180(d_yaw)
    y = np.asarray(d_pitch, dtype=np.float64)
    if x.ndim == 0 and y.ndim == 0:
        return float(x), float(y)
    return x, y


def wrap180(angle_deg: ArrayLike) -> np.ndarray:
    """Wrap angles to [-180, 180).

    Args:
        angle_deg: Angle(s) in degrees.

    Returns:
        Wrapped angle(s), same shape.
    """
    return (np.asarray(angle_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


def counts_to_deg(counts: float, sensitivity: float, m_yaw: float = 0.022) -> float:
    """Convert raw mouse counts to degrees: ``counts * m_yaw * sensitivity``.

    Args:
        counts: Mouse counts accumulated over one frame.
        sensitivity: In-game sensitivity (``play.sensitivity``).
        m_yaw: Degrees per count at sensitivity 1.

    Returns:
        Rotation in degrees.
    """
    return counts * m_yaw * sensitivity


def bin_index(value_deg: ArrayLike, centers_deg: Sequence[float]) -> np.ndarray:
    """Index of the nearest bin centre. Clipped at the extremes; ties go to the lower index.

    Args:
        value_deg: Value(s) in degrees.
        centers_deg: Sorted bin centres in degrees.

    Returns:
        int64 bin index per value (0-d for scalar input).

    Raises:
        ValueError: If any value or centre is not finite.
    """
    v = np.asarray(value_deg, dtype=np.float64)
    c = np.asarray(centers_deg, dtype=np.float64)
    if not (np.isfinite(v).all() and np.isfinite(c).all()):
        raise ValueError("bin_index requires finite values and centres")
    # Integer mouse counts and P&Z unit values land exactly on midpoints in exact arithmetic, so float error would
    # pick the side. Snapping both to a 1e-9 deg grid (far above float error, far below any real delta) makes the
    # tie exact; side="left" then sends it to the lower bin, like P&Z's min() (first minimum wins).
    edges = np.round((c[:-1] + c[1:]) / 2, 9)
    return np.searchsorted(edges, np.round(v, 9), side="left").astype(np.int64)


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
    vec = np.zeros(spec.dim, dtype=np.float32)
    for name in keys:
        vec[spec.key_index(name)] = 1.0
    vec[spec.idx_lmb] = float(lmb)
    vec[spec.idx_rmb] = float(rmb)
    vec[spec.mouse_x_slice.start + int(bin_index(mouse_x_deg, spec.mouse_x_centers_deg))] = 1.0
    vec[spec.mouse_y_slice.start + int(bin_index(mouse_y_deg, spec.mouse_y_centers_deg))] = 1.0
    return vec


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

    Raises:
        ValueError: If the input shapes disagree or any mouse value is not finite.
    """
    key_matrix = np.asarray(key_matrix)
    n = key_matrix.shape[0] if key_matrix.ndim == 2 else -1
    if key_matrix.ndim != 2 or key_matrix.shape[1] != len(spec.keys):
        raise ValueError(f"key_matrix must have shape (n, {len(spec.keys)}), got {key_matrix.shape}")
    lmb, rmb, mouse_x_deg, mouse_y_deg = (np.asarray(a) for a in (lmb, rmb, mouse_x_deg, mouse_y_deg))
    for name, arr in (("lmb", lmb), ("rmb", rmb), ("mouse_x_deg", mouse_x_deg), ("mouse_y_deg", mouse_y_deg)):
        if arr.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    out = np.zeros((n, spec.dim), dtype=np.float32)
    out[:, : len(spec.keys)] = key_matrix
    out[:, spec.idx_lmb] = lmb
    out[:, spec.idx_rmb] = rmb
    rows = np.arange(n)
    out[rows, spec.mouse_x_slice.start + bin_index(mouse_x_deg, spec.mouse_x_centers_deg)] = 1.0
    out[rows, spec.mouse_y_slice.start + bin_index(mouse_y_deg, spec.mouse_y_centers_deg)] = 1.0
    return out


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
    vec = np.asarray(vec)
    if vec.shape != (spec.dim,):
        raise ValueError(f"vec must have shape ({spec.dim},), got {vec.shape}")
    pressed = vec[: len(spec.keys)] >= 0.5
    return DecodedAction(
        keys=tuple(name for name, on in zip(spec.keys, pressed) if on),
        lmb=bool(vec[spec.idx_lmb] >= 0.5),
        rmb=bool(vec[spec.idx_rmb] >= 0.5),
        mouse_x_deg=float(spec.mouse_x_centers_deg[int(np.argmax(vec[spec.mouse_x_slice]))]),
        mouse_y_deg=float(spec.mouse_y_centers_deg[int(np.argmax(vec[spec.mouse_y_slice]))]),
    )


def slot_for_weapon(weapon_name: str | None, cfg: DictConfig) -> str | None:
    """Map ``active_weapon_name`` to a slot key via ``actions.slots`` (unmatched names map to ``"3"``).

    Args:
        weapon_name: Weapon string from the demo (matched lower-cased, by substring, ignoring a ``weapon_``
            prefix); ``None`` or empty when unknown.
        cfg: A validated cs2wm config.

    Returns:
        ``"1"``, ``"2"`` or ``"3"``; ``None`` if ``weapon_name`` is missing.
    """
    if not weapon_name:
        return None
    name = weapon_name.lower().removeprefix("weapon_")
    slots = cfg.actions.slots
    for slot in ("1", "2"):
        if any(sub in name for sub in slots[slot]):
            return slot
    return str(slots.default)
