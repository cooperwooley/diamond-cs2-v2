"""The frame/tick convention, defined once (#10). See docs/PIPELINE.md.

Frame ``k`` of an episode is rendered at tick ``T_k = T0 + k * ticks_per_frame``. Action ``k`` covers the ticks
``(T_k, T_{k+1}]``, so an event at tick ``t`` belongs to frame ``ceil((t - T0) / ticks_per_frame) - 1``. Labeling the
last frame needs one extra rendered frame, ``T_n``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class EpisodeSpec:
    """One episode (at most ``episodes.max_frames`` frames of one player's view).

    Attributes:
        episode_id: ``<kind>-<demo_id>-s<segment:03d>-c<chunk:02d>``.
        demo_id: Parent demo.
        kind: ``dm``, ``human`` or ``smoke``.
        player_name: Spectated player.
        user_id: Spectated player's ``user_id`` in the demo.
        steamid: Spectated player's steamid (may be 0 for bots).
        start_tick: ``T0``, the tick of frame 0.
        n_frames: Labeled frames (one more is rendered).
        ticks_per_frame: Ticks per frame (4).
        segment_idx: Index of the view segment within the demo.
        chunk_idx: Index of the episode within its segment.
    """

    episode_id: str
    demo_id: str
    kind: str
    player_name: str
    user_id: int
    steamid: int
    start_tick: int
    n_frames: int
    ticks_per_frame: int
    segment_idx: int
    chunk_idx: int

    def frame_tick(self, k: int) -> int:
        """Return ``T_k`` for frame ``k`` (``0 <= k <= n_frames``)."""
        if not 0 <= k <= self.n_frames:
            raise IndexError(f"frame {k} outside [0, {self.n_frames}]")
        return self.start_tick + k * self.ticks_per_frame

    def window(self, k: int) -> tuple[int, int]:
        """Return the open-closed tick window ``(T_k, T_{k+1}]`` covered by action ``k``."""
        if not 0 <= k < self.n_frames:
            raise IndexError(f"action {k} outside [0, {self.n_frames})")
        return self.frame_tick(k), self.frame_tick(k + 1)

    @property
    def end_tick(self) -> int:
        """Tick of frame ``n_frames``, the extra frame needed to label frame ``n_frames - 1``."""
        return self.frame_tick(self.n_frames)


def frame_to_tick(k: ArrayLike, start_tick: int, ticks_per_frame: int) -> np.ndarray:
    """Tick at which frame ``k`` is rendered: ``start_tick + k * ticks_per_frame``.

    Args:
        k: Frame index or indices.
        start_tick: ``T0``.
        ticks_per_frame: Ticks per frame.

    Returns:
        Integer tick(s).
    """
    return start_tick + np.asarray(k, dtype=np.int64) * ticks_per_frame


def tick_to_frame(tick: ArrayLike, start_tick: int, ticks_per_frame: int) -> np.ndarray:
    """Frame whose action window contains ``tick``: ``ceil((t - T0) / ticks_per_frame) - 1``.

    Args:
        tick: Tick(s) of an event.
        start_tick: ``T0``.
        ticks_per_frame: Ticks per frame.

    Returns:
        Integer frame index (negative or ``>= n_frames`` when outside the episode).
    """
    # Integer ceil: -((T0 - t) // tpf) == ceil((t - T0) / tpf); floats could misplace exact boundaries.
    return -((start_tick - np.asarray(tick, dtype=np.int64)) // ticks_per_frame) - 1


def make_episode_id(kind: str, demo_id: str, segment_idx: int, chunk_idx: int) -> str:
    """Build the stable episode id ``<kind>-<demo_id>-s<segment:03d>-c<chunk:02d>``.

    Args:
        kind: Episode kind.
        demo_id: Parent demo id.
        segment_idx: Segment index.
        chunk_idx: Chunk index within the segment.

    Returns:
        The episode id.
    """
    return f"{kind}-{demo_id}-s{segment_idx:03d}-c{chunk_idx:02d}"
