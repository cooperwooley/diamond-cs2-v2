"""segment: lives, spectate schedule and episode split for one demo (#10).

Turns ``player_spawn``/``player_death`` events into lives, chooses one in-eye point of view at a time (switching only
at the current target's death), drops ``segment.settle_ticks`` after each spawn and before each death, splits segments
into episodes of at most ``episodes.max_frames`` frames (remainders below ``episodes.min_frames`` are dropped) and
writes ``plans/<demo_id>/render_plan.json`` for the render stage. Frame/tick helpers live in ``cs2wm.timeline``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from cs2wm.timeline import EpisodeSpec


@dataclass(frozen=True)
class Life:
    """One spawn-to-death interval of a player.

    Attributes:
        player_name: Display name.
        user_id: ``user_id`` in the demo.
        steamid: Steamid (may be 0 or small for bots).
        spawn_tick: Tick of the spawn.
        death_tick: Tick of the death, or the demo end if still alive.
    """

    player_name: str
    user_id: int
    steamid: int
    spawn_tick: int
    death_tick: int


@dataclass(frozen=True)
class ViewSegment:
    """A stretch of one player's view to render.

    Attributes:
        target: The spectated life.
        start_tick: First tick of the segment (after the settle time).
        end_tick: Last tick of the segment (before the settle time).
        segment_idx: Index within the demo.
    """

    target: Life
    start_tick: int
    end_tick: int
    segment_idx: int


def find_lives(spawns: pd.DataFrame, deaths: pd.DataFrame, demo_end_tick: int) -> list[Life]:
    """Pair spawns and deaths into lives.

    Robust to duplicate spawns, a death with no spawn, disconnects and the demo ending while the player is alive.

    Args:
        spawns: ``player_spawn`` events.
        deaths: ``player_death`` events.
        demo_end_tick: Last tick of the demo.

    Returns:
        Lives sorted by spawn tick.
    """
    # TODO(#10): pair per user_id; ignore duplicate spawns; close open lives at demo_end_tick.
    raise NotImplementedError("TODO(#10): find_lives")


def plan_spectate(
    lives: Sequence[Life],
    policy: str,
    cfg: DictConfig,
    rng: np.random.Generator,
    player: str | None = None,
) -> list[ViewSegment]:
    """Choose whom to spectate when.

    Segments never overlap and switch only at the current target's death. Policies: ``longest_remaining`` (bots:
    the alive player with the most life left, ties broken round-robin for POV diversity) and ``fixed_player``
    (human sessions, requires ``player``).

    Args:
        lives: Output of ``find_lives``.
        policy: ``longest_remaining`` or ``fixed_player``.
        cfg: A validated cs2wm config (``segment.settle_ticks``).
        rng: Seeded generator, so plans are deterministic.
        player: Player name for ``fixed_player``.

    Returns:
        Non-overlapping view segments.

    Raises:
        ValueError: On an unknown policy or ``fixed_player`` without ``player``.
    """
    # TODO(#10): greedy schedule with settle_ticks trimmed at both ends of every life.
    raise NotImplementedError("TODO(#10): plan_spectate")


def split_episodes(segments: Sequence[ViewSegment], demo_id: str, kind: str, cfg: DictConfig) -> list[EpisodeSpec]:
    """Split segments into episodes of at most ``episodes.max_frames`` frames.

    Args:
        segments: Output of ``plan_spectate``.
        demo_id: Demo id.
        kind: Episode kind (``dm``, ``human``, ``smoke``).
        cfg: A validated cs2wm config.

    Returns:
        Episodes; remainders shorter than ``episodes.min_frames`` are dropped.
    """
    # TODO(#10): chunk each segment in ticks_per_frame steps; ids via timeline.make_episode_id.
    raise NotImplementedError("TODO(#10): split_episodes")


def write_render_plan(
    path: Path,
    demo_id: str,
    segments: Sequence[ViewSegment],
    episodes: Sequence[EpisodeSpec],
    cfg: DictConfig,
    header: Mapping[str, str],
) -> None:
    """Write ``render_plan.json``: demo id, CS2 build, tick offset, segments with target identifiers, the
    ``mirv_cmd`` schedule ticks and planned frame counts.

    Args:
        path: Output file.
        demo_id: Demo id.
        segments: View segments.
        episodes: Episodes cut from the segments.
        cfg: A validated cs2wm config.
        header: Demo header from ``cs2wm.demo.header``.
    """
    # TODO(#10): serialise atomically (temp file + rename).
    raise NotImplementedError("TODO(#10): write_render_plan")


def run(demo_path: Path, cfg: DictConfig, policy: str | None = None, player: str | None = None) -> Path:
    """CLI stage: parse the demo, plan the views and write the render plan.

    Args:
        demo_path: Path to the ``.dem`` file.
        cfg: A validated cs2wm config.
        policy: Override for ``segment.policy``.
        player: Player name for ``fixed_player``.

    Returns:
        Path of the written ``render_plan.json``.
    """
    # TODO(#10): demo.load_events -> find_lives -> plan_spectate -> split_episodes -> write_render_plan.
    raise NotImplementedError("TODO(#10): segment.run")
