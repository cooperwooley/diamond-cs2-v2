"""segment: lives, spectate schedule and episode split for one demo (#10).

Turns ``player_spawn``/``player_death`` events into lives, chooses one in-eye point of view at a time (switching only
at the current target's death), drops ``segment.settle_ticks`` after each spawn and before each death, splits segments
into episodes of at most ``episodes.max_frames`` frames (remainders below ``episodes.min_frames`` are dropped) and
writes ``plans/<demo_id>/render_plan.json`` for the render stage. Frame/tick helpers live in ``cs2wm.timeline``.

Scheduling rules (all values from ``config/cs2wm.yaml``):

- A life ``[spawn, death]`` can be viewed on ``[spawn + settle, death - settle]``. Shorter lives are skipped.
- After a segment ends the camera is switched at once, but the next segment starts ``settle`` ticks later, so the
  switch transition is never recorded (the settle time of #14 is applied to switches as well as to spawns/deaths).
- A segment is chunked into episodes of ``max_frames`` labeled frames. A segment is rendered once, continuously:
  episode ``c`` is the slice of frames ``[c * max_frames, c * max_frames + n_frames]``, so the extra frame of one
  episode is the first frame of the next (the lag-free convention: ``n_frames`` labeled + 1 rendered).
  The last episode of a segment may be shorter; if it has fewer than ``min_frames`` frames it is dropped and the
  recording ends at the last kept episode. Segments left without an episode are not rendered.

``render_plan.json`` (schema_version 1). Ticks are demo ticks unless the key starts with ``mirv_``; a ``mirv_`` tick
is ``demo tick + tick_offset`` and is the tick to pass to ``mirv_cmd addAtTick``::

    {
      "schema_version": 1,
      "demo_id": "dm-0001",
      "demo_file": "dm-0001.dem",
      "cs2_build": 1234567 | null,            # installed Steam build id at planning time (null if unknown)
      "parser_version": "0.42.0",
      "header": {"map_name": "de_dust2", ...},  # demo header, strings
      "demo_end_tick": 90000,                 # last tick of the demo (render timeout, final quit)
      "tick_offset": 0,                       # render.tick_offset
      "settle_ticks": 32,
      "timing": {"tickrate": 64, "fps": 16, "ticks_per_frame": 4},
      "meta": {"policy": "longest_remaining", "player": null, "seed": 0, "kind": "dm",
               "diagnostics": {"duplicate_spawns": 0, "deaths_without_spawn": 0}},
      "segments": [{
        "segment_idx": 0,
        "target": {"name": "Bot A", "user_id": 3, "steamid": 0, "entity_id": 7 | null},
                                              # entity_id: the player's tick-table value at/after spawn (CONFIRM(#14));
                                              # null if the demo gives none; the likely spec_player argument
        "life": {"spawn_tick": 1000, "death_tick": 9000},
        "record_start_tick": 1032,            # T0 of the first episode (frame 0)
        "record_end_tick": 5032,              # tick of the extra frame of the last episode
        "mirv_record_start_tick": 1032, "mirv_record_end_tick": 5032, "mirv_spec_tick": 1000,
        "labeled_frames": 1000,               # sum of the episodes' n_frames
        "rendered_frames": 1001,              # labeled_frames + 1
        "episodes": [{
          "episode_id": "dm-dm-0001-s000-c00", "kind": "dm", "chunk_idx": 0,
          "start_tick": 1032, "end_tick": 5032, "n_frames": 1000, "rendered_frames": 1001,
          "first_frame": 0                    # index of frame 0 within the segment's rendered frames
        }]
      }],
      "mirv_cmd_schedule": [                  # execution order: record_end, then spec, then record_start per tick
        {"mirv_tick": 1000, "demo_tick": 1000, "action": "spec", "segment_idx": 0,
         "target": {"name": ..., "user_id": ..., "steamid": ..., "entity_id": ...}},
        {"mirv_tick": 1032, "demo_tick": 1032, "action": "record_start", "segment_idx": 0},
        {"mirv_tick": 5032, "demo_tick": 5032, "action": "record_end", "segment_idx": 0}
      ],
      "totals": {"segments": 1, "episodes": 1, "labeled_frames": 1000, "rendered_frames": 1001}
    }

``render_plan.json`` is enough to write the HLAE cfg and to rebuild the ``EpisodeSpec`` list (``episodes_from_plan``)
without parsing the demo again. Whether the frame at ``record_end_tick`` is captured is measured in #14.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from cs2wm import demo
from cs2wm.config import ticks_per_frame
from cs2wm.timeline import EpisodeSpec, make_episode_id

log = logging.getLogger(__name__)

PLAN_SCHEMA_VERSION = 1
PLAN_EVENTS = ("player_spawn", "player_death", "player_disconnect")

# Order of the mirv_cmd entries that share a tick: end the old recording, switch the target, start the new one.
_ACTION_ORDER = {"record_end": 0, "spec": 1, "record_start": 2}


@dataclass(frozen=True)
class Life:
    """One spawn-to-death interval of a player.

    Attributes:
        player_name: Display name.
        user_id: ``user_id`` in the demo.
        steamid: Steamid (may be 0 or small for bots).
        spawn_tick: Tick of the spawn.
        death_tick: Tick of the death, or the demo end if still alive.
        entity_id: Entity/slot id of the player when the demo provides it (CONFIRM(#14)), else None.
    """

    player_name: str
    user_id: int
    steamid: int
    spawn_tick: int
    death_tick: int
    entity_id: int | None = None


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


def _opt_int(value: Any) -> int | None:
    """Return ``int(value)``, or None for a missing/NaN value."""
    return None if value is None or pd.isna(value) else int(value)


def find_lives(
    spawns: pd.DataFrame,
    deaths: pd.DataFrame,
    demo_end_tick: int,
    disconnects: pd.DataFrame | None = None,
    diagnostics: dict[str, int] | None = None,
) -> list[Life]:
    """Pair spawns and deaths into lives.

    Robust to duplicate spawns, a death with no spawn, disconnects and the demo ending while the player is alive.
    Players are told apart by ``user_id`` only (steamids of bots are 0 or shared, demoparser2 #358).

    - A spawn while the player is already alive is a duplicate and ignored (the life keeps its first spawn).
    - A death (or disconnect) with no open life is ignored: the spawn is unknown, e.g. the demo chunk started
      mid-life, so there is no valid start tick.
    - A disconnect closes an open life like a death; without it the life would run to the demo end.
    - A life still open at the end is closed at ``demo_end_tick``. Events after ``demo_end_tick`` are ignored, and
      lives of zero length are dropped. At one tick a death/disconnect is processed before a spawn.

    Args:
        spawns: ``player_spawn`` events, columns ``tick, user_id`` and optionally ``name, steamid, entity_id``
            (see ``cs2wm.demo.player_event_table``).
        deaths: ``player_death`` events (the victim), same columns.
        demo_end_tick: Last tick of the demo.
        disconnects: Optional ``player_disconnect`` events, same columns.
        diagnostics: Optional dict filled with ``duplicate_spawns`` and ``deaths_without_spawn`` counts (also logged),
            so real demos show whether these cases occur.

    Returns:
        Lives sorted by spawn tick (then ``user_id``).
    """
    events: list[tuple[int, int, str, dict[str, Any]]] = []  # (tick, order, kind, row); order 0 closes, 1 opens
    for order, kind, table in ((1, "spawn", spawns), (0, "death", deaths), (0, "disconnect", disconnects)):
        if table is None or table.empty:
            continue
        for row in table.to_dict("records"):
            if int(row["tick"]) <= demo_end_tick:
                events.append((int(row["tick"]), order, kind, row))
    events.sort(key=lambda e: (e[0], e[1]))
    dup_spawns = orphan_deaths = 0

    lives: list[Life] = []
    open_lives: dict[int, tuple[int, dict[str, Any]]] = {}  # user_id -> (spawn_tick, spawn row)

    def close(user_id: int, tick: int) -> None:
        spawn_tick, row = open_lives.pop(user_id)
        if tick > spawn_tick:
            name = row.get("name")
            lives.append(
                Life(
                    player_name=name if isinstance(name, str) else "",
                    user_id=user_id,
                    steamid=_opt_int(row.get("steamid")) or 0,
                    spawn_tick=spawn_tick,
                    death_tick=tick,
                    entity_id=_opt_int(row.get("entity_id")),
                )
            )

    for tick, order, kind, row in events:
        user_id = int(row["user_id"])
        if order == 1:
            if user_id in open_lives:
                dup_spawns += 1
            else:
                open_lives[user_id] = (tick, row)
        elif user_id in open_lives:
            close(user_id, tick)
        elif kind == "death":
            orphan_deaths += 1
    for user_id in list(open_lives):
        close(user_id, demo_end_tick)
    counts = {"duplicate_spawns": dup_spawns, "deaths_without_spawn": orphan_deaths}
    if diagnostics is not None:
        diagnostics.update(counts)
    if dup_spawns or orphan_deaths:
        log.info("find_lives: ignored %d duplicate spawn(s) and %d death(s) with no spawn", dup_spawns, orphan_deaths)
    return sorted(lives, key=lambda l: (l.spawn_tick, l.user_id, l.death_tick))


def attach_entity_ids(lives: Sequence[Life], ticks: pd.DataFrame) -> list[Life]:
    """Fill ``Life.entity_id`` from the tick table: each player's value at, or nearest after, its spawn tick.

    A value found in ``ticks`` replaces one that came from the events; lives without a tick-table value keep theirs.
    A warning is logged once if no life ends up with an entity id.

    Args:
        lives: Output of ``find_lives``.
        ticks: Per-tick table with ``tick``, ``user_id`` and (optionally) ``entity_id`` columns.

    Returns:
        New lives with ``entity_id`` set where known.
    """
    out = list(lives)
    if demo.TICK_ENTITY_ID in ticks.columns and not ticks.empty:
        table = ticks.dropna(subset=[demo.TICK_USER_ID, demo.TICK_ENTITY_ID])
        by_user = {int(uid): g.sort_values("tick") for uid, g in table.groupby(demo.TICK_USER_ID)}
        for i, life in enumerate(out):
            g = by_user.get(life.user_id)
            if g is None:
                continue
            pos = int(np.searchsorted(g["tick"].to_numpy(), life.spawn_tick, side="left"))
            if pos < len(g):
                out[i] = dataclasses.replace(life, entity_id=int(g[demo.TICK_ENTITY_ID].iloc[pos]))
    if out and all(life.entity_id is None for life in out):
        log.warning("no entity_id for any player (tick table and events lack it); plan targets have entity_id null")
    return out


def plan_spectate(
    lives: Sequence[Life],
    policy: str,
    cfg: DictConfig,
    rng: np.random.Generator,
    player: str | int | None = None,
) -> list[ViewSegment]:
    """Choose whom to spectate when.

    Segments never overlap and switch only at the current target's death. Policies: ``longest_remaining`` (bots:
    the alive player with the most life left, ties broken round-robin for POV diversity) and ``fixed_player``
    (human sessions, requires ``player``, a name or a numeric ``user_id``).

    A life is viewable from ``spawn + settle_ticks`` to ``death - settle_ticks``. After a segment the next one starts
    ``settle_ticks`` later (camera switch), so ``longest_remaining`` considers the lives alive at that tick. Ties in
    remaining life go to the player viewed least often so far; remaining ties are drawn with ``rng``.

    Args:
        lives: Output of ``find_lives``.
        policy: ``longest_remaining`` or ``fixed_player``.
        cfg: A validated cs2wm config (``segment.settle_ticks``).
        rng: Seeded generator, so plans are deterministic.
        player: For ``fixed_player``: the player's name, or a numeric ``user_id`` if no player has that name.

    Returns:
        Non-overlapping view segments.

    Raises:
        ValueError: On an unknown policy, ``fixed_player`` without ``player``, a ``player`` with no lives, or a name
            shared by several ``user_id`` values (pass the ``user_id`` instead).
    """
    if policy not in ("longest_remaining", "fixed_player"):
        raise ValueError(f"unknown segment policy {policy!r}")
    settle = int(cfg.segment.settle_ticks)

    if policy == "fixed_player":
        if player is None or player == "":
            raise ValueError("policy 'fixed_player' needs a player name")
        player = str(player)
        mine = [life for life in lives if life.player_name == player]
        if not mine and player.strip().isdigit():
            mine = [life for life in lives if life.user_id == int(player)]
        if not mine:
            raise ValueError(f"no lives for player {player!r}")
        user_ids = sorted({life.user_id for life in mine})
        if len(user_ids) > 1:
            raise ValueError(f"player name {player!r} is shared by user_ids {user_ids}; pass one user_id as --player")
        windows = [(l.spawn_tick + settle, l.death_tick - settle, l) for l in sorted(mine, key=lambda l: l.spawn_tick)]
        return [
            ViewSegment(life, start, end, idx)
            for idx, (start, end, life) in enumerate(w for w in windows if w[1] > w[0])
        ]

    windows = [(l.spawn_tick + settle, l.death_tick - settle, l) for l in lives]
    windows = [w for w in windows if w[1] > w[0]]
    segments: list[ViewSegment] = []
    if not windows:
        return segments
    views: Counter[int] = Counter()
    t = min(w[0] for w in windows)
    while True:
        alive = [w for w in windows if w[0] <= t < w[1]]
        if not alive:
            later = [w[0] for w in windows if w[0] > t]
            if not later:
                break
            t = min(later)
            continue
        longest = max(w[1] for w in alive)
        tied = sorted((w for w in alive if w[1] == longest), key=lambda w: w[2].user_id)
        fewest = min(views[w[2].user_id] for w in tied)
        tied = [w for w in tied if views[w[2].user_id] == fewest]
        pick = tied[0] if len(tied) == 1 else tied[int(rng.integers(len(tied)))]
        segments.append(ViewSegment(pick[2], t, longest, len(segments)))
        views[pick[2].user_id] += 1
        t = longest + settle
    return segments


def split_episodes(segments: Sequence[ViewSegment], demo_id: str, kind: str, cfg: DictConfig) -> list[EpisodeSpec]:
    """Split segments into episodes of at most ``episodes.max_frames`` frames.

    A segment of ``(end - start) // ticks_per_frame`` labeled frames is cut into chunks of ``max_frames``; chunk ``c``
    starts at frame ``c * max_frames``. Each episode also needs the frame at its ``end_tick`` rendered (lag-free
    convention), which exists because it lies inside the segment.

    Args:
        segments: Output of ``plan_spectate``.
        demo_id: Demo id.
        kind: Episode kind (``dm``, ``human``, ``smoke``).
        cfg: A validated cs2wm config.

    Returns:
        Episodes; remainders shorter than ``episodes.min_frames`` are dropped.
    """
    tpf = ticks_per_frame(cfg)
    max_frames, min_frames = int(cfg.episodes.max_frames), int(cfg.episodes.min_frames)
    episodes: list[EpisodeSpec] = []
    for seg in segments:
        total = (seg.end_tick - seg.start_tick) // tpf
        for chunk, first in enumerate(range(0, total, max_frames)):
            n = min(max_frames, total - first)
            if n < min_frames:
                break
            episodes.append(
                EpisodeSpec(
                    episode_id=make_episode_id(kind, demo_id, seg.segment_idx, chunk),
                    demo_id=demo_id,
                    kind=kind,
                    player_name=seg.target.player_name,
                    user_id=seg.target.user_id,
                    steamid=seg.target.steamid,
                    start_tick=seg.start_tick + first * tpf,
                    n_frames=n,
                    ticks_per_frame=tpf,
                    segment_idx=seg.segment_idx,
                    chunk_idx=chunk,
                )
            )
    return episodes


def _target(life: Life) -> dict[str, Any]:
    """Target identifiers of a life, as stored in the plan."""
    return {
        "name": life.player_name,
        "user_id": life.user_id,
        "steamid": life.steamid,
        "entity_id": life.entity_id,
    }


def build_render_plan(
    demo_id: str,
    segments: Sequence[ViewSegment],
    episodes: Sequence[EpisodeSpec],
    cfg: DictConfig,
    header: Mapping[str, str],
    cs2_build: int | None = None,
    meta: Mapping[str, Any] | None = None,
    demo_end_tick: int | None = None,
) -> dict[str, Any]:
    """Build the ``render_plan.json`` content (see the module docstring for the schema).

    Only segments that kept at least one episode are rendered. The recording of a segment runs from its first
    episode's ``start_tick`` to its last episode's ``end_tick``; the target is switched ``settle_ticks`` before the
    recording starts, but never before the previous recording ended.

    Args:
        demo_id: Demo id.
        segments: View segments.
        episodes: Episodes cut from the segments.
        cfg: A validated cs2wm config.
        header: Demo header from ``cs2wm.demo.header``.
        cs2_build: Installed CS2 build id, if known.
        meta: Extra provenance (policy, player, seed, kind, parser version, diagnostics).
        demo_end_tick: Last tick of the demo.

    Returns:
        A JSON-serialisable dict.

    Raises:
        ValueError: If ``render.tick_offset`` would schedule a command at a negative tick.
    """
    tpf = ticks_per_frame(cfg)
    offset, settle = int(cfg.render.tick_offset), int(cfg.segment.settle_ticks)
    meta = dict(meta or {})
    by_segment: dict[int, list[EpisodeSpec]] = {}
    for ep in episodes:
        by_segment.setdefault(ep.segment_idx, []).append(ep)

    plan_segments: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    prev_end = 0
    for seg in sorted(segments, key=lambda s: s.segment_idx):
        eps = sorted(by_segment.get(seg.segment_idx, ()), key=lambda e: e.chunk_idx)
        if not eps:
            continue
        rec_start, rec_end = eps[0].start_tick, eps[-1].end_tick
        spec_tick = max(rec_start - settle, prev_end)
        labeled = sum(e.n_frames for e in eps)
        plan_segments.append(
            {
                "segment_idx": seg.segment_idx,
                "target": _target(seg.target),
                "life": {"spawn_tick": seg.target.spawn_tick, "death_tick": seg.target.death_tick},
                "record_start_tick": rec_start,
                "record_end_tick": rec_end,
                "mirv_record_start_tick": rec_start + offset,
                "mirv_record_end_tick": rec_end + offset,
                "mirv_spec_tick": spec_tick + offset,
                "labeled_frames": labeled,
                "rendered_frames": labeled + 1,
                "episodes": [
                    {
                        "episode_id": e.episode_id,
                        "kind": e.kind,
                        "chunk_idx": e.chunk_idx,
                        "start_tick": e.start_tick,
                        "end_tick": e.end_tick,
                        "n_frames": e.n_frames,
                        "rendered_frames": e.n_frames + 1,
                        "first_frame": (e.start_tick - rec_start) // tpf,
                    }
                    for e in eps
                ],
            }
        )
        for action, tick in (("spec", spec_tick), ("record_start", rec_start), ("record_end", rec_end)):
            entry: dict[str, Any] = {
                "mirv_tick": tick + offset,
                "demo_tick": tick,
                "action": action,
                "segment_idx": seg.segment_idx,
            }
            if action == "spec":
                entry["target"] = _target(seg.target)
            schedule.append(entry)
        prev_end = rec_end

    schedule.sort(key=lambda e: (e["mirv_tick"], _ACTION_ORDER[e["action"]], e["segment_idx"]))
    if schedule and schedule[0]["mirv_tick"] < 0:
        raise ValueError(f"render.tick_offset={offset} schedules mirv_cmd at tick {schedule[0]['mirv_tick']} < 0")
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "demo_id": demo_id,
        "demo_file": f"{demo_id}.dem",
        "cs2_build": cs2_build,
        "demo_end_tick": demo_end_tick,
        "parser_version": meta.pop("parser_version", None),
        "header": dict(header),
        "tick_offset": offset,
        "settle_ticks": settle,
        "timing": {"tickrate": int(cfg.timing.tickrate), "fps": int(cfg.timing.fps), "ticks_per_frame": tpf},
        "meta": meta,
        "segments": plan_segments,
        "mirv_cmd_schedule": schedule,
        "totals": {
            "segments": len(plan_segments),
            "episodes": sum(len(s["episodes"]) for s in plan_segments),
            "labeled_frames": sum(s["labeled_frames"] for s in plan_segments),
            "rendered_frames": sum(s["rendered_frames"] for s in plan_segments),
        },
    }


def write_render_plan(
    path: Path,
    demo_id: str,
    segments: Sequence[ViewSegment],
    episodes: Sequence[EpisodeSpec],
    cfg: DictConfig,
    header: Mapping[str, str],
    cs2_build: int | None = None,
    meta: Mapping[str, Any] | None = None,
    demo_end_tick: int | None = None,
) -> None:
    """Write ``render_plan.json``: demo id, CS2 build, tick offset, segments with target identifiers, the
    ``mirv_cmd`` schedule ticks and planned frame counts.

    The file is written to ``<name>.tmp`` and renamed, so a crash never leaves a half-written plan.

    Args:
        path: Output file.
        demo_id: Demo id.
        segments: View segments.
        episodes: Episodes cut from the segments.
        cfg: A validated cs2wm config.
        header: Demo header from ``cs2wm.demo.header``.
        cs2_build: Installed CS2 build id, if known (added to the original signature).
        meta: Extra provenance, see ``build_render_plan`` (added to the original signature).
        demo_end_tick: Last tick of the demo (added to the original signature).
    """
    plan = build_render_plan(demo_id, segments, episodes, cfg, header, cs2_build, meta, demo_end_tick)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_render_plan(path: Path) -> dict[str, Any]:
    """Load a ``render_plan.json``.

    Args:
        path: Plan file.

    Returns:
        The plan dict.

    Raises:
        ValueError: If the schema version is not supported.
    """
    with open(path, encoding="utf-8") as f:
        plan = json.load(f)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported render plan schema {plan.get('schema_version')!r}")
    return plan


def episodes_from_plan(plan: Mapping[str, Any]) -> list[EpisodeSpec]:
    """Rebuild the episode specs from a plan, so later stages need not re-parse the demo.

    Args:
        plan: Output of ``read_render_plan``.

    Returns:
        Episodes in plan order.
    """
    tpf = plan["timing"]["ticks_per_frame"]
    return [
        EpisodeSpec(
            episode_id=e["episode_id"],
            demo_id=plan["demo_id"],
            kind=e["kind"],
            player_name=seg["target"]["name"],
            user_id=seg["target"]["user_id"],
            steamid=seg["target"]["steamid"],
            start_tick=e["start_tick"],
            n_frames=e["n_frames"],
            ticks_per_frame=tpf,
            segment_idx=seg["segment_idx"],
            chunk_idx=e["chunk_idx"],
        )
        for seg in plan["segments"]
        for e in seg["episodes"]
    ]


def _seed(cfg: DictConfig, demo_id: str) -> int:
    """Deterministic per-demo seed from ``segment.seed`` and the demo id (not Python's salted ``hash``)."""
    return int(hashlib.sha1(f"{cfg.segment.seed}:{demo_id}".encode()).hexdigest()[:8], 16)


def run(demo_path: Path, cfg: DictConfig, policy: str | None = None, player: str | None = None) -> Path:
    """CLI stage: parse the demo, plan the views and write the render plan.

    The episode kind is the name of the demo's parent folder if it is one of ``episodes.kinds`` (the
    ``demos/<kind>/<demo_id>.dem`` layout), else ``human`` for ``fixed_player`` and ``dm`` otherwise.

    Args:
        demo_path: Path to the ``.dem`` file.
        cfg: A validated cs2wm config.
        policy: Override for ``segment.policy``.
        player: Player name (or numeric ``user_id``) for ``fixed_player``.

    Returns:
        Path of the written ``render_plan.json``.

    Raises:
        ValueError: If the demo has no ticks, or the policy cannot be applied (see ``plan_spectate``).
    """
    demo_path = Path(demo_path)
    demo_id = demo_path.stem
    policy = policy or cfg.segment.policy
    kind = demo_path.parent.name
    if kind not in cfg.episodes.kinds:
        kind = "human" if policy == "fixed_player" else "dm"

    hdr = demo.header(demo_path)
    events = demo.load_events(demo_path, PLAN_EVENTS, list(cfg.segment.event_player_props))
    try:
        disconnects = demo.player_event_table(events["player_disconnect"])
    except KeyError:
        log.warning("player_disconnect has no user_id column; disconnects are not used to close lives")
        disconnects = None
    # One tick-table parse gives the demo end and the targets' entity ids (CONFIRM(#14): entity_id as a tick prop).
    try:
        ticks = demo.load_ticks(demo_path, [demo.TICK_ENTITY_ID])
    except Exception:
        log.warning("parse_ticks rejected %s; entity ids come from events only", demo.TICK_ENTITY_ID, exc_info=True)
        ticks = demo.load_ticks(demo_path, [])
    if ticks.empty:
        raise ValueError(f"{demo_path} has no ticks")
    demo_end = int(ticks["tick"].max())
    diagnostics: dict[str, int] = {}
    lives = find_lives(
        demo.player_event_table(events["player_spawn"]),
        demo.player_event_table(events["player_death"]),
        demo_end,
        disconnects,
        diagnostics,
    )
    lives = attach_entity_ids(lives, ticks)
    seed = _seed(cfg, demo_id)
    segments = plan_spectate(lives, policy, cfg, np.random.default_rng(seed), player)
    episodes = split_episodes(segments, demo_id, kind, cfg)

    try:
        build: int | None = demo.cs2_build_id(cfg.paths.cs2_dir)
    except FileNotFoundError:
        build = cfg.generate.expected_build  # CS2 is not installed on this machine (e.g. planning in WSL)
    meta = {
        "policy": policy,
        "player": player,
        "seed": int(cfg.segment.seed),
        "kind": kind,
        "parser_version": demo.parser_version(),
        "diagnostics": diagnostics,
    }
    out = Path(cfg.paths.plans) / demo_id / "render_plan.json"
    write_render_plan(out, demo_id, segments, episodes, cfg, hdr, build, meta, demo_end)
    if not episodes:
        log.warning("%s: plan has zero episodes (%d lives, %d segments)", demo_id, len(lives), len(segments))
    log.info("%s: %d lives, %d segments, %d episodes -> %s", demo_id, len(lives), len(segments), len(episodes), out)
    return out
