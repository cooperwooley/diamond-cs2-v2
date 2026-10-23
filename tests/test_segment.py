"""Tests for segment (#10): lives, spectate schedule, episodes and render_plan.json, on synthetic event tables."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cs2wm import demo
from cs2wm.config import load_config
from cs2wm.pipeline import segment
from cs2wm.pipeline.segment import Life, ViewSegment, find_lives, plan_spectate, split_episodes
from tests.test_demo import FakeParser

SETTLE = 10
CFG = load_config(overrides=[f"segment.settle_ticks={SETTLE}", "render.tick_offset=0"])
TPF = 4
END = 100_000


def events(*rows: tuple[int, int, str]) -> pd.DataFrame:
    """Event table with the canonical columns from ``(tick, user_id, name)`` rows (steamid 0, like bots)."""
    return pd.DataFrame(
        {
            "tick": [r[0] for r in rows],
            "user_id": [r[1] for r in rows],
            "name": [r[2] for r in rows],
            "steamid": [0] * len(rows),
        },
        columns=["tick", "user_id", "name", "steamid"],
    )


def life(uid: int, spawn: int, death: int, name: str | None = None) -> Life:
    return Life(name or f"bot{uid}", uid, 0, spawn, death)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---- find_lives -----------------------------------------------------------------------------------------------


def test_find_lives_pairs_spawns_and_deaths_per_user() -> None:
    spawns = events((100, 1, "A"), (150, 2, "B"), (500, 1, "A"))
    deaths = events((400, 1, "A"), (300, 2, "B"), (900, 1, "A"))
    lives = find_lives(spawns, deaths, END)
    assert [(l.user_id, l.spawn_tick, l.death_tick) for l in lives] == [(1, 100, 400), (2, 150, 300), (1, 500, 900)]
    assert lives[0].player_name == "A" and lives[0].steamid == 0


def test_find_lives_ignores_duplicate_spawns() -> None:
    lives = find_lives(events((100, 1, "A"), (100, 1, "A"), (200, 1, "A")), events((400, 1, "A")), END)
    assert [(l.spawn_tick, l.death_tick) for l in lives] == [(100, 400)]


def test_find_lives_ignores_death_with_no_spawn() -> None:
    # The first death belongs to a life that began before the demo; its start is unknown.
    lives = find_lives(events((500, 1, "A")), events((50, 1, "A"), (900, 1, "A")), END)
    assert [(l.spawn_tick, l.death_tick) for l in lives] == [(500, 900)]
    assert find_lives(events(), events((50, 1, "A")), END) == []


def test_find_lives_disconnect_closes_the_life() -> None:
    lives = find_lives(events((100, 1, "A"), (100, 2, "B")), events(), END, events((600, 1, "A")))
    assert {(l.user_id, l.death_tick) for l in lives} == {(1, 600), (2, END)}


def test_find_lives_disconnect_without_open_life_is_ignored() -> None:
    lives = find_lives(events((100, 1, "A")), events((200, 1, "A")), END, events((300, 1, "A")))
    assert [(l.spawn_tick, l.death_tick) for l in lives] == [(100, 200)]


def test_find_lives_closes_open_life_at_demo_end() -> None:
    lives = find_lives(events((100, 1, "A")), events(), 5000)
    assert lives == [Life("A", 1, 0, 100, 5000)]


def test_find_lives_ignores_events_after_demo_end_and_zero_length_lives() -> None:
    lives = find_lives(events((100, 1, "A"), (6000, 2, "B"), (5000, 3, "C")), events((7000, 1, "A")), 5000)
    assert lives == [Life("A", 1, 0, 100, 5000)]  # B spawned after the end; C spawned on the last tick


def test_find_lives_death_and_respawn_on_the_same_tick() -> None:
    lives = find_lives(events((100, 1, "A"), (400, 1, "A")), events((400, 1, "A")), 1000)
    assert [(l.spawn_tick, l.death_tick) for l in lives] == [(100, 400), (400, 1000)]


def test_find_lives_separates_bots_by_user_id_not_steamid() -> None:
    lives = find_lives(events((100, 2, "Bot"), (100, 3, "Bot")), events((200, 3, "Bot")), 1000)
    assert [(l.user_id, l.death_tick) for l in lives] == [(2, 1000), (3, 200)]


def test_find_lives_keeps_entity_id_and_steamid() -> None:
    spawns = events((100, 1, "A"))
    spawns["steamid"] = [76561198000000001]
    spawns["entity_id"] = pd.array([7], dtype="Int64")
    (l,) = find_lives(spawns, events(), 500)
    assert (l.steamid, l.entity_id) == (76561198000000001, 7)


# ---- plan_spectate --------------------------------------------------------------------------------------------


def test_settle_ticks_trimmed_after_spawn_and_before_death() -> None:
    (seg,) = plan_spectate([life(1, 100, 1000)], "longest_remaining", CFG, rng())
    assert (seg.start_tick, seg.end_tick) == (100 + SETTLE, 1000 - SETTLE)
    assert plan_spectate([life(1, 100, 100 + 2 * SETTLE)], "longest_remaining", CFG, rng()) == []


def test_longest_remaining_picks_longest_life_left_and_switches_only_at_death() -> None:
    lives = [life(1, 0, 500), life(2, 100, 900), life(3, 200, 1200)]
    segs = plan_spectate(lives, "longest_remaining", CFG, rng())
    # At t=10 only bot1 is alive; at its end (490) + settle, the longest-living alive bot is bot3 (ends 1190).
    assert [(s.target.user_id, s.start_tick, s.end_tick) for s in segs] == [(1, 10, 490), (3, 500, 1190)]
    for prev, nxt in itertools.pairwise(segs):
        assert prev.end_tick == prev.target.death_tick - SETTLE  # the target dies; no earlier switch
        assert nxt.start_tick == prev.end_tick + SETTLE


def test_longest_remaining_jumps_over_gaps_with_nobody_alive() -> None:
    segs = plan_spectate([life(1, 0, 300), life(2, 1000, 1400)], "longest_remaining", CFG, rng())
    assert [(s.target.user_id, s.start_tick, s.end_tick) for s in segs] == [(1, 10, 290), (2, 1010, 1390)]


def test_round_robin_prefers_the_less_viewed_player_on_ties() -> None:
    lives = [life(1, 0, 500), life(2, 0, 500), life(1, 600, 1500), life(2, 600, 1500)]
    for seed in range(8):
        segs = plan_spectate(lives, "longest_remaining", CFG, rng(seed))
        assert len(segs) == 2
        assert segs[0].target.user_id != segs[1].target.user_id


def test_plan_is_deterministic_for_a_seed_and_never_overlaps() -> None:
    gen = np.random.default_rng(123)
    lives = []
    for uid in range(1, 9):
        t = int(gen.integers(0, 200))
        while t < 20_000:
            death = t + int(gen.integers(40, 2000))
            lives.append(life(uid, t, death))
            t = death + int(gen.integers(0, 300))
    first = plan_spectate(lives, "longest_remaining", CFG, rng(7))
    assert first == plan_spectate(lives, "longest_remaining", CFG, rng(7))
    assert len(first) > 3
    for seg in first:
        assert seg.target.spawn_tick + SETTLE <= seg.start_tick < seg.end_tick <= seg.target.death_tick - SETTLE
    for prev, nxt in itertools.pairwise(first):
        assert nxt.start_tick >= prev.end_tick + SETTLE
    assert [s.segment_idx for s in first] == list(range(len(first)))


def test_tied_lives_are_drawn_from_the_seeded_rng() -> None:
    lives = [life(uid, 0, 500) for uid in range(1, 6)]
    picks = {plan_spectate(lives, "longest_remaining", CFG, rng(s))[0].target.user_id for s in range(30)}
    assert len(picks) > 1  # the draw depends on the seed
    assert plan_spectate(lives, "longest_remaining", CFG, rng(4)) == plan_spectate(
        lives, "longest_remaining", CFG, rng(4)
    )


def test_fixed_player_follows_only_that_player() -> None:
    lives = [
        life(1, 0, 500, "Me"),
        life(2, 0, 2000, "Bot"),
        life(3, 700, 1500, "Me"),
        life(4, 800, 800 + 2 * SETTLE, "Me"),
    ]
    segs = plan_spectate(lives, "fixed_player", CFG, rng(), player="Me")
    # The longer-living bot is ignored, and the too-short life of "Me" is skipped.
    assert [(s.target.player_name, s.start_tick, s.end_tick) for s in segs] == [
        ("Me", SETTLE, 500 - SETTLE),
        ("Me", 700 + SETTLE, 1500 - SETTLE),
    ]
    assert [s.segment_idx for s in segs] == [0, 1]


def test_policy_errors() -> None:
    with pytest.raises(ValueError, match="unknown"):
        plan_spectate([], "random", CFG, rng())
    with pytest.raises(ValueError, match="needs a player"):
        plan_spectate([life(1, 0, 500)], "fixed_player", CFG, rng())
    with pytest.raises(ValueError, match="no lives"):
        plan_spectate([life(1, 0, 500)], "fixed_player", CFG, rng(), player="Nobody")
    assert plan_spectate([], "longest_remaining", CFG, rng()) == []


# ---- split_episodes -------------------------------------------------------------------------------------------


def seg_of_frames(n_frames: int, start: int = 1000, idx: int = 0) -> ViewSegment:
    return ViewSegment(life(1, 0, 10**7), start, start + n_frames * TPF, idx)


def test_split_into_chunks_of_max_frames_and_extra_frame_fits() -> None:
    max_frames, min_frames = CFG.episodes.max_frames, CFG.episodes.min_frames
    seg = seg_of_frames(2 * max_frames + min_frames + 5)
    eps = split_episodes([seg], "demoX", "dm", CFG)
    assert [e.n_frames for e in eps] == [max_frames, max_frames, min_frames + 5]
    assert [e.start_tick for e in eps] == [1000, 1000 + max_frames * TPF, 1000 + 2 * max_frames * TPF]
    assert [e.episode_id for e in eps] == [f"dm-demoX-s000-c0{c}" for c in range(3)]
    assert eps[-1].end_tick == seg.end_tick  # the extra rendered frame T_n is inside the segment
    assert all(e.ticks_per_frame == TPF and e.kind == "dm" and e.demo_id == "demoX" for e in eps)


def test_split_drops_remainder_below_min_frames() -> None:
    max_frames, min_frames = CFG.episodes.max_frames, CFG.episodes.min_frames
    eps = split_episodes([seg_of_frames(max_frames + min_frames - 1)], "d", "dm", CFG)
    assert [e.n_frames for e in eps] == [max_frames]
    assert split_episodes([seg_of_frames(min_frames - 1)], "d", "dm", CFG) == []
    assert [e.n_frames for e in split_episodes([seg_of_frames(min_frames)], "d", "dm", CFG)] == [min_frames]


def test_split_truncates_partial_frames_with_integer_ticks() -> None:
    seg = ViewSegment(life(1, 0, 10**7), 1000, 1000 + 100 * TPF + 3, 2)
    (ep,) = split_episodes([seg], "d", "human", CFG)
    assert ep.n_frames == 100 and ep.end_tick <= seg.end_tick
    assert ep.episode_id == "human-d-s002-c00" and isinstance(ep.start_tick, int)


def test_split_carries_target_identifiers() -> None:
    target = Life("Me", 5, 76561198000000001, 0, 10**7, 3)
    (ep,) = split_episodes([ViewSegment(target, 0, 100 * TPF, 0)], "d", "dm", CFG)
    assert (ep.player_name, ep.user_id, ep.steamid) == ("Me", 5, 76561198000000001)


# ---- render plan ----------------------------------------------------------------------------------------------


def make_plan_inputs(cfg=CFG):
    lives = [life(1, 0, 200), life(2, 100, 20_000, "Two"), life(3, 30_000, 30_040)]  # first and last are too short
    segs = plan_spectate(lives, "longest_remaining", cfg, rng())
    return segs, split_episodes(segs, "demoX", "dm", cfg)


def test_plan_has_everything_render_needs(tmp_path: Path) -> None:
    cfg = load_config(overrides=[f"segment.settle_ticks={SETTLE}", "render.tick_offset=2"])
    segs, eps = make_plan_inputs(cfg)
    assert len(segs) == 3 and {e.segment_idx for e in eps} == {1}  # the first and last segments are below min_frames
    path = tmp_path / "plans" / "demoX" / "render_plan.json"
    segment.write_render_plan(path, "demoX", segs, eps, cfg, {"map_name": "de_dust2"}, 123, {"policy": "p"})
    plan = segment.read_render_plan(path)

    assert plan["demo_id"] == "demoX" and plan["cs2_build"] == 123 and plan["tick_offset"] == 2
    assert plan["header"] == {"map_name": "de_dust2"} and plan["meta"] == {"policy": "p"}
    assert plan["timing"] == {"tickrate": 64, "fps": 16, "ticks_per_frame": 4}
    (s,) = plan["segments"]  # the segment without an episode is not rendered
    assert s["target"] == {"name": "Two", "user_id": 2, "steamid": 0, "entity_id": None}
    n = sum(e.n_frames for e in eps)
    assert (s["labeled_frames"], s["rendered_frames"]) == (n, n + 1)  # lag-free: one extra rendered frame
    assert [e["rendered_frames"] for e in s["episodes"]] == [e.n_frames + 1 for e in eps]
    assert [e["first_frame"] for e in s["episodes"]] == [i * CFG.episodes.max_frames for i in range(len(eps))]
    assert plan["totals"] == {"segments": 1, "episodes": len(eps), "labeled_frames": n, "rendered_frames": n + 1}

    start, end = eps[0].start_tick, eps[-1].end_tick
    assert plan["mirv_cmd_schedule"] == [
        {"mirv_tick": max(start - SETTLE, 0) + 2, "demo_tick": max(start - SETTLE, 0), "action": "spec",
         "segment_idx": s["segment_idx"], "target": s["target"]},
        {"mirv_tick": start + 2, "demo_tick": start, "action": "record_start", "segment_idx": s["segment_idx"]},
        {"mirv_tick": end + 2, "demo_tick": end, "action": "record_end", "segment_idx": s["segment_idx"]},
    ]  # fmt: skip
    assert (s["mirv_record_start_tick"], s["mirv_record_end_tick"]) == (start + 2, end + 2)


def test_plan_round_trips_episodes(tmp_path: Path) -> None:
    segs, eps = make_plan_inputs()
    path = tmp_path / "render_plan.json"
    segment.write_render_plan(path, "demoX", segs, eps, CFG, {})
    plan = segment.read_render_plan(path)
    assert segment.episodes_from_plan(plan) == eps
    assert json.loads(path.read_text()) == plan
    segment.write_render_plan(tmp_path / "again.json", "demoX", segs, eps, CFG, {})
    assert (tmp_path / "again.json").read_text() == path.read_text()  # byte-identical: deterministic


def test_schedule_orders_end_before_switch_before_start_on_one_tick() -> None:
    lives = [life(1, 0, 4000), life(2, 3000, 9000)]
    segs = plan_spectate(lives, "longest_remaining", CFG, rng())
    eps = split_episodes(segs, "d", "dm", CFG)
    sched = segment.build_render_plan("d", segs, eps, CFG, {})["mirv_cmd_schedule"]
    ticks = [e["mirv_tick"] for e in sched]
    assert ticks == sorted(ticks)
    first_end = next(e for e in sched if e["action"] == "record_end")
    second_spec = [e for e in sched if e["action"] == "spec"][1]
    assert second_spec["mirv_tick"] >= first_end["mirv_tick"]  # never switches while still recording
    assert sched.index(first_end) < sched.index(second_spec)


def test_negative_mirv_tick_is_rejected() -> None:
    cfg = load_config(overrides=[f"segment.settle_ticks={SETTLE}", "render.tick_offset=-100000"])
    segs, eps = make_plan_inputs(cfg)
    with pytest.raises(ValueError, match="tick_offset"):
        segment.build_render_plan("d", segs, eps, cfg, {})


def test_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    segs, eps = make_plan_inputs()
    path = tmp_path / "p" / "render_plan.json"
    segment.write_render_plan(path, "d", segs, eps, CFG, {})
    good = path.read_text()
    assert [p.name for p in path.parent.iterdir()] == ["render_plan.json"]  # no temp file left behind

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(segment.os, "replace", boom)
    with pytest.raises(OSError):
        segment.write_render_plan(path, "d", segs, [], CFG, {})
    assert path.read_text() == good and [p.name for p in path.parent.iterdir()] == ["render_plan.json"]


def test_read_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text('{"schema_version": 99}')
    with pytest.raises(ValueError, match="schema"):
        segment.read_render_plan(path)


# ---- run (fake parser) ----------------------------------------------------------------------------------------


def fake_demo_parser() -> FakeParser:
    def raw(rows):
        return pd.DataFrame(
            {
                "tick": [r[0] for r in rows],
                "user_user_id": [r[1] for r in rows],
                "user_name": [r[2] for r in rows],
                "user_steamid": [0] * len(rows),
            }
        )

    ticks = pd.DataFrame({"tick": [0, 6000], "user_id": [1, 1], "name": ["A", "A"], "steamid": [0, 0]})
    return FakeParser(
        ticks=ticks,
        events={
            "player_spawn": raw([(0, 1, "A"), (0, 2, "B")]),
            "player_death": raw([(3000, 1, "A")]),
            "player_disconnect": raw([(5000, 2, "B")]),
        },
    )


def test_run_writes_plan_for_a_demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    game = tmp_path / "steamapps" / "common" / "CS"
    game.mkdir(parents=True)
    (tmp_path / "steamapps" / "appmanifest_730.acf").write_text('"AppState"\n{\n"buildid" "4242"\n}\n')
    cfg = load_config(
        overrides=[f"paths.plans={tmp_path / 'plans'}", f"paths.cs2_dir={game}", f"segment.settle_ticks={SETTLE}"]
    )
    monkeypatch.setattr(demo, "_make_parser", lambda path: fake_demo_parser())

    out = segment.run(tmp_path / "demos" / "dm" / "demo1.dem", cfg)
    assert out == tmp_path / "plans" / "demo1" / "render_plan.json"
    plan = segment.read_render_plan(out)
    assert plan["cs2_build"] == 4242 and plan["meta"]["kind"] == "dm" and plan["meta"]["policy"] == "longest_remaining"
    assert [(s["target"]["user_id"], s["record_start_tick"]) for s in plan["segments"]] == [(2, SETTLE)]
    # bot B was disconnected at 5000, so the segment ends at 5000 - settle, not at the demo end (6000)
    assert plan["segments"][0]["life"]["death_tick"] == 5000
    assert plan["segments"][0]["record_end_tick"] <= 5000 - SETTLE
    assert out == segment.run(tmp_path / "demos" / "dm" / "demo1.dem", cfg)  # idempotent


def test_run_fixed_player_marks_episodes_human(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(overrides=[f"paths.plans={tmp_path}", f"paths.cs2_dir={tmp_path / 'none'}"])
    monkeypatch.setattr(demo, "_make_parser", lambda path: fake_demo_parser())
    plan = segment.read_render_plan(segment.run(tmp_path / "d1.dem", cfg, "fixed_player", "A"))
    assert plan["meta"]["kind"] == "human" and plan["meta"]["player"] == "A"
    assert plan["cs2_build"] is None  # no manifest and generate.expected_build is null
    assert {s["target"]["name"] for s in plan["segments"]} == {"A"}


def test_cli_segment_single_demo_and_all_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from cs2wm import cli

    monkeypatch.setenv("DCS2_ROOT", str(tmp_path))
    monkeypatch.setattr(demo, "_make_parser", lambda path: fake_demo_parser())
    dem = tmp_path / "demos" / "dm" / "demo9.dem"
    dem.parent.mkdir(parents=True)
    dem.touch()
    assert cli.main(["segment", "demo9"]) == 0  # resolved by demo id under paths.demos
    out = Path(capsys.readouterr().out.strip())
    assert out == tmp_path / "plans" / "demo9" / "render_plan.json" and out.is_file()
    assert cli.main(["segment", "--all-pending"]) == 2  # needs the supervisor state (#20)
    with pytest.raises(SystemExit):
        cli.main(["segment", "missing"])
