"""Tests for the demodump shim (#72) against a small checked-in dump; no .dem file and no Go toolchain needed."""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import stat
from pathlib import Path

import pandas as pd
import pytest

from cs2wm import demo, demodump
from cs2wm.actions import slot_for_weapon
from cs2wm.config import ConfigError, load_config
from cs2wm.pipeline import segment

# Sliced from a real dump of a deathmatch GOTV demo (3 bots, 80 ticks); the smoke rows are written by hand.
FIXTURE = Path(__file__).parent / "fixtures" / "demodump_fixture"
REPO = Path(__file__).resolve().parent.parent
USERS = {3: "Wesley", 4: "Dashen", 5: "Waldo"}


@pytest.fixture
def parser() -> demodump.DemodumpParser:
    return demodump.DemodumpParser("fixture.dem", FIXTURE, dump=FIXTURE)


@pytest.fixture
def use_fixture(monkeypatch):
    monkeypatch.setattr(demo, "_make_parser", lambda path: demodump.DemodumpParser(path, FIXTURE, dump=FIXTURE))


# --- the Python constants follow the Go sources ---


def test_format_version_matches_the_go_exporter() -> None:
    main = (REPO / "tools" / "demodump" / "main.go").read_text(encoding="utf-8")
    assert re.search(rf'const version = "{demodump.DEMODUMP_FORMAT}"', main)


def test_behavior_version_matches_the_go_exporter() -> None:
    main = (REPO / "tools" / "demodump" / "main.go").read_text(encoding="utf-8")
    assert re.search(rf'const behaviorVersion = "{demodump.DEMODUMP_BEHAVIOR}"', main)


def test_demoinfocs_version_matches_go_mod() -> None:
    mod = (REPO / "tools" / "demodump" / "go.mod").read_text(encoding="utf-8")
    assert f"demoinfocs-golang/v6 {demodump.DEMOINFOCS_VERSION}\n" in mod


# --- binary lookup ---


@pytest.fixture
def lookup(monkeypatch, tmp_path):
    """Isolated lookup: empty repo bin/, empty PATH, no env override."""
    monkeypatch.setattr(demodump, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.delenv(demodump.ENV_BINARY, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    return tmp_path


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


def test_find_binary_prefers_env_then_config_then_repo_then_path(lookup, monkeypatch) -> None:
    env, cfg = _touch(lookup / "env" / "d"), _touch(lookup / "cfg" / "d")
    repo = _touch(lookup / "repo" / "bin" / demodump.binary_name())
    on_path = _touch(lookup / "path" / "demodump")
    on_path.chmod(on_path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(on_path.parent))
    monkeypatch.setenv(demodump.ENV_BINARY, str(env))
    assert demodump.find_binary(cfg) == env
    monkeypatch.delenv(demodump.ENV_BINARY)
    assert demodump.find_binary(cfg) == cfg
    assert demodump.find_binary(None) == repo
    repo.unlink()
    assert demodump.find_binary(None).name == "demodump"


def test_find_binary_skips_a_configured_path_that_does_not_exist(lookup) -> None:
    repo = _touch(lookup / "repo" / "bin" / demodump.binary_name())
    assert demodump.find_binary(lookup / "nope") == repo


def test_find_binary_error_names_the_build_script(lookup) -> None:
    with pytest.raises(demodump.DemodumpNotFound, match=r"scripts/build_demodump\.sh") as err:
        demodump.find_binary(None)
    assert demodump.ENV_BINARY in str(err.value)


# --- running and caching ---

pytestmark_posix = pytest.mark.skipif(os.name == "nt", reason="the fake exporter is a shell script")

FAKE_EXPORTER = """#!/bin/sh
if [ "$1" = "--version" ]; then echo "demodump {fmt} demoinfocs {infocs} behavior {behavior}"; exit 0; fi
echo run >> "{counter}"
if [ -n "{fail}" ]; then echo "boom: bad demo" >&2; exit 1; fi
cp -r "{fixture}/." "$2"
"""


def _fake_exporter(
    tmp_path: Path, fmt: str = demodump.DEMODUMP_FORMAT, fail: bool = False, behavior: str = demodump.DEMODUMP_BEHAVIOR
) -> tuple[Path, Path]:
    counter = tmp_path / "runs.txt"
    exe = tmp_path / "fake_demodump"
    exe.write_text(
        FAKE_EXPORTER.format(
            fmt=fmt,
            infocs=demodump.DEMOINFOCS_VERSION,
            behavior=behavior,
            counter=counter,
            fail="1" if fail else "",
            fixture=FIXTURE,
        ),
        encoding="utf-8",
    )
    exe.chmod(0o755)
    return exe, counter


def _runs(counter: Path) -> int:
    return len(counter.read_text().split()) if counter.exists() else 0


@pytestmark_posix
def test_dump_is_cached_per_path_size_and_mtime(tmp_path) -> None:
    exe, counter = _fake_exporter(tmp_path)
    demo_file = tmp_path / "a.dem"
    demo_file.write_bytes(b"12345")
    cache = tmp_path / "cache"

    first = demodump.dump_demo(demo_file, cache, exe)
    assert demodump.dump_demo(demo_file, cache, exe) == first
    assert _runs(counter) == 1
    assert (first / "header.json").is_file()
    assert [p.name for p in cache.iterdir()] == [first.name]  # no temp dirs left behind

    os.utime(demo_file, ns=(1, 1))  # mtime changed -> new entry
    second = demodump.dump_demo(demo_file, cache, exe)
    assert second != first and _runs(counter) == 2
    demo_file.write_bytes(b"123456")  # size changed
    os.utime(demo_file, ns=(1, 1))
    assert demodump.dump_demo(demo_file, cache, exe) not in (first, second)
    assert _runs(counter) == 3


@pytestmark_posix
def test_failed_dump_raises_with_stderr_and_caches_nothing(tmp_path) -> None:
    exe, _ = _fake_exporter(tmp_path, fail=True)
    demo_file = tmp_path / "bad.dem"
    demo_file.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="boom: bad demo"):
        demodump.dump_demo(demo_file, tmp_path / "cache", exe)
    assert list((tmp_path / "cache").iterdir()) == []


@pytestmark_posix
def test_exporter_with_another_format_version_is_rejected(tmp_path) -> None:
    exe, counter = _fake_exporter(tmp_path, fmt="99")
    demo_file = tmp_path / "a.dem"
    demo_file.write_bytes(b"x")
    with pytest.raises(RuntimeError, match=r"is format 99.*build_demodump\.sh"):
        demodump.dump_demo(demo_file, tmp_path / "cache", exe)
    assert _runs(counter) == 0


@pytestmark_posix
def test_exporter_with_another_behavior_version_is_rejected(tmp_path) -> None:
    exe, counter = _fake_exporter(tmp_path, behavior="99")
    demo_file = tmp_path / "a.dem"
    demo_file.write_bytes(b"x")
    with pytest.raises(RuntimeError, match=r"behavior 99.*build_demodump\.sh"):
        demodump.dump_demo(demo_file, tmp_path / "cache", exe)
    assert _runs(counter) == 0


def test_cache_key_changes_with_the_exporter_versions(tmp_path, monkeypatch) -> None:
    demo_file = tmp_path / "a.dem"
    demo_file.write_bytes(b"x")
    keys = {demodump.cache_key(demo_file)}
    for name in ("DEMODUMP_BEHAVIOR", "DEMODUMP_FORMAT", "DEMOINFOCS_VERSION"):
        with monkeypatch.context() as m:
            m.setattr(demodump, name, "changed")
            keys.add(demodump.cache_key(demo_file))
    assert len(keys) == 4


def test_cached_dump_needs_no_binary(tmp_path, lookup) -> None:
    demo_file = tmp_path / "a.dem"
    demo_file.write_bytes(b"x")
    cache = tmp_path / "cache"
    shutil.copytree(FIXTURE, cache / demodump.cache_key(demo_file))
    parser = demodump.DemodumpParser(demo_file, cache)  # lookup fixture: no binary anywhere
    assert parser.parse_header()["map_name"] == "de_dust2"


# --- the DemoParser-compatible surface ---


def test_parse_header(parser) -> None:
    hdr = parser.parse_header()
    assert hdr["map_name"] == "de_dust2" and hdr["patch_version"] == 14196 and hdr["tick_rate"] == 64


def test_parse_ticks_returns_always_columns_plus_wanted_known_props(parser) -> None:
    df = parser.parse_ticks(["X", "Y", "Z", "yaw", "pitch", "FORWARD", "buttons"])
    assert list(df.columns) == ["tick", "name", "steamid", "user_id", "X", "Y", "Z", "yaw", "pitch"]
    assert len(df) == 3 * 80
    assert df.groupby("user_id")["tick"].nunique().to_dict() == {3: 80, 4: 80, 5: 80}  # one row per tick and player
    assert not df.duplicated(["tick", "user_id"]).any()
    assert dict(df.groupby("user_id")["name"].first()) == USERS


def test_parse_ticks_has_every_column_the_callers_require(parser) -> None:
    from cs2wm.pipeline.extract_actions import REQUIRED_TICK_COLUMNS

    wanted = [*REQUIRED_TICK_COLUMNS, "is_walking", "is_alive", "entity_id", "health"]
    df = parser.parse_ticks(wanted)
    assert set(wanted) <= set(df.columns)
    assert df["is_airborne"].dtype == bool and df["is_alive"].dtype == bool
    assert set(df["active_weapon_name"]) == {"AK-47", "Glock-18", "Knife"}


def test_parse_ticks_filters(parser) -> None:
    df = parser.parse_ticks(["X"], players=[4], ticks=[2000, 2001, 5])
    assert df[["tick", "user_id"]].values.tolist() == [[2000, 4], [2001, 4]]


def test_list_game_events_includes_derived_and_raw(parser) -> None:
    assert {"player_spawn", "player_death", "weapon_fire", "weapon_reload", "smokegrenade_detonate"} <= set(
        parser.list_game_events()
    )


def test_parse_events_attaches_the_user_role(parser) -> None:
    [(name, fire)] = parser.parse_events(["weapon_fire"], player=["user_id"])
    assert name == "weapon_fire"
    assert fire["tick"].is_monotonic_increasing
    assert (fire["user_user_id"] == fire["user_id"]).all()
    assert dict(zip(fire["user_user_id"], fire["user_name"])) == {
        k: v for k, v in USERS.items() if k in set(fire["user_id"])
    }
    assert (fire["user_steamid"] == 0).all()
    assert "user_entity_id" not in fire.columns


def test_parse_events_entity_id_on_request_and_skips_absent_events(parser) -> None:
    out = parser.parse_events(["player_spawn", "no_such_event"], player=["user_id", "entity_id"])
    assert [n for n, _ in out] == ["player_spawn"]
    spawn = out[0][1]
    assert dict(zip(spawn["user_user_id"], spawn["user_entity_id"])) == {3: 4, 4: 5, 5: 6}  # controller entities


def test_derived_events_are_flagged(parser) -> None:
    [(_, reload_)] = parser.parse_events(["weapon_reload"])
    assert reload_["derived"].all()
    [(_, death)] = parser.parse_events(["player_death"])
    assert not death["derived"].any()


def test_parse_grenades(parser) -> None:
    g = parser.parse_grenades()
    assert g["grenade_entity_id"].nunique() == 1 and len(g) == 3
    assert {"tick", "grenade_type", "thrower_steamid", "X", "Y", "Z"} <= set(g.columns)


# --- through cs2wm.demo, as the pipeline stages call it ---


def test_demo_inspect_reports_players_and_event_counts(use_fixture) -> None:
    info = demo.inspect("fixture.dem")
    assert info["ok"] and info["error"] is None
    assert (info["first_tick"], info["last_tick"]) == (2000, 2079)
    assert {p["user_id"]: p["name"] for p in info["players"]} == USERS
    assert set(info["events"]) == {"player_spawn", "player_death", "weapon_fire"}
    assert info["events"]["weapon_fire"] > 0


def test_demo_load_events_feeds_player_event_table(use_fixture) -> None:
    events = demo.load_events("fixture.dem", ["player_spawn", "player_death", "player_disconnect"], ["user_id"])
    spawn = demo.player_event_table(events["player_spawn"])
    assert set(spawn["user_id"]) == set(USERS)
    assert dict(zip(spawn["user_id"], spawn["name"])) == USERS
    assert events["player_disconnect"].empty  # the dump lacks it: filled with an empty frame


def test_demo_load_ticks_and_last_tick(use_fixture) -> None:
    assert demo.last_tick("fixture.dem") == 2079
    assert demo.load_ticks("fixture.dem", ["entity_id"], players=[5])["entity_id"].unique().tolist() == [6]


def test_segment_attaches_entity_ids_from_the_dump(use_fixture) -> None:
    ticks = demo.load_ticks("fixture.dem", [demo.TICK_ENTITY_ID])
    life = segment.Life("Dashen", 4, 0, 2000, 2079, None)
    assert segment.attach_entity_ids([life], ticks)[0].entity_id == 5


def test_smoke_events_and_grenade_throws(use_fixture) -> None:
    smokes = demo.smoke_events("fixture.dem")
    row = smokes.iloc[0]
    assert (row["entity_id"], row["detonate_tick"], row["thrower_user_id"]) == (901, 2040, 4)
    assert (row["x"], row["y"], row["z"]) == (100.5, -200.25, 8.0)
    assert row["expire_tick"] == 2040 + demo.SMOKE_DURATION_TICKS
    throws = demo.grenade_throws("fixture.dem")
    assert throws["tick"].tolist() == [2010]


# --- weapon strings ---

# Every display name demoinfocs-golang v6 can return for an equipment (common/equipment.go).
RIFLES_AND_SMGS = [
    "AK-47", "AUG", "AWP", "PP-Bizon", "FAMAS", "G3SG1", "Galil AR", "M249", "M4A4", "M4A1", "MAC-10", "MAG-7", "MP7",
    "MP5-SD", "MP9", "Negev", "Nova", "P90", "Sawed-Off", "SCAR-20", "SG 553", "SSG 08", "UMP-45", "XM1014",
]  # fmt: skip
PISTOLS = ["Desert Eagle", "Dual Berettas", "Five-SeveN", "Glock-18", "P2000", "P250", "Tec-9", "CZ75 Auto", "USP-S", "R8 Revolver"]  # fmt: skip
OTHER = ["Knife", "C4", "Zeus x27", "Smoke Grenade", "Flashbang", "HE Grenade", "Molotov", "Incendiary Grenade", "Decoy Grenade", "UNKNOWN"]  # fmt: skip


@pytest.mark.parametrize(("names", "slot"), [(RIFLES_AND_SMGS, "1"), (PISTOLS, "2"), (OTHER, "3")])
def test_demoinfocs_weapon_names_map_to_the_expected_slot(names, slot) -> None:
    cfg = load_config()
    assert {n: slot_for_weapon(n, cfg) for n in names} == {n: slot for n in names}


# --- backend selection ---


@pytest.fixture
def restore_settings(monkeypatch):
    monkeypatch.setattr(demo, "_settings", demo._Settings())


def test_default_backend_is_demoinfocs_and_configure_applies_the_demo_section(restore_settings, tmp_path) -> None:
    cfg = load_config(overrides=[f"paths.root={tmp_path}", "demo.demodump=/x/demodump"])
    assert cfg.demo.backend == "demoinfocs"
    assert demo._settings.backend == "demoinfocs"
    assert demo._settings.demodump == "/x/demodump"
    assert demo._cache_dir() == tmp_path / "demo_cache"


def test_demoparser2_backend_returns_the_demoparser2_parser(restore_settings) -> None:
    pytest.importorskip("demoparser2")
    load_config(overrides=["demo.backend=demoparser2"])
    assert type(demo._make_parser(FIXTURE / "header.json")).__name__ == "DemoParser"


def test_cache_dir_must_be_configured(restore_settings) -> None:
    with pytest.raises(RuntimeError, match="load_config"):
        demo._cache_dir()


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ConfigError, match="demo.backend"):
        load_config(overrides=["demo.backend=nope"])


def test_make_parser_uses_the_demodump_shim_by_default(restore_settings, tmp_path, monkeypatch) -> None:
    demo_file = tmp_path / "a.dem"
    demo_file.write_bytes(b"x")
    monkeypatch.setattr(demo, "_settings", demo._Settings(cache_dir=tmp_path / "cache"))
    shutil.copytree(FIXTURE, tmp_path / "cache" / demodump.cache_key(demo_file))
    assert isinstance(demo._make_parser(demo_file), demodump.DemodumpParser)


# --- weapon columns ---


def test_weapon_columns_raw_and_derived(parser) -> None:
    [(_, fire)] = parser.parse_events(["weapon_fire"])
    assert fire["weapon"].str.startswith("weapon_").all()  # raw string, as the game sends it
    assert fire["weapon_name"].isin({"AK-47", "Glock-18", "Knife"}).all()  # display name, as active_weapon_name
    [(_, reload_)] = parser.parse_events(["weapon_reload"])
    assert "weapon" not in reload_.columns  # derived events have no raw string
    assert reload_["weapon_name"].isin({"AK-47", "Glock-18", "Knife"}).all()


# --- hand-made dumps: truncation, pawn-less rows, reused user ids ---


@pytest.fixture
def dump_copy(tmp_path):
    dest = tmp_path / "dump"
    shutil.copytree(FIXTURE, dest)
    return dest


def _use(monkeypatch, dump: Path) -> None:
    monkeypatch.setattr(demo, "_make_parser", lambda path: demodump.DemodumpParser(path, dump, dump=dump))


def _set_header(dump: Path, **fields) -> None:
    path = dump / "header.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **fields}), encoding="utf-8")


def test_complete_demo_is_not_truncated(use_fixture) -> None:
    assert demo.inspect("fixture.dem")["truncated"] is False


def test_truncated_dump_is_parseable_and_flagged(dump_copy, monkeypatch, caplog) -> None:
    _set_header(dump_copy, truncated=True, last_tick=2079)
    _use(monkeypatch, dump_copy)
    with caplog.at_level("WARNING", logger="cs2wm.demodump"):
        info = demo.inspect("cut.dem")
    assert info["ok"] and info["truncated"] is True and info["error"] is None
    assert info["last_tick"] == 2079 and len(info["players"]) == 3
    assert "truncated" in caplog.text
    assert demo.is_parseable("cut.dem")


def test_truncated_dump_without_ticks_is_not_parseable(dump_copy, monkeypatch) -> None:
    _set_header(dump_copy, truncated=True)
    (dump_copy / "ticks.csv.gz").write_bytes(gzip.compress(b"tick,user_id,name,steamid\n"))
    _use(monkeypatch, dump_copy)
    info = demo.inspect("cut.dem")
    assert not info["ok"] and "no ticks" in info["error"] and info["truncated"] is True
    assert not demo.is_parseable("cut.dem")


def test_rows_without_a_pose_load_as_nan_and_dead(dump_copy) -> None:
    path = dump_copy / "ticks.csv.gz"
    df = pd.read_csv(path, dtype=str)
    blank = ["X", "Y", "Z", "yaw", "pitch", "duck_amount", "active_weapon_name"]
    first = df["tick"] == df["tick"].iloc[0]
    df.loc[first, blank] = None
    df.loc[first, ["is_alive", "is_airborne", "is_walking", "is_scoped"]] = "false"
    df.to_csv(path, index=False)

    out = demodump.DemodumpParser("x.dem", dump_copy, dump=dump_copy).parse_ticks(
        ["X", "Z", "yaw", "pitch", "duck_amount", "active_weapon_name", "is_alive", "is_airborne"]
    )
    blank_rows = out[out["tick"] == out["tick"].min()]
    assert len(blank_rows) == 3
    assert blank_rows[["X", "Z", "yaw", "pitch", "duck_amount", "active_weapon_name"]].isna().all().all()
    assert not blank_rows["is_alive"].any() and not blank_rows["is_airborne"].any()
    assert out["is_alive"].dtype == bool and out["is_airborne"].dtype == bool  # still plain booleans
    assert out.loc[out["tick"] > out["tick"].min(), ["X", "yaw"]].notna().all().all()


def _roster_dump(dump: Path, rows: list[tuple]) -> demodump.DemodumpParser:
    roster = pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "entity_id",
            "pawn_entity_id",
            "name",
            "steamid",
            "is_bot",
            "team",
            "first_tick",
            "last_tick",
        ],
    )
    roster.to_csv(dump / "players.csv", index=False)
    events = pd.DataFrame({"tick": [5, 150, 400, 900, 120], "derived": False, "user_id": [3, 3, 3, 3, 9]})
    with gzip.open(dump / "events" / "player_jump.csv.gz", "wt") as f:
        events.to_csv(f, index=False)
    return demodump.DemodumpParser("x.dem", dump, dump=dump)


def test_events_join_the_occupant_of_the_user_id_at_their_tick(dump_copy) -> None:
    parser = _roster_dump(
        dump_copy,
        [
            (3, 4, 339, "Ann", 111, True, 2, 100, 300),
            (3, 20, 340, "Bob", 222, False, 3, 350, 800),  # user_id 3 reused by another player
        ],
    )
    [(_, ev)] = parser.parse_events(["player_jump"], player=["user_id", "entity_id"])
    by_tick = ev.set_index("tick")
    assert by_tick.loc[5, "user_name"] == "Ann"  # before the first occupancy: the first one
    assert by_tick.loc[150, "user_name"] == "Ann" and by_tick.loc[150, "user_entity_id"] == 4
    assert by_tick.loc[400, "user_name"] == "Bob" and by_tick.loc[400, "user_steamid"] == 222
    assert by_tick.loc[400, "user_entity_id"] == 20
    assert by_tick.loc[900, "user_name"] == "Bob"  # after the last occupancy: the last one
    assert pd.isna(by_tick.loc[120, "user_name"]) and by_tick.loc[120, "user_user_id"] == 9  # not in the roster


def test_events_without_a_user_id_get_nan(dump_copy) -> None:
    parser = _roster_dump(dump_copy, [(3, 4, 339, "Ann", 111, True, 2, 100, 300)])
    frame = pd.DataFrame({"tick": [10, 20], "derived": False, "user_id": [None, 3.0]})
    with gzip.open(dump_copy / "events" / "player_jump.csv.gz", "wt") as f:
        frame.to_csv(f, index=False)
    [(_, ev)] = parser.parse_events(["player_jump"])
    assert pd.isna(ev.loc[0, "user_name"]) and ev.loc[1, "user_name"] == "Ann"
