"""Tests for the demoparser2 wrapper (#11, #13, #14) with a fake parser; no .dem file needed."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cs2wm import demo


class FakeParser:
    """Mimics the parts of ``demoparser2.DemoParser`` 0.42 that ``cs2wm.demo`` uses."""

    def __init__(self, ticks=None, events=None, grenades=None, header=None):
        self._ticks = ticks if ticks is not None else pd.DataFrame()
        self._events = events or {}
        self._grenades = grenades if grenades is not None else pd.DataFrame()
        self._header = header or {"map_name": "de_dust2", "demo_version_name": "valve_demo_2"}
        self.calls: list[tuple] = []

    def parse_header(self):
        return self._header

    def parse_ticks(self, wanted_props, *, players=None, ticks=None, prop_states=None):
        self.calls.append(("parse_ticks", list(wanted_props), players, ticks))
        df = self._ticks
        if ticks is not None:
            df = df[df["tick"].isin(ticks)]
        return df[[c for c in df.columns if c in {"tick", "name", "steamid", *wanted_props}]].copy()

    def list_game_events(self):
        return list(self._events)

    def parse_events(self, names, *, player=None, other=None):
        self.calls.append(("parse_events", list(names), player))
        return [(n, self._events[n]) for n in names]

    def parse_grenades(self, *, extra=None, grenades=True):
        return self._grenades


@pytest.fixture
def use_parser(monkeypatch):
    def install(parser: FakeParser) -> FakeParser:
        monkeypatch.setattr(demo, "_make_parser", lambda path: parser)
        return parser

    return install


def tick_table() -> pd.DataFrame:
    # Two bots with steamid 0 (demoparser2 #358) and one with a small steamid: told apart by user_id only.
    rows = []
    for tick in (100, 104, 108):
        rows += [
            {"tick": tick, "user_id": 2, "name": "Bot A", "steamid": 0, "X": 1.0},
            {"tick": tick, "user_id": 3, "name": "Bot B", "steamid": 0, "X": 2.0},
            {"tick": tick, "user_id": 4, "name": "Bot C", "steamid": 7, "X": 3.0},
        ]
    return pd.DataFrame(rows)


def test_parser_version_is_installed_version() -> None:
    assert demo.parser_version().split(".")[0].isdigit()


def test_header_values_are_strings(use_parser) -> None:
    use_parser(FakeParser(header={"map_name": "de_dust2", "network_protocol": 14000}))
    assert demo.header("x.dem") == {"map_name": "de_dust2", "network_protocol": "14000"}


def test_load_ticks_adds_user_id_and_filters_by_it_not_steamid(use_parser) -> None:
    parser = use_parser(FakeParser(ticks=tick_table()))
    df = demo.load_ticks("x.dem", ["X"], players=[2, 3])
    assert set(df["user_id"]) == {2, 3}  # both have steamid 0 and are still distinct
    assert parser.calls[0][1] == ["X", "user_id"]
    assert parser.calls[0][2] is None  # players is never forwarded to demoparser2


def test_load_ticks_filters_ticks_and_resets_index(use_parser) -> None:
    use_parser(FakeParser(ticks=tick_table()))
    df = demo.load_ticks("x.dem", ["X"], players=[4], ticks=[104])
    assert df["tick"].tolist() == [104]
    assert df.index.tolist() == [0]


def test_load_events_fills_missing_events_with_empty_frames(use_parser) -> None:
    fire = pd.DataFrame({"tick": [5], "user_user_id": [2]})
    parser = use_parser(FakeParser(events={"weapon_fire": fire}))
    out = demo.load_events("x.dem", ["weapon_fire", "player_jump"], ["user_id"])
    assert out["weapon_fire"].equals(fire)
    assert out["player_jump"].empty
    assert parser.calls[0] == ("parse_events", ["weapon_fire"], ["user_id"])  # absent events are not requested


def test_load_events_with_no_present_event_does_not_parse(use_parser) -> None:
    parser = use_parser(FakeParser(events={}))
    assert demo.load_events("x.dem", ["weapon_fire"])["weapon_fire"].empty
    assert parser.calls == []


def test_player_event_table_normalises_columns() -> None:
    raw = pd.DataFrame(
        {
            "tick": [10, 20],
            "user_user_id": [2, 3],
            "user_name": ["Bot A", None],
            "user_steamid": [0, 76561198000000000],
            "user_entity_id": [5, None],
        }
    )
    out = demo.player_event_table(raw)
    assert out[["tick", "user_id", "name", "steamid"]].values.tolist() == [
        [10, 2, "Bot A", 0],
        [20, 3, "", 76561198000000000],
    ]
    assert out["entity_id"].tolist()[0] == 5
    assert pd.isna(out["entity_id"].tolist()[1])


def test_player_event_table_without_optional_columns() -> None:
    out = demo.player_event_table(pd.DataFrame({"tick": [1], "userid": [9]}))
    assert out.to_dict("records") == [{"tick": 1, "user_id": 9, "name": "", "steamid": 0}]


def test_player_event_table_empty_and_missing_user_id() -> None:
    assert list(demo.player_event_table(pd.DataFrame()).columns) == ["tick", "user_id", "name", "steamid"]
    with pytest.raises(KeyError):
        demo.player_event_table(pd.DataFrame({"tick": [1], "foo": [2]}))


def test_grenade_throws_uses_first_row_of_each_trajectory(use_parser) -> None:
    g = pd.DataFrame(
        {
            "tick": [10, 11, 12, 30, 31],
            "grenade_entity_id": [100, 100, 100, 101, 101],
            "thrower_steamid": [0, 0, 0, 0, 0],
            "grenade_type": ["Smoke", "Smoke", "Smoke", "Smoke", "Smoke"],
        }
    )
    use_parser(FakeParser(grenades=g))
    assert demo.grenade_throws("x.dem")["tick"].tolist() == [10, 30]


def test_grenade_throws_without_trajectory_id_splits_on_gaps(use_parser) -> None:
    g = pd.DataFrame(
        {
            "tick": [10, 11, 500, 501, 12, 13],
            "thrower_steamid": [1, 1, 1, 1, 2, 2],
            "grenade_type": ["Smoke", "Smoke", "Smoke", "Smoke", "HeGrenade", "HeGrenade"],
        }
    )
    use_parser(FakeParser(grenades=g))
    out = demo.grenade_throws("x.dem")
    assert out[["tick", "thrower_steamid"]].values.tolist() == [[10, 1], [12, 2], [500, 1]]


def test_grenade_throws_empty(use_parser) -> None:
    use_parser(FakeParser())
    assert demo.grenade_throws("x.dem").empty


ACF = """"AppState"
{
\t"appid"\t\t"730"
\t"buildid"\t\t"19876543"
\t"TargetBuildID"\t\t"19999999"
\t"InstalledDepots"
\t{
\t\t"2347771"
\t\t{
\t\t\t"manifest"\t\t"123"
\t\t}
\t}
}
"""


def test_cs2_build_id_parses_acf_next_to_game_folder(tmp_path) -> None:
    game = tmp_path / "steamapps" / "common" / "Counter-Strike Global Offensive"
    game.mkdir(parents=True)
    (tmp_path / "steamapps" / "appmanifest_730.acf").write_text(ACF, encoding="utf-8")
    assert demo.cs2_build_id(game) == 19876543  # not the TargetBuildID


def test_cs2_build_id_finds_manifest_in_given_dir(tmp_path) -> None:
    (tmp_path / "appmanifest_730.acf").write_text(ACF, encoding="utf-8")
    assert demo.cs2_build_id(tmp_path) == 19876543


def test_cs2_build_id_errors(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        demo.cs2_build_id(tmp_path / "nope" / "game")
    (tmp_path / "appmanifest_730.acf").write_text('"AppState" { "appid" "730" }', encoding="utf-8")
    with pytest.raises(ValueError):
        demo.cs2_build_id(tmp_path)


def test_last_tick(use_parser) -> None:
    use_parser(FakeParser(ticks=tick_table()))
    assert demo.last_tick("x.dem") == 108


def test_inspect_summarises_players_and_events(use_parser) -> None:
    events = {
        "player_spawn": pd.DataFrame({"tick": [100, 100, 100]}),
        "player_death": pd.DataFrame({"tick": [108]}),
    }
    use_parser(FakeParser(ticks=tick_table(), events=events))
    info = demo.inspect("x.dem")
    json.dumps(info)  # must be JSON-serialisable
    assert info["first_tick"] == 100 and info["last_tick"] == 108
    assert info["events"] == {"player_spawn": 3, "player_death": 1, "weapon_fire": 0}
    assert [(p["user_id"], p["name"], p["steamid"], p["n_ticks"]) for p in info["players"]] == [
        (2, "Bot A", 0, 3),
        (3, "Bot B", 0, 3),
        (4, "Bot C", 7, 3),
    ]
    assert info["header"]["map_name"] == "de_dust2"
