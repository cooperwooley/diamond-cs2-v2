"""Tests for the H2 smoke eval (#33): demo tables, masked metrics, context selection, statistics and the end-to-end run."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from agent import Agent
from cs2wm import cli, demo
from cs2wm.actions import ACTION_DIM
from cs2wm.config import load_config
from cs2wm.eval import h2_smoke as h2
from cs2wm.eval.upsampler_check import ssim, ssim_map
from cs2wm.pipeline.pack import write_episode
from cs2wm.play.agent import DEFAULT_AGENT_CONFIG_PATH, build_play_agent

H, W = 150, 280
STEPS = 3
START_TICK = 1000


@pytest.fixture(scope="module")
def cfg():
    return load_config(
        overrides=[
            "pack.fullres_mod=1",
            "eval.h2.bootstrap=50",
            "eval.h2.contexts=3",
            "eval.h2.min_smoke_frames=2",
            "eval.h2.examples=2",
        ]
    )


def fake_lpips(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """A stand-in for AlexNet LPIPS (no weights offline): the mean absolute error map."""
    return (pred - target).abs().mean(1)


# --- a fake demoparser2 --------------------------------------------------------------------------------------------


class FakeParser:
    """The parts of ``demoparser2.DemoParser`` that ``cs2wm.demo`` uses, answered from tables."""

    def __init__(self, ticks: pd.DataFrame, events: list[tuple[str, pd.DataFrame]]) -> None:
        self.ticks, self.events = ticks, dict(events)
        self.calls: list[tuple] = []

    def parse_ticks(self, wanted_props, *, players=None, ticks=None, prop_states=None):
        self.calls.append(("ticks", tuple(wanted_props), None if ticks is None else len(ticks)))
        df = self.ticks if ticks is None else self.ticks[self.ticks["tick"].isin(ticks)]
        return df.copy()

    def list_game_events(self):
        return list(self.events)

    def parse_events(self, names, *, player=None, other=None):
        self.calls.append(("events", tuple(names), tuple(player or ())))
        return [(n, self.events[n].copy()) for n in names]


def tick_table(first: int, last: int, bob_in_view: range = range(0), bob_dead: range = range(0)) -> pd.DataFrame:
    """Alice (user 7) stands at the origin facing +x; Bob (user 8) is behind her, or in front for ``bob_in_view``."""
    rows = []
    for t in range(first, last + 1):
        rows.append(
            {"tick": t, "name": "Alice", "user_id": 7, "X": 0.0, "Y": 0.0, "Z": 0.0, "yaw": 0.0, "pitch": 0.0}
            | {"duck_amount": 0.0, "health": 100}
        )
        front = t in bob_in_view
        rows.append(
            {"tick": t, "name": "Bob", "user_id": 8, "X": 800.0 if front else -800.0, "Y": 100.0 if front else 0.0}
            | {"Z": 0.0, "yaw": 0.0, "pitch": 0.0, "duck_amount": 0.0, "health": 0 if t in bob_dead else 100}
        )
    return pd.DataFrame(rows)


def smoke_event_tables(detonate: int = 1000, expire: int = 1100):
    det = pd.DataFrame(
        {"tick": [detonate], "entityid": [55], "x": [1000.0], "y": [0.0], "z": [64.0], "user_user_id": [8]}
    )
    exp = pd.DataFrame({"tick": [expire], "entityid": [55], "x": [1000.0], "y": [0.0], "z": [64.0]})
    return [(demo.SMOKE_DETONATE_EVENT, det), (demo.SMOKE_EXPIRED_EVENT, exp)]


# --- cs2wm.demo ----------------------------------------------------------------------------------------------------


def test_smoke_events_joins_detonate_and_expired_by_entity_id(monkeypatch) -> None:
    det = pd.DataFrame(
        {
            "tick": [100, 400, 500, 900],
            "entityid": [10, 11, 10, 12],  # entity 10 is recycled by the third smoke
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [5.0, 6.0, 7.0, 8.0],
            "z": [9.0, 10.0, 11.0, 12.0],
            "user_user_id": [3, 4, 3, 5],
        }
    )
    exp = pd.DataFrame({"tick": [1250, 300, 1650], "entityid": [11, 10, 10]})  # unsorted, no expiry for entity 12
    parser = FakeParser(pd.DataFrame(), [(demo.SMOKE_DETONATE_EVENT, det), (demo.SMOKE_EXPIRED_EVENT, exp)])
    monkeypatch.setattr(demo, "_make_parser", lambda path: parser)
    table = demo.smoke_events("x.dem")
    assert list(table.columns) == list(demo.SMOKE_COLUMNS)
    assert table["detonate_tick"].tolist() == [100, 400, 500, 900]
    # entity 10: the first smoke ends at 300 (not 1650), the recycled one at 1650
    assert table["expire_tick"].tolist() == [300, 1250, 1650, 900 + demo.SMOKE_DURATION_TICKS]
    assert table["entity_id"].tolist() == [10, 11, 10, 12] and table["thrower_user_id"].tolist() == [3, 4, 3, 5]
    assert table[["x", "y", "z"]].to_numpy().tolist() == det[["x", "y", "z"]].to_numpy().tolist()
    assert parser.calls == [("events", (demo.SMOKE_DETONATE_EVENT, demo.SMOKE_EXPIRED_EVENT), ("user_id",))]


def test_smoke_events_without_smokes_or_thrower(monkeypatch) -> None:
    monkeypatch.setattr(demo, "_make_parser", lambda path: FakeParser(pd.DataFrame(), []))
    empty = demo.smoke_events("x.dem")
    assert list(empty.columns) == list(demo.SMOKE_COLUMNS) and len(empty) == 0
    det = pd.DataFrame({"tick": [64], "entityid": [1], "x": [0.0], "y": [0.0], "z": [0.0]})
    monkeypatch.setattr(
        demo, "_make_parser", lambda path: FakeParser(pd.DataFrame(), [(demo.SMOKE_DETONATE_EVENT, det)])
    )
    row = demo.smoke_events("x.dem").iloc[0]
    assert row["thrower_user_id"] == -1 and row["expire_tick"] == 64 + demo.SMOKE_DURATION_TICKS


# --- masked metrics ------------------------------------------------------------------------------------------------


def test_ssim_map_mean_is_ssim() -> None:
    g = torch.Generator().manual_seed(0)
    a, b = torch.rand(2, 3, 40, 50, generator=g), torch.rand(2, 3, 40, 50, generator=g)
    m = ssim_map(a, b)
    assert m.shape == (2, 30, 40)
    assert torch.allclose(m.flatten(1).mean(1), ssim(a, b))


def test_masked_scores_see_only_the_region_that_differs() -> None:
    g = torch.Generator().manual_seed(1)
    target = torch.rand(2, 3, H, W, generator=g) * 2 - 1
    pred = target.clone()
    smoke = np.zeros((H, W), dtype=bool)
    smoke[40:90, 100:180] = True
    pred[:, :, 40:90, 100:180] = torch.rand(2, 3, 50, 80, generator=g) * 2 - 1  # wrong only inside the smoke
    lp, ss = h2.score_maps(pred, target, fake_lpips, torch.device("cpu"), chunk=1)
    assert lp.shape == ss.shape == (2, H, W) and np.isnan(ss[:, :5]).all() and np.isfinite(ss[:, 5:-5, 5:-5]).all()
    static = ~smoke
    static[:5] = static[-5:] = False
    static[:, :5] = static[:, -5:] = False
    for i in range(2):
        in_smoke, n_smoke = h2.masked_mean(lp[i], smoke)
        on_static, _ = h2.masked_mean(lp[i], ~smoke)
        assert n_smoke == smoke.sum() and in_smoke > 0.3 and on_static < 0.2 * in_smoke
        ssim_in, n_in = h2.masked_mean(ss[i], smoke)
        ssim_out, n_out = h2.masked_mean(ss[i], static)
        assert n_in == smoke.sum() and n_out < static.sum() + 1
        assert ssim_in < 0.3 and ssim_out > 0.6  # the 11 px windows around the smoke edge are partly wrong
    far = np.zeros((H, W), dtype=bool)
    far[:, :40] = True  # nothing wrong there
    assert h2.masked_mean(lp[0], far)[0] == 0.0
    assert h2.masked_mean(ss[0], far)[0] == pytest.approx(1.0, abs=1e-4)


def test_masked_mean_handles_no_mask_and_nan_only_regions() -> None:
    values = np.full((H, W), np.nan, dtype=np.float32)
    assert h2.masked_mean(values, None) == (pytest.approx(float("nan"), nan_ok=True), 0)
    mask = np.ones((H, W), dtype=bool)
    mean, n = h2.masked_mean(values, mask)
    assert np.isnan(mean) and n == 0
    values[3, 4] = 2.0
    assert h2.masked_mean(values, mask) == (2.0, 1)


# --- poses -----------------------------------------------------------------------------------------------------------


def attrs_for(**kw):
    return {"start_tick": START_TICK, "player": "Alice", "demo_id": "demoS", "kind": "smoke", "user_id": 7} | kw


def test_build_geoms_pose_eye_enemies_and_tick_mapping(cfg) -> None:
    table = tick_table(990, 1200, bob_in_view=range(1040, 1061), bob_dead=range(1100, 1121))
    table.loc[(table["name"] == "Alice") & (table["tick"] == 1008), ["yaw", "pitch", "duck_amount"]] = [90.0, 10.0, 1.0]
    tables = h2.DemoTables(pd.DataFrame(), table)
    geoms = h2.build_geoms(attrs_for(), 30, tables, cfg)
    assert len(geoms) == 30 and all(g is not None for g in geoms)
    assert [g.tick for g in geoms][:3] == [1000, 1004, 1008]  # start_tick + 4 k
    g = geoms[2]
    assert (g.yaw, g.pitch) == (90.0, 10.0) and np.allclose(g.eye, [0, 0, 46])  # fully ducked
    assert np.allclose(geoms[0].eye, [0, 0, 64]) and geoms[0].enemies.tolist() == [[-800.0, 0.0, 0.0]]
    assert geoms[10].tick == 1040 and geoms[10].enemies.tolist() == [[800.0, 100.0, 0.0]]
    dead = [g for g in geoms if 1100 <= g.tick <= 1120]
    assert dead and all(len(g.enemies) == 0 for g in dead)  # the dead are not enemies


def test_build_geoms_fills_small_gaps_and_marks_big_ones(cfg) -> None:
    table = tick_table(990, 1200)
    table = table[~table["tick"].isin([*range(1002, 1007), 1012])]  # five ticks around frame 1, one at frame 3
    geoms = h2.build_geoms(attrs_for(), 6, h2.DemoTables(pd.DataFrame(), table), cfg)
    assert geoms[1] is None  # the nearest tick is 3 away, more than qa.max_fill_ticks
    assert geoms[3].tick == 1011 and all(geoms[k] is not None for k in (0, 2, 4, 5))
    # 1004 is missing but 1003 and 1006 are present: max_fill_ticks = 2 allows 1002..1006
    table = tick_table(990, 1200)
    table = table[table["tick"] != 1004]
    assert h2.build_geoms(attrs_for(), 6, h2.DemoTables(pd.DataFrame(), table), cfg)[1].tick == 1003


def test_build_geoms_identifies_the_player_by_user_id_or_name(cfg) -> None:
    tables = h2.DemoTables(pd.DataFrame(), tick_table(990, 1100))
    by_name = h2.build_geoms(attrs_for(user_id=""), 3, tables, cfg)
    assert by_name[0].enemies.tolist() == [[-800.0, 0.0, 0.0]]
    by_bob = h2.build_geoms(attrs_for(user_id=8, player="Bob"), 3, tables, cfg)
    assert by_bob[0].enemies.tolist() == [[0.0, 0.0, 0.0]]  # Alice is Bob's enemy
    with pytest.raises(ValueError, match="cannot identify"):
        h2.build_geoms(attrs_for(user_id="", player=""), 3, tables, cfg)
    with pytest.raises(ValueError, match="not in the tick table"):
        h2.build_geoms(attrs_for(user_id=99), 3, tables, cfg)


def test_frame_regions_gives_smoke_and_drops_static_when_an_enemy_is_in_view(cfg) -> None:
    table = tick_table(990, 1200, bob_in_view=range(1040, 1061))
    smokes = pd.DataFrame(
        {"entity_id": [55], "detonate_tick": [1000], "expire_tick": [1100], "x": [1000.0], "y": [0.0], "z": [64.0]}
    )
    geoms = h2.build_geoms(attrs_for(), 30, h2.DemoTables(smokes, table), cfg)
    smoke, static = h2.frame_regions(geoms[0], smokes, cfg, (H, W))
    assert smoke.sum() > 600 and static is not None and not (smoke & static).any()
    smoke, static = h2.frame_regions(geoms[10], smokes, cfg, (H, W))  # tick 1040, Bob ahead
    assert smoke.any() and static is None
    smoke, static = h2.frame_regions(geoms[26], smokes, cfg, (H, W))  # tick 1104: smoke over
    assert not smoke.any() and static is not None
    assert h2.frame_regions(None, smokes, cfg, (H, W)) == (None, None)


# --- statistics and selection --------------------------------------------------------------------------------------


def make_rows(values: dict, n_ctx: int = 4, steps: int = 2):
    """Rows for one model/condition where context ``c`` has smoke LPIPS ``values['smoke'][c]`` etc."""
    rows = []
    for c in range(n_ctx):
        for s in range(steps):
            row = {col: float("nan") for col in h2.FRAME_COLUMNS} | {
                "model": "m",
                "condition": "rollout",
                "context": c,
                "step": s + 1,
                "smoke_px": 100,
                "static_px": 1000,
                "ssim_px_smoke": 80,
                "ssim_px_static": 900,
            }
            for key, v in values.items():
                row[key] = v[c]
            rows.append(row)
    return rows


def test_summarize_rows_means_counts_and_paired_difference() -> None:
    vals = {
        "lpips_smoke": [0.5, 0.6, 0.7, 0.8],
        "lpips_static": [0.2, 0.3, 0.4, 0.5],
        "ssim_smoke": [0.3, 0.4, 0.5, 0.6],
        "ssim_static": [0.6, 0.7, 0.8, 0.9],
    }
    res = h2.summarize_rows(make_rows(vals), n_boot=200, seed=0)["m"]["rollout"]
    smoke, static, diff = res["smoke"]["lpips"], res["static"]["lpips"], res["smoke_minus_static"]["lpips"]
    assert smoke["mean"] == pytest.approx(0.65) and static["mean"] == pytest.approx(0.35)
    assert smoke["ci_lo"] < 0.65 < smoke["ci_hi"] and 0.5 <= smoke["ci_lo"] and smoke["ci_hi"] <= 0.8
    assert (smoke["n_contexts"], smoke["n_frames"], smoke["n_pixels"]) == (4, 8, 800)
    assert (static["n_pixels"], res["smoke"]["ssim"]["n_pixels"], res["static"]["ssim"]["n_pixels"]) == (
        8000,
        640,
        7200,
    )
    assert diff["mean"] == pytest.approx(0.3) and diff["ci_lo"] == pytest.approx(0.3) == diff["ci_hi"]  # constant gap
    assert res["smoke_minus_static"]["ssim"]["mean"] == pytest.approx(-0.3) and diff["n_contexts"] == 4


def test_summarize_rows_skips_missing_regions_and_pairs_only_complete_contexts() -> None:
    nan = float("nan")
    vals = {
        "lpips_smoke": [0.5, 0.6, nan, 0.8],
        "lpips_static": [0.2, nan, 0.4, 0.5],
        "ssim_smoke": [nan] * 4,
        "ssim_static": [nan] * 4,
    }
    res = h2.summarize_rows(make_rows(vals), n_boot=50, seed=0)["m"]["rollout"]
    assert res["smoke"]["lpips"]["n_contexts"] == 3 and res["static"]["lpips"]["n_contexts"] == 3
    assert res["smoke_minus_static"]["lpips"]["n_contexts"] == 2  # contexts 0 and 3 have both regions
    assert res["smoke_minus_static"]["lpips"]["mean"] == pytest.approx(0.3)
    ssim_cell = res["smoke"]["ssim"]
    assert np.isnan(ssim_cell["mean"]) and ssim_cell["n_contexts"] == 0 and ssim_cell["n_frames"] == 0


def fake_episode(n: int, smoke_frames: range, path: str = "e") -> h2.EpisodeInfo:
    px = np.zeros(n, dtype=np.int64)
    px[list(smoke_frames)] = 500
    return h2.EpisodeInfo(Path(path), {}, n, h2.DemoTables(pd.DataFrame(), pd.DataFrame()), [None] * n, px, {})


def test_select_contexts_is_deterministic_spread_and_non_overlapping() -> None:
    eps = [fake_episode(60, range(10, 50), "a"), fake_episode(60, range(60), "b"), fake_episode(60, range(0), "c")]
    a = h2.select_contexts(eps, 6, 3, 50, 2, seed=0)
    b = h2.select_contexts(eps, 6, 3, 50, 2, seed=0)
    c = h2.select_contexts(eps, 6, 3, 50, 2, seed=1)
    assert [(x.episode.path, x.start) for x in a] == [(x.episode.path, x.start) for x in b]
    assert [(x.episode.path, x.start) for x in a] != [(x.episode.path, x.start) for x in c]
    assert [x.index for x in a] == list(range(6)) and {x.episode.path.name for x in a} == {"a", "b"}
    for ep in ("a", "b"):
        starts = sorted(x.start for x in a if x.episode.path.name == ep)
        assert all(y - x >= h2.N_COND + 3 for x, y in itertools.pairwise(starts))  # windows do not overlap
    for x in a:  # every window has >= 2 target frames with smoke
        targets = x.episode.smoke_px[x.start + 4 : x.start + 4 + 3]
        assert (targets >= 50).sum() >= 2 and x.start + 7 <= x.episode.n_frames
    assert [x.episode.path.name for x in a] == sorted(x.episode.path.name for x in a)  # sorted by episode


def test_select_contexts_returns_fewer_when_data_run_out_and_errors_without_smoke() -> None:
    eps = [fake_episode(20, range(20), "a")]
    got = h2.select_contexts(eps, 50, 3, 50, 2, seed=0)
    assert 1 < len(got) < 50 and len({x.start for x in got}) == len(got)
    with pytest.raises(ValueError, match="no window"):
        h2.select_contexts([fake_episode(20, range(0))], 2, 3, 50, 2, seed=0)
    with pytest.raises(ValueError, match="no window"):
        h2.select_contexts([fake_episode(5, range(5))], 2, 3, 50, 2, seed=0)  # shorter than 4 + steps
    with pytest.raises(ValueError, match="at least 1"):
        h2.select_contexts(eps, 0, 3, 50, 2, seed=0)


def test_overlay_marks_smoke_static_and_box(cfg) -> None:
    from cs2wm.eval import smoke_masks as sm

    frame = np.full((H, W, 3), 100, dtype=np.uint8)
    smoke = np.zeros((H, W), dtype=bool)
    smoke[40:60, 40:80] = True
    box = sm.viewmodel_mask(cfg, (H, W))
    img = h2.overlay_masks(frame, smoke, ~smoke & ~box, box, "label")
    k = h2.OVERLAY_SCALE
    assert img.shape == (H * k, W * k, 3) and img.dtype == np.uint8
    inside = img[50 * k, 60 * k]
    assert inside[2] > inside[0] + 30  # tinted red (BGR)
    assert (img == np.array(h2.STATIC_OUTLINE, dtype=np.uint8)).all(axis=-1).any()  # outline drawn
    assert (img == np.array(h2.BOX_COLOUR, dtype=np.uint8)).all(axis=-1).any()  # box drawn
    assert np.array_equal(img[100 * k, 5 * k + 2], [100, 100, 100])  # elsewhere untouched
    no_static = h2.overlay_masks(frame, smoke, None, box, "label")
    assert not (no_static == np.array(h2.STATIC_OUTLINE, dtype=np.uint8)).all(axis=-1).any()
    panel = h2.mask_panel(smoke, ~smoke & ~box, (H, W))
    assert (
        tuple(panel[50, 60]) == (0, 0, 220)
        and tuple(panel[5, 5]) == (0, 140, 0)
        and tuple(panel[-1, W // 2]) == (40,) * 3
    )


# --- end to end ----------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agent_cfg(tiny_model_dotlist):
    return OmegaConf.merge(OmegaConf.load(DEFAULT_AGENT_CONFIG_PATH), OmegaConf.from_dotlist(tiny_model_dotlist()))


def make_agent(agent_cfg, seed: int):
    """A tiny agent with random weights (DIAMOND zero-initialises its output layers; randomise those too)."""
    torch.manual_seed(seed)
    agent = Agent(instantiate(agent_cfg, num_actions=ACTION_DIM))
    for p in agent.parameters():
        if p.ndim > 1 and not p.abs().max() > 0:
            torch.nn.init.normal_(p, std=0.1)
    return agent.eval().requires_grad_(False)


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory, agent_cfg):
    root = tmp_path_factory.mktemp("ckpts")
    out = {}
    for i, name in enumerate(("v2", "v1")):
        torch.save(make_agent(agent_cfg, i + 1).state_dict(), root / f"{name}.pt")
        out[name] = root / f"{name}.pt"
    return out


def write_smoke_episode(root: Path, name: str, cfg, start_tick: int, kind: str = "smoke", demo_id: str = "demoS", n=30):
    rng = np.random.default_rng(start_tick)
    frames = rng.integers(0, 256, size=(n, H, W, 3), dtype=np.uint8)
    actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
    actions[:, 0] = 1.0  # W held
    meta = {
        "episode_id": name,
        "demo_id": demo_id,
        "kind": kind,
        "player": "Alice",
        "user_id": 7,
        "start_tick": start_tick,
        "cs2_build": "1",
        "demoparser2_version": "0.42.0",
        "cs2wm_git_sha": "abc1234",
    }
    write_episode(frames, actions, meta, root / "low", cfg, fullres_dir=root / "full")


@pytest.fixture(scope="module")
def smoke_data(tmp_path_factory, cfg):
    """Two smoke episodes of one demo (and a dm episode), full-res twins, and an empty ``demoS.dem``."""
    root = tmp_path_factory.mktemp("smoke")
    write_smoke_episode(root, "smoke-demoS-s000-c00", cfg, 1000)
    write_smoke_episode(root, "smoke-demoS-s000-c01", cfg, 1040)
    write_smoke_episode(root, "dm-demoD-s000-c00", cfg, 1000, kind="dm", demo_id="demoD")
    (root / "demos").mkdir()
    (root / "demos" / "demoS.dem").write_bytes(b"")
    (root / "demos" / "demoD.dem").write_bytes(b"")
    return root


@pytest.fixture
def parser(monkeypatch):
    p = FakeParser(tick_table(980, 1200, bob_in_view=range(1060, 1101)), smoke_event_tables())
    monkeypatch.setattr(demo, "_make_parser", lambda path: p)
    return p


def run(ckpts, smoke_data, out, cfg, agent_cfg, **kw):
    def factory(name, ckpt):
        return build_play_agent(ckpt, ckpt if name == "v1" else ckpts["v1"], agent_cfg=agent_cfg)

    kw = {"seed": 0, "batch_size": 4, "device": "cpu", "rollout_steps": STEPS, "n_validate": 5, **kw}
    return h2.run_h2(
        dict(ckpts),
        cfg,
        out,
        data_dir=smoke_data / "full",
        demos_dir=smoke_data / "demos",
        events_dir=smoke_data / "no-events",
        agent_factory=factory,
        lpips_map_fn=fake_lpips,
        **kw,
    )


def no_nan_constant(x):
    raise AssertionError(f"non-finite number {x} in summary.json")


def test_run_h2_end_to_end(ckpts, smoke_data, parser, tmp_path, cfg, agent_cfg, capsys) -> None:
    summary = run(ckpts, smoke_data, tmp_path, cfg, agent_cfg)
    on_disk = json.loads((tmp_path / "summary.json").read_text(), parse_constant=no_nan_constant)
    assert on_disk["schema_version"] == 1 and on_disk["seed"] == 0
    assert set(on_disk) >= {"config", "data", "mask_stats", "models", "results", "warnings", "limitations", "versions"}
    assert on_disk["config"]["rollout_steps"] == STEPS and on_disk["config"]["smoke_radius_units"] == 144
    assert on_disk["data"]["n_episodes"] == 2 and not on_disk["data"]["held_out"]
    assert [s["episode"] for s in on_disk["data"]["skipped"]] == ["dm-demoD-s000-c00"]
    assert len(on_disk["data"]["contexts"]) == 3
    assert any("NOT HELD OUT" in w for w in on_disk["warnings"])
    assert any("Smoke masks are geometric approximations" in x for x in on_disk["limitations"])
    stats = on_disk["mask_stats"]
    assert stats["frames"] == 60 and stats["no_pose"] == 0 and stats["with_smoke"] > 20
    assert 0 < stats["enemy_excluded"] < stats["frames"] and 0 < stats["mean_smoke_fraction"] < 0.2
    assert set(on_disk["models"]) == set(on_disk["results"]) == {"v2", "v1"}
    assert on_disk["models"]["v2"]["upsampler"]["path"] == str(ckpts["v1"])
    assert on_disk["models"]["v1"]["upsampler"]["path"] == str(ckpts["v1"])
    for model in ("v2", "v1"):
        for cond in ("single", "rollout"):
            res = on_disk["results"][model][cond]
            assert set(res) == {"smoke", "static", "smoke_minus_static"}
            for region in ("smoke", "static"):
                for metric in ("lpips", "ssim"):
                    cell = res[region][metric]
                    assert set(cell) == {"mean", "ci_lo", "ci_hi", "n_contexts", "n_frames", "n_pixels"}
                    assert cell["ci_lo"] <= cell["mean"] <= cell["ci_hi"] and cell["n_frames"] > 0
                    assert cell["n_pixels"] > 0 and cell["n_contexts"] >= 1
            diff = res["smoke_minus_static"]["lpips"]
            assert set(diff) == {"mean", "ci_lo", "ci_hi", "n_contexts"}
            # the smoke region is small and the static region is large, so pixel counts differ a lot
            assert res["static"]["lpips"]["n_pixels"] > res["smoke"]["lpips"]["n_pixels"]
    assert (
        summary["results"]["v2"]["rollout"]["smoke"]["lpips"]["mean"]
        == on_disk["results"]["v2"]["rollout"]["smoke"]["lpips"]["mean"]
    )

    lines = (tmp_path / "frames.csv").read_text().splitlines()
    header = lines[0].split(",")
    assert header == list(h2.FRAME_COLUMNS) and len(lines) == 1 + 2 * 2 * 3 * STEPS  # models x conditions x ctx x steps
    table = pd.read_csv(tmp_path / "frames.csv")
    assert set(table["status"]) <= {"ok", "enemy"} and "enemy" in set(table["status"])
    enemy = table[table["status"] == "enemy"]
    assert enemy["lpips_static"].isna().all() and enemy["lpips_smoke"].notna().any() and (enemy["static_px"] == 0).all()
    assert (table["frame"] == table["start"] + 4 + table["step"] - 1).all()
    assert (table["tick"] == 1000 + 4 * table["frame"] + 40 * table["episode"].str.endswith("c01")).all()

    overlays = sorted((tmp_path / "mask_validation").glob("*.png"))
    assert len(overlays) == 5
    triptychs = sorted((tmp_path / "triptychs").glob("*.png"))
    assert len(triptychs) == 2 * 2 * 2  # models x examples x conditions
    for model in ("v2", "v1"):
        for ext in ("pdf", "png"):
            assert (tmp_path / f"{model}_triptychs.{ext}").stat().st_size > 0
    assert set(on_disk["files"]) >= {"summary.json", "frames.csv", "v2_triptychs.pdf"}
    assert "rollout" in capsys.readouterr().out
    # the demo was parsed once for the two episodes
    assert [c[0] for c in parser.calls].count("ticks") == 1 and [c[0] for c in parser.calls].count("events") == 1


def test_single_step_one_is_rollout_step_one_and_models_share_contexts(
    ckpts, smoke_data, parser, tmp_path, cfg, agent_cfg
) -> None:
    run(ckpts, smoke_data, tmp_path, cfg, agent_cfg)
    table = pd.read_csv(tmp_path / "frames.csv")
    first = table[table["step"] == 1].pivot_table(index=["model", "context"], columns="condition", values="lpips_smoke")
    assert np.allclose(first["single"], first["rollout"], equal_nan=True)  # same context, action and noise
    later = table[table["step"] == STEPS].pivot_table(
        index=["model", "context"], columns="condition", values="lpips_static"
    )
    assert not np.allclose(later["single"], later["rollout"], equal_nan=True)  # errors accumulate in the rollout
    v2, v1 = (table[table["model"] == m].sort_values(["condition", "context", "step"]) for m in ("v2", "v1"))
    assert (
        v2[["episode", "frame", "tick", "smoke_px", "static_px"]].to_numpy()
        == v1[["episode", "frame", "tick", "smoke_px", "static_px"]].to_numpy()
    ).all()
    assert not np.allclose(v2["lpips_smoke"], v1["lpips_smoke"], equal_nan=True)  # but the models differ


def test_run_h2_is_reproducible_seed_dependent_and_batch_size_invariant(
    ckpts, smoke_data, parser, tmp_path, cfg, agent_cfg
) -> None:
    a = run(ckpts, smoke_data, tmp_path / "a", cfg, agent_cfg)
    b = run(ckpts, smoke_data, tmp_path / "b", cfg, agent_cfg)
    c = run(ckpts, smoke_data, tmp_path / "c", cfg, agent_cfg, seed=5)
    d = run(ckpts, smoke_data, tmp_path / "d", cfg, agent_cfg, batch_size=1)
    assert a["results"] == b["results"]
    assert a["results"] != c["results"]
    for model in ("v2", "v1"):
        for cond in ("single", "rollout"):
            x, y = a["results"][model][cond]["smoke"]["lpips"], d["results"][model][cond]["smoke"]["lpips"]
            assert x["mean"] == pytest.approx(y["mean"], abs=1e-5)


def test_run_h2_uses_saved_event_tables_and_split_json(ckpts, smoke_data, parser, tmp_path, cfg, agent_cfg) -> None:
    events = tmp_path / "events"
    events.mkdir()
    pd.DataFrame(
        {"entity_id": [1], "detonate_tick": [5000], "expire_tick": [6000], "x": [0.0], "y": [0.0], "z": [0.0]}
    ).to_parquet(events / "demoS.parquet")  # a smoke that never overlaps the episodes
    with pytest.raises(ValueError, match="no window"):
        h2.run_h2(
            dict(ckpts),
            cfg,
            tmp_path / "out",
            data_dir=smoke_data / "full",
            demos_dir=smoke_data / "demos",
            events_dir=events,
            rollout_steps=STEPS,
            device="cpu",
            agent_factory=lambda *a: None,
            lpips_map_fn=fake_lpips,
        )
    assert [c[0] for c in parser.calls].count("events") == 0  # the saved table replaced the parse
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"demoS": "test", "demoD": "train"}))
    s = run(ckpts, smoke_data, tmp_path / "held", cfg, agent_cfg, split_json=split)
    assert s["data"]["held_out"] is True and s["data"]["split_json"] == str(split)
    assert not any("NOT HELD OUT" in w for w in s["warnings"]) and s["data"]["skipped"] == []


def test_run_h2_input_errors(ckpts, smoke_data, parser, tmp_path, cfg, agent_cfg) -> None:
    with pytest.raises(ValueError, match="data_dir"):
        h2.run_h2(dict(ckpts), cfg, tmp_path)
    with pytest.raises(ValueError, match="no models"):
        h2.run_h2({}, cfg, tmp_path, data_dir=smoke_data / "full")
    with pytest.raises(ValueError, match="upsampler"):
        h2.run_h2({"v2": ckpts["v2"]}, cfg, tmp_path, data_dir=smoke_data / "full")
    with pytest.raises(ValueError, match="rollout_steps"):
        run(ckpts, smoke_data, tmp_path, cfg, agent_cfg, rollout_steps=0)
    with pytest.raises(ValueError, match="no usable smoke episode"):  # only the dm episode is left
        h2.run_h2(
            dict(ckpts),
            cfg,
            tmp_path,
            data_dir=smoke_data / "full",
            demos_dir=smoke_data / "no-demos",
            agent_factory=lambda *a: None,
        )
    with pytest.raises(ValueError, match="150x280"):
        h2.run_h2(
            dict(ckpts),
            cfg,
            tmp_path,
            data_dir=smoke_data / "low",
            demos_dir=smoke_data / "demos",
            agent_factory=lambda *a: None,
            events_dir=smoke_data / "no-events",
        )


# --- cli -----------------------------------------------------------------------------------------------------------


def test_cli_h2_passes_options(monkeypatch, tmp_path) -> None:
    seen = {}

    def fake_run(models, cfg, out, seed, **kw):
        seen.update(models=models, out=out, seed=seed, cfg=cfg, **kw)
        return {}

    monkeypatch.setattr(h2, "run_h2", fake_run)
    argv = ["eval", "h2", "--models", "v2=a.pt", "v1=b.pt", "--seed", "3", "--data", "d", "--split-json", "s.json"]
    argv += ["--demos", "dem", "--upsampler-ckpt", "u.pt", "--v1-name", "base", "--out", str(tmp_path)]
    argv += ["--device", "cpu", "--batch-size", "4", "--rollout-steps", "8", "--n-validate", "7", "--contexts", "9"]
    assert cli.main(argv) == 0
    assert seen["models"] == {"v2": Path("a.pt"), "v1": Path("b.pt")} and (seen["seed"], seen["out"]) == (3, tmp_path)
    assert (seen["data_dir"], seen["split_json"], seen["demos_dir"]) == (Path("d"), Path("s.json"), Path("dem"))
    assert (seen["upsampler_ckpt"], seen["v1_name"], seen["device"], seen["batch_size"]) == (
        Path("u.pt"),
        "base",
        "cpu",
        4,
    )
    assert (seen["rollout_steps"], seen["n_validate"], seen["contexts"]) == (8, 7, 9)
    cli.main(["eval", "h2", "--models", "v2=a.pt", "--data", "d"])
    assert seen["out"] == Path("results/h2") and seen["rollout_steps"] == 16 and seen["n_validate"] == 20
    assert seen["demos_dir"] is None and seen["v1_name"] == "v1" and seen["contexts"] is None


def test_cli_h2_requires_data() -> None:
    with pytest.raises(SystemExit):
        cli.main(["eval", "h2", "--models", "v2=a.pt"])
