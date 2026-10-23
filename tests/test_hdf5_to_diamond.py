"""Tests for the Mill dataset assembly scripts (#17): hdf5_to_diamond, dataset_stats, verify_manifests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from cs2wm.actions import ACTION_DIM, ActionSpec
from cs2wm.config import load_config
from cs2wm.pipeline.pack import append_manifest, read_episode, write_episode

REPO_ROOT = Path(__file__).resolve().parent.parent
H, W = 150, 280
CFG = load_config()
SPEC = ActionSpec.from_config(CFG)
X0, Y0 = SPEC.mouse_x_units.index(0.0), SPEC.mouse_y_units.index(0.0)


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def conv() -> ModuleType:
    return load_script("hdf5_to_diamond")


@pytest.fixture(scope="module")
def stats_mod() -> ModuleType:
    return load_script("dataset_stats")


@pytest.fixture(scope="module")
def verify() -> ModuleType:
    return load_script("verify_manifests")


def action(n: int, keys: dict[int, list[int]] | None = None, x: dict[int, int] | None = None) -> np.ndarray:
    """Actions with the mouse on the zero bins; ``keys`` maps slot -> frames pressed, ``x`` maps frame -> x bin."""
    a = np.zeros((n, ACTION_DIM), dtype=np.float32)
    for slot, frames in (keys or {}).items():
        a[frames, slot] = 1
    xs = np.full(n, X0)
    for frame, b in (x or {}).items():
        xs[frame] = b
    a[np.arange(n), SPEC.mouse_x_slice.start + xs] = 1
    a[np.arange(n), SPEC.mouse_y_slice.start + Y0] = 1
    return a


def add_episode(root: Path, kind: str, demo: str, episode_id: str, n: int, acts: np.ndarray, seed: int = 0) -> Path:
    frames = np.random.default_rng(seed).integers(0, 256, (n, H, W, 3), dtype=np.uint8)
    meta = {
        "episode_id": episode_id,
        "demo_id": demo,
        "kind": kind,
        "player": None,
        "start_tick": 0,
        "cs2_build": "1",
        "demoparser2_version": "0",
        "cs2wm_git_sha": "abc",
    }
    out_dir = root / kind / demo
    res = write_episode(frames, acts, meta, out_dir, CFG)
    append_manifest(
        out_dir / "manifest.jsonl",
        {"file": res.low_res_path.name, "frames": n, "sha256": res.sha256[res.low_res_path.name]},
    )
    return res.low_res_path


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """dm: 4 demos, human: 2, smoke: 5; episode i has i % 4 + 3 frames and its own key slot, so actions are unique."""
    root = tmp_path_factory.mktemp("hdf5")
    i = 0
    for kind, n_demos in (("dm", 4), ("human", 2), ("smoke", 5)):
        for d in range(n_demos):
            n = i % 4 + 3
            add_episode(
                root, kind, f"{kind}{d}", f"{kind}{d}_e0", n, action(n, keys={i % 11: [0, 1]}, x={0: i % 20}), i
            )
            i += 1
    return root


def convert(conv: ModuleType, tree: Path, out: Path, *extra: str) -> int:
    return conv.main(["--hdf5", str(tree), "--out", str(out), *extra])


def split_of(out: Path) -> dict[str, str]:
    return json.loads((out / "split.json").read_text())


def demo_kind(name: str) -> str:
    return name.rstrip("0123456789")


# Split


def test_split_is_disjoint_by_demo_and_complete(conv, tree, tmp_path) -> None:
    assert convert(conv, tree, tmp_path / "ds") == 0
    split = split_of(tmp_path / "ds")
    assert set(split) == {f"{k}{d}" for k, n in (("dm", 4), ("human", 2), ("smoke", 5)) for d in range(n)}
    assert set(split.values()) == {"train", "test"}
    meta = json.loads((tmp_path / "ds" / "dataset_meta.json").read_text())
    for row in meta["episodes"]:
        assert split[row["demo_id"]] == row["split"]
    train_demos = {r["demo_id"] for r in meta["episodes"] if r["split"] == "train"}
    test_demos = {r["demo_id"] for r in meta["episodes"] if r["split"] == "test"}
    assert not train_demos & test_demos


def test_split_is_deterministic_and_seed_dependent(conv, tree, tmp_path) -> None:
    splits = []
    for name, seed in (("a", 0), ("b", 0), ("c", 1), ("d", 2), ("e", 3)):
        convert(conv, tree, tmp_path / name, "--seed", str(seed))
        splits.append(split_of(tmp_path / name))
    assert splits[0] == splits[1]
    assert any(s != splits[0] for s in splits[2:])


@pytest.mark.parametrize("seed", range(4))
def test_smoke_demos_in_test_at_least_30_percent_and_3(conv, tree, tmp_path, seed) -> None:
    convert(conv, tree, tmp_path / "ds", "--seed", str(seed))
    smoke = [s for d, s in split_of(tmp_path / "ds").items() if demo_kind(d) == "smoke"]
    assert smoke.count("test") >= max(3, 0.3 * len(smoke))
    # Other kinds follow --test-frac (ceil, train keeps at least one demo).
    assert [s for d, s in split_of(tmp_path / "ds").items() if demo_kind(d) == "dm"].count("test") == 1


@pytest.mark.parametrize(
    ("kind", "n", "frac", "expected"),
    [
        ("smoke", 10, 0.1, 3),
        ("smoke", 20, 0.1, 6),  # ceil(0.3 * 20) with an exact tie
        ("smoke", 30, 0.1, 9),
        ("smoke", 10, 0.5, 5),
        ("smoke", 3, 0.1, 3),
        ("smoke", 2, 0.1, 2),  # fewer than 3: all to test
        ("smoke", 1, 0.1, 1),
        ("dm", 10, 0.1, 1),
        ("dm", 5, 0.1, 1),
        ("dm", 1, 0.1, 0),  # a single demo stays in train
        ("dm", 4, 0.99, 3),
        ("dm", 4, 0.0, 0),
    ],
)
def test_n_test_demos(conv, kind, n, frac, expected) -> None:
    assert conv.n_test_demos(kind, n, frac) == expected


def test_too_few_smoke_demos_all_go_to_test(conv, tmp_path, capsys) -> None:
    root = tmp_path / "hdf5"
    for kind, demo in (("dm", "dm0"), ("dm", "dm1"), ("smoke", "smoke0"), ("smoke", "smoke1")):
        add_episode(root, kind, demo, f"{demo}_e0", 3, action(3))
    assert convert(conv, root, tmp_path / "ds") == 0
    assert split_of(tmp_path / "ds")["smoke0"] == split_of(tmp_path / "ds")["smoke1"] == "test"
    assert "WARNING" in capsys.readouterr().out


def test_kinds_filter(conv, tree, tmp_path) -> None:
    convert(conv, tree, tmp_path / "ds", "--kinds", "dm,human")
    assert {demo_kind(d) for d in split_of(tmp_path / "ds")} == {"dm", "human"}


def test_errors_on_empty_input_and_existing_output(conv, tree, tmp_path) -> None:
    with pytest.raises(SystemExit, match="no episodes"):
        convert(conv, tree, tmp_path / "ds", "--kinds", "nope")
    convert(conv, tree, tmp_path / "ds")
    with pytest.raises(SystemExit, match="already exists"):
        convert(conv, tree, tmp_path / "ds")


# Derangement and shuffled actions


@pytest.mark.parametrize("n", [2, 3, 7, 50])
@pytest.mark.parametrize("seed", range(5))
def test_derangement_has_no_fixed_point(conv, n, seed) -> None:
    p = conv.derangement(n, seed)
    assert sorted(p) == list(range(n))
    assert all(p[i] != i for i in range(n))
    assert p == conv.derangement(n, seed)


def test_derangement_needs_two(conv) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        conv.derangement(1, 0)


def test_tile_to_length(conv) -> None:
    a = np.arange(6, dtype=np.float32).reshape(3, 2)
    assert np.array_equal(conv.tile_to_length(a, 2), a[:2])
    assert np.array_equal(conv.tile_to_length(a, 3), a)
    assert np.array_equal(conv.tile_to_length(a, 7), a[[0, 1, 2, 0, 1, 2, 0]])


def load_split(out: Path, split: str) -> list[dict]:
    meta = json.loads((out / "dataset_meta.json").read_text())
    rows = [r for r in meta["episodes"] if r["split"] == split]
    return [
        {**r, **torch.load(min((out / "low_res" / split).rglob(f"{r['index']}.pt")), weights_only=True)} for r in rows
    ]


def true_actions(tree: Path) -> dict[str, np.ndarray]:
    return {p.stem: read_episode(p)[1] for p in tree.rglob("*.hdf5")}


def test_shuffle_actions_derange_train_and_keep_test(conv, tree, tmp_path) -> None:
    convert(conv, tree, tmp_path / "ds", "--shuffle-actions")
    truth = true_actions(tree)
    train = load_split(tmp_path / "ds", "train")
    assert len({r["actions_from"] for r in train}) == len(train)
    for r in train:
        assert r["actions_from"] != r["episode_id"]
        assert r["actions_from"] in {t["episode_id"] for t in train}
        expected = conv.tile_to_length(truth[r["actions_from"]], r["frames"])
        assert r["act"].shape == (r["frames"], ACTION_DIM)
        assert np.array_equal(r["act"].numpy(), expected)
        # The unique key slot of the source differs from the episode's own, so the labels really moved.
        assert not np.array_equal(r["act"].numpy(), truth[r["episode_id"]])
    for r in load_split(tmp_path / "ds", "test"):
        assert r["actions_from"] == r["episode_id"]
        assert np.array_equal(r["act"].numpy(), truth[r["episode_id"]])


def test_plain_conversion_keeps_true_actions_and_frames_exactly(conv, tree, tmp_path) -> None:
    convert(conv, tree, tmp_path / "ds")
    paths = {p.stem: p for p in tree.rglob("*.hdf5")}
    for split in ("train", "test"):
        for r in load_split(tmp_path / "ds", split):
            frames, actions, _ = read_episode(paths[r["episode_id"]])
            assert np.array_equal(r["act"].numpy(), actions)
            # Exact uint8 round trip: RGB CHW pixels equal the HDF5 BGR HWC pixels, including the 63 values that
            # Episode.save's float round trip would truncate.
            assert r["obs"].dtype == torch.uint8
            assert np.array_equal(r["obs"].numpy(), frames[..., ::-1].transpose(0, 3, 1, 2))


def test_shuffle_with_single_train_episode_errors(conv, tmp_path) -> None:
    root = tmp_path / "hdf5"
    add_episode(root, "dm", "dm0", "dm0_e0", 3, action(3))
    add_episode(root, "dm", "dm1", "dm1_e0", 3, action(3))
    assert convert(conv, root, tmp_path / "plain") == 0  # 1 train + 1 test demo is fine without the flag
    with pytest.raises(SystemExit, match="at least 2 train episodes"):
        convert(conv, root, tmp_path / "ds", "--shuffle-actions")


def test_meta_records_provenance(conv, tree, tmp_path) -> None:
    convert(conv, tree, tmp_path / "ds", "--seed", "7", "--shuffle-actions")
    meta = json.loads((tmp_path / "ds" / "dataset_meta.json").read_text())
    assert meta["seed"] == 7
    assert meta["shuffle_actions"] is True
    assert meta["test_frac"] == 0.1
    assert len(meta["source_manifests"]) == 11
    assert all(len(v) == 64 for v in meta["source_manifests"].values())


# DIAMOND loading


def test_output_loads_with_diamond_dataset(conv, tree, tmp_path) -> None:
    convert(conv, tree, tmp_path / "ds")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from data import Dataset, SegmentId

    for split in ("train", "test"):
        ds = Dataset(tmp_path / "ds" / "low_res" / split, None, f"{split}_dataset")
        ds.load_from_default_path()
        rows = load_split(tmp_path / "ds", split)
        assert ds.num_episodes == len(rows)
        assert ds.num_steps == sum(r["frames"] for r in rows)
        assert ds.lengths.tolist() == [r["frames"] for r in sorted(rows, key=lambda r: r["index"])]
        for r in rows:
            seg = ds[SegmentId(r["index"], 0, r["frames"] + 1)]  # one step of right padding
            assert seg.obs.shape == (r["frames"] + 1, 3, 30, 56)
            assert seg.obs.dtype == torch.float32
            assert -1 <= seg.obs.min() and seg.obs.max() <= 1
            assert seg.act.shape == (r["frames"] + 1, ACTION_DIM)
            assert np.array_equal(seg.act[: r["frames"]].numpy(), r["act"].numpy())
            assert seg.mask_padding.tolist() == [True] * r["frames"] + [False]
            assert seg.id.episode_id == r["index"]
            assert seg.info["kind"] == r["kind"]
            assert seg.rew.sum() == 0 and seg.end.sum() == 0 and seg.trunc.sum() == 0
        assert ds.counts_end == [ds.num_steps, 0]


# Stats


@pytest.fixture(scope="module")
def stats_ds(conv, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two dm demos (4 and 2 frames) and four 1-frame-long smoke demos with hand-known labels."""
    root = tmp_path_factory.mktemp("stats_hdf5")
    # w (slot 0) on frames 0-1, lmb (slot 11) on frame 0; mouse x bin X0 + 1 on frame 3.
    add_episode(root, "dm", "dm0", "dm0_e0", 4, action(4, keys={0: [0, 1], 11: [0], 12: [2]}, x={3: X0 + 1}))
    # Entirely idle, plus one strictly zero row (frame 1) to exercise all_zero_frac.
    zero = action(2)
    zero[1] = 0
    add_episode(root, "dm", "dm1", "dm1_e0", 2, zero, 1)
    for d in range(4):
        add_episode(root, "smoke", f"smoke{d}", f"smoke{d}_e0", 2, action(2, keys={2: [0, 1]}), 2 + d)
    out = tmp_path_factory.mktemp("stats_out") / "ds"
    assert conv.main(["--hdf5", str(root), "--out", str(out), "--test-frac", "0.5"]) == 0
    return out


def test_dataset_stats_values(stats_mod, stats_ds, tmp_path, capsys) -> None:
    out_json = tmp_path / "stats.json"
    assert stats_mod.main([str(stats_ds), "--json", str(out_json)]) == 0
    printed = capsys.readouterr().out
    stats = json.loads(out_json.read_text())
    assert stats == stats_mod.compute_stats(stats_ds)
    assert "label rates" in printed and "episode length histogram" in printed

    total = stats["total"]
    assert total["episodes"] == 6 and total["frames"] == 4 + 2 + 4 * 2
    assert total["hours"] == pytest.approx(14 / 16 / 3600)
    assert stats["train"]["frames"] + stats["test"]["frames"] == 14
    assert stats["by_kind"]["dm"]["train"]["frames"] + stats["by_kind"]["dm"]["test"]["frames"] == 6
    assert stats["by_kind"]["smoke"]["train"]["frames"] + stats["by_kind"]["smoke"]["test"]["frames"] == 8
    assert stats["by_kind"]["smoke"]["test"]["episodes"] == 3  # max(3, ceil(0.3 * 4), ceil(0.5 * 4)) = 3

    rates = total["label_rates"]
    assert list(rates) == [*SPEC.keys, *SPEC.clicks]  # slot names come from the config
    assert rates["w"] == pytest.approx(2 / 14)
    assert rates["s"] == pytest.approx(8 / 14)  # slot 2 on 8 smoke frames
    assert rates["lmb"] == pytest.approx(1 / 14)
    assert rates["rmb"] == pytest.approx(1 / 14)

    centres = [f"{d:+.3f}" for d in SPEC.mouse_x_centers_deg]
    assert sum(total["mouse_x_hist"].values()) == 13  # the zero row has no mouse label
    assert total["mouse_x_hist"][centres[X0 + 1]] == 1
    assert total["mouse_x_hist"][centres[X0]] == 12
    assert (
        sum(total["mouse_y_hist"].values()) == 13
        and total["mouse_y_hist"][f"{SPEC.mouse_y_centers_deg[Y0]:+.3f}"] == 13
    )

    assert total["all_zero_frac"] == pytest.approx(1 / 14)
    # Idle: dm1's first row; dm0's frames 2 (rmb) and 3 (moved), the 8 smoke frames (slot s) and dm0's pressed frames are not.
    assert total["idle_frac"] == pytest.approx(1 / 14)
    assert total["length_hist"]["counts"][0] == 6 and sum(total["length_hist"]["counts"]) == 6


def test_dataset_stats_cli_errors(stats_mod, tmp_path) -> None:
    with pytest.raises(SystemExit):
        stats_mod.main([])
    with pytest.raises(FileNotFoundError):
        stats_mod.compute_stats(tmp_path)


# verify_manifests


def test_verify_manifests_ok(verify, tree, capsys) -> None:
    assert verify.main([str(tree)]) == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == "OK: 11 files verified in 11 manifests"


def test_verify_manifests_detects_mismatch_and_missing(verify, tmp_path, capsys) -> None:
    root = tmp_path / "hdf5"
    a = add_episode(root, "dm", "dm0", "dm0_e0", 3, action(3))
    b = add_episode(root, "dm", "dm0", "dm0_e1", 3, action(3), 1)
    add_episode(root, "dm", "dm1", "dm1_e0", 3, action(3), 2)
    assert verify.main([str(root)]) == 0
    capsys.readouterr()
    with open(a, "ab") as f:
        f.write(b"x")
    b.unlink()
    assert verify.main([str(root)]) == 1
    out = capsys.readouterr().out
    assert f"MISMATCH {a}" in out and f"MISSING {b}" in out
    assert out.strip().splitlines()[-1].startswith("FAILED: 2 problems; 1 files verified in 2 manifests")


def test_verify_manifests_bad_record_unlisted_file_and_no_manifest(verify, tmp_path, capsys) -> None:
    assert verify.main([str(tmp_path)]) == 1
    assert "no manifest.jsonl" in capsys.readouterr().out
    root = tmp_path / "hdf5"
    add_episode(root, "dm", "dm0", "dm0_e0", 3, action(3))
    (root / "dm" / "dm0" / "extra.hdf5").write_bytes(b"")
    assert verify.main([str(root)]) == 0
    assert "warning: not in manifest" in capsys.readouterr().out
    with open(root / "dm" / "dm0" / "manifest.jsonl", "a") as f:
        f.write('{"file": "../../../etc/passwd", "sha256": "0"}\nnot json\n{"frames": 1}\n')
    assert verify.main([str(root)]) == 1
    out = capsys.readouterr().out
    assert "points outside" in out and out.count("unreadable record") == 2


def test_verify_manifests_fullres_records(verify, tmp_path, capsys) -> None:
    root, fullres = tmp_path / "hdf5", tmp_path / "hdf5_fullres"
    d = fullres / "dm" / "dm0"
    d.mkdir(parents=True)
    (d / "e.hdf5").write_bytes(b"abc")
    append_manifest(
        root / "dm" / "dm0" / "manifest.jsonl",
        {"file": "fullres/e.hdf5", "frames": 1, "sha256": hashlib.sha256(b"abc").hexdigest()},
    )
    assert verify.main([str(root)]) == 0
    assert "1 full-res records skipped" in capsys.readouterr().out
    assert verify.main([str(root), "--fullres-root", str(fullres)]) == 0
    (d / "e.hdf5").write_bytes(b"abd")
    assert verify.main([str(root), "--fullres-root", str(fullres)]) == 1


# Cross-demo shuffle


@pytest.mark.parametrize("sizes", [[2, 2, 2], [3, 3], [1, 2, 3, 4], [2, 2, 2, 2, 2, 2, 1], [1, 1]])
@pytest.mark.parametrize("seed", range(5))
def test_assign_action_sources_avoids_own_demo(conv, sizes, seed) -> None:
    demos = [f"d{d}" for d, size in enumerate(sizes) for _ in range(size)]
    n = len(demos)
    source, method = conv.assign_action_sources(demos, seed)
    assert (source, method) == conv.assign_action_sources(demos, seed)
    assert sorted(source) == list(range(n))
    assert all(source[i] != i for i in range(n))
    if 2 * max(sizes) <= n:
        assert method == "cross_demo_shift"
        assert all(demos[source[i]] != demos[i] for i in range(n))
    else:
        assert method == "derangement_fallback"


def test_assign_action_sources_fallback_warns(conv, capsys) -> None:
    demos = ["a"] * 4 + ["b"]
    source, method = conv.assign_action_sources(demos, 0)
    assert method == "derangement_fallback" and all(source[i] != i for i in range(5))
    assert "WARNING" in capsys.readouterr().out
    with pytest.raises(ValueError, match="at least 2"):
        conv.assign_action_sources(["a"], 0)


def multi_tree(root: Path, per_demo: dict[str, int]) -> Path:
    i = 0
    for demo, count in per_demo.items():
        for e in range(count):
            add_episode(root, "dm", demo, f"{demo}_e{e}", 3 + i % 3, action(3 + i % 3, keys={i % 11: [0]}), i)
            i += 1
    return root


def test_shuffle_records_method_and_zero_same_demo_swaps(conv, tmp_path) -> None:
    root = multi_tree(tmp_path / "hdf5", {f"dm{d}": 2 for d in range(4)})
    convert(conv, root, tmp_path / "ds", "--shuffle-actions", "--test-frac", "0.25")
    meta = json.loads((tmp_path / "ds" / "dataset_meta.json").read_text())
    assert meta["shuffle"] == {"method": "cross_demo_shift", "same_demo_swaps": 0}
    by_id = {r["episode_id"]: r for r in meta["episodes"]}
    for r in meta["episodes"]:
        if r["split"] == "train":
            assert by_id[r["actions_from"]]["demo_id"] != r["demo_id"]


def test_shuffle_fallback_records_same_demo_swaps(conv, tmp_path, capsys) -> None:
    root = multi_tree(tmp_path / "hdf5", {"dm0": 4, "dm1": 1, "dm2": 1, "dm3": 1})
    # dm0 (4 of 7 episodes) must land in train for the fallback to be needed; pick a seed that does that.
    for seed in range(20):
        out = tmp_path / f"ds{seed}"
        convert(conv, root, out, "--shuffle-actions", "--test-frac", "0.25", "--seed", str(seed))
        if split_of(out)["dm0"] == "train":
            break
    else:
        pytest.fail("no seed puts dm0 in train")
    meta = json.loads((out / "dataset_meta.json").read_text())
    assert meta["shuffle"]["method"] == "derangement_fallback"
    assert "WARNING" in capsys.readouterr().out
    rows = {r["episode_id"]: r for r in meta["episodes"]}
    expected = sum(
        rows[r["actions_from"]]["demo_id"] == r["demo_id"] for r in meta["episodes"] if r["split"] == "train"
    )
    assert meta["shuffle"]["same_demo_swaps"] == expected
    assert all(r["actions_from"] != r["episode_id"] for r in meta["episodes"] if r["split"] == "train")


def test_plain_conversion_has_no_shuffle_info(conv, tree, tmp_path) -> None:
    convert(conv, tree, tmp_path / "ds")
    assert json.loads((tmp_path / "ds" / "dataset_meta.json").read_text())["shuffle"] is None


# Stats: long episodes and no obs reads


def test_length_hist_keeps_episodes_longer_than_max(stats_mod) -> None:
    acc = stats_mod._Acc(SPEC)
    acc.add(action(5))
    acc.add(action(1200))
    hist = acc.summary(16, 1000)["length_hist"]
    assert sum(hist["counts"]) == 2 and hist["counts"][-1] == 1 and hist["counts"][0] == 1


def test_dataset_stats_mmaps_episodes(stats_mod, stats_ds, monkeypatch) -> None:
    calls = []
    real = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: calls.append(k.get("mmap")) or real(*a, **k))
    stats_mod.compute_stats(stats_ds)
    assert calls and all(calls)


# Trainer-style batching


def test_batch_sampler_and_collate_build_trainer_batches(conv, tree, tmp_path, monkeypatch) -> None:
    convert(conv, tree, tmp_path / "ds")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from data import BatchSampler, Dataset, collate_segments_to_batch

    ds = Dataset(tmp_path / "ds" / "low_res" / "train", None, "train_dataset")
    ds.load_from_default_path()
    batch_size, seq_length = 4, 5
    # torch >= 2.2 dropped Sampler's data_source argument, which v1's BatchSampler still passes to super().__init__.
    monkeypatch.setattr(torch.utils.data.Sampler, "__init__", lambda self, *a, **k: None)
    sampler = BatchSampler(ds, 0, 1, batch_size, seq_length, None)
    it = iter(sampler)
    for _ in range(3):
        batch = collate_segments_to_batch([ds[sid] for sid in next(it)])
        assert batch.obs.shape == (batch_size, seq_length, 3, 30, 56) and batch.obs.dtype == torch.float32
        assert batch.act.shape == (batch_size, seq_length, ACTION_DIM) and batch.act.dtype == torch.float32
        assert batch.rew.shape == batch.end.shape == batch.trunc.shape == (batch_size, seq_length)
        assert batch.end.dtype == batch.trunc.dtype == torch.uint8
        assert batch.mask_padding.shape == (batch_size, seq_length) and batch.mask_padding.dtype == torch.bool
        assert batch.obs.min() >= -1 and batch.obs.max() <= 1
        assert len(batch.info) == batch_size and len(batch.segment_ids) == batch_size


def test_verify_manifests_empty_and_undecodable_manifests(verify, tmp_path, capsys) -> None:
    root = tmp_path / "hdf5"
    (root / "dm" / "dm0").mkdir(parents=True)
    (root / "dm" / "dm0" / "manifest.jsonl").write_text("\n\n")
    assert verify.main([str(root)]) == 1
    out = capsys.readouterr().out
    assert "empty manifest" in out and out.strip().splitlines()[-1].startswith("FAILED")
    (root / "dm" / "dm0" / "manifest.jsonl").write_bytes(b"\xff\xfe\x00{")
    assert verify.main([str(root)]) == 1
    out = capsys.readouterr().out
    assert "unreadable manifest" in out and out.strip().splitlines()[-1].startswith("FAILED")
