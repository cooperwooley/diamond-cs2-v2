"""Tests for scripts/shift_labels.py (#88): drop the last frame and the first action of already packed episodes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pytest

from cs2wm.actions import ACTION_DIM, ActionSpec, bin_index
from cs2wm.config import load_config
from cs2wm.pipeline.pack import read_episode, sha256_file, write_episode, write_manifest
from tests.test_render_offset import TPF, synthetic_render

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG = load_config(overrides=["pack.fullres_mod=1"])  # every episode also gets a full-res file
SPEC = ActionSpec.from_config(CFG)
N = 6
KIND = "dm"
DEMO = "dm_demo_a"
OTHER = "dm_demo_b"


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def shift() -> ModuleType:
    return _load_script("shift_labels")


@pytest.fixture(scope="module")
def verify() -> ModuleType:
    return _load_script("verify_manifests")


def _frames(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, 150, 280, 3), dtype=np.uint8)


def _actions(n: int, seed: int) -> np.ndarray:
    a = np.zeros((n, ACTION_DIM), dtype=np.float32)
    rng = np.random.default_rng(seed)
    a[np.arange(n), SPEC.mouse_x_slice.start + rng.integers(0, len(SPEC.mouse_x_units), n)] = 1
    a[np.arange(n), SPEC.mouse_y_slice.start + SPEC.mouse_y_units.index(0.0)] = 1
    a[:, SPEC.keys.index("w")] = rng.integers(0, 2, n)  # something else that tells the frames apart
    return a


def _pack_demo(
    root: Path, demo_id: str, n_episodes: int = 2, build: int = 26598758, data: list | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pack episodes the way ``pack`` does (files + manifest); returns ``{episode_id: (frames, actions)}``."""
    low, full = root / "hdf5" / KIND / demo_id, root / "hdf5_fullres" / KIND / demo_id
    made, records = {}, []
    for i in range(n_episodes):
        ep = f"{KIND}-{demo_id}-s{i:03d}-c00"
        frames, actions = data[i] if data else (_frames(N, i), _actions(N, i))
        meta = {
            "episode_id": ep,
            "demo_id": demo_id,
            "kind": KIND,
            "player": "Bot",
            "user_id": 3,
            "start_tick": 1000 + 100 * i,
            "cs2_build": build,
            "demoparser2_version": "x",
            "cs2wm_git_sha": "abc",
        }
        result = write_episode(frames, actions, meta, low, CFG, fullres_dir=full)
        assert result.fullres_path is not None
        made[ep] = (frames, actions)
        records += [{"file": name, "frames": len(frames), "sha256": sha} for name, sha in result.sha256.items()]
    write_manifest(low / "manifest.jsonl", records)
    return made


def _files(root: Path, demo_id: str) -> list[Path]:
    return sorted((root / "hdf5").glob(f"*/{demo_id}/*.hdf5")) + sorted(
        (root / "hdf5_fullres").glob(f"*/{demo_id}/*.hdf5")
    )


def _digests(root: Path, demo_id: str) -> dict[str, str]:
    manifest = root / "hdf5" / KIND / demo_id / "manifest.jsonl"
    return {str(p.relative_to(root)): sha256_file(p) for p in [*_files(root, demo_id), manifest]}


def _verify(verify: ModuleType, root: Path, demo_id: str) -> None:
    manifest = root / "hdf5" / KIND / demo_id / "manifest.jsonl"
    checked, skipped, problems, unlisted = verify.check_manifest(manifest, root / "hdf5", root / "hdf5_fullres")
    assert problems == [] and unlisted == [] and checked > 0 and skipped == 0


def _demos_file(tmp_path: Path, *ids: str) -> Path:
    path = tmp_path / "demos.txt"
    path.write_text("# the shifted ones\n" + "\n".join(ids) + "\n\n", encoding="utf-8")
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "synced"
    _pack_demo(root, DEMO)
    _pack_demo(root, OTHER)
    return root


def _run(shift: ModuleType, root: Path, demos: Path, *flags: str) -> int:
    build = ["--build", "26598758"] if "--apply" in flags and "--build" not in flags else []
    return shift.main(["--root", str(root), "--demos", str(demos), *build, *flags])


def test_shift_drops_the_last_frame_and_the_first_action(shift, root, tmp_path) -> None:
    before = {p.relative_to(root): read_episode(p) for p in _files(root, DEMO)}
    assert _run(shift, root, _demos_file(tmp_path, DEMO), "--apply") == 0
    assert len(before) == 4  # two low-res and two full-res files
    for path in _files(root, DEMO):
        frames0, actions0, attrs0 = before[path.relative_to(root)]
        frames, actions, attrs = read_episode(path)
        assert len(frames) == N - 1 and attrs["n_frames"] == N - 1
        assert np.array_equal(frames, frames0[:-1]) and np.array_equal(actions, actions0[1:])
        assert attrs["start_tick"] == attrs0["start_tick"] + TPF and attrs["label_shift_frames"] == 1
        assert {k: v for k, v in attrs.items() if k not in ("n_frames", "start_tick", "label_shift_frames")} == {
            k: v for k, v in attrs0.items() if k not in ("n_frames", "start_tick")
        }  # episode_id, user_id, encoding, jpeg_quality, config_sha256, ... as before
        with h5py.File(path, "r") as f:
            assert sorted(f.keys()) == sorted(f"frame_{i}_{c}" for i in range(N - 1) for c in "xy")


def test_fullres_jpeg_bytes_are_copied_not_reencoded(shift, root, tmp_path) -> None:
    full = sorted((root / "hdf5_fullres").glob("*/*/*.hdf5"))
    stored = {}
    for path in full:
        with h5py.File(path, "r") as f:
            stored[path.name] = [f[f"frame_{i}_x"][()].tobytes() for i in range(N)]
    assert _run(shift, root, _demos_file(tmp_path, DEMO), "--apply") == 0
    for path in full:
        if path.parent.name != DEMO:
            continue
        with h5py.File(path, "r") as f:
            assert f.attrs["encoding"] == "jpeg"
            assert [f[f"frame_{i}_x"][()].tobytes() for i in range(N - 1)] == stored[path.name][: N - 1]


def test_manifest_verifies_after_the_shift_and_records_the_new_frame_counts(shift, verify, root, tmp_path) -> None:
    assert _run(shift, root, _demos_file(tmp_path, DEMO), "--apply") == 0
    _verify(verify, root, DEMO)
    _verify(verify, root, OTHER)
    lines = (root / "hdf5" / KIND / DEMO / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == 4 and all('"frames": 5' in line for line in lines)
    assert any('"file": "fullres/' in line for line in lines)


def test_other_demos_are_not_touched(shift, root, tmp_path) -> None:
    other = _digests(root, OTHER)
    assert _run(shift, root, _demos_file(tmp_path, DEMO), "--apply") == 0
    assert _digests(root, OTHER) == other


def test_dry_run_is_the_default_and_writes_nothing(shift, root, tmp_path, capsys) -> None:
    before = _digests(root, DEMO)
    assert _run(shift, root, _demos_file(tmp_path, DEMO)) == 0
    out = capsys.readouterr().out
    assert _digests(root, DEMO) == before
    assert "would shift" in out and "DRY RUN" in out and "would rewrite 4 record(s)" in out


def test_a_second_run_changes_nothing(shift, verify, root, tmp_path, capsys) -> None:
    demos = _demos_file(tmp_path, DEMO)
    assert _run(shift, root, demos, "--apply") == 0
    once = _digests(root, DEMO)
    capsys.readouterr()
    assert _run(shift, root, demos, "--apply") == 0
    assert _digests(root, DEMO) == once
    out = capsys.readouterr().out
    assert "skipped" in out and "shifted" not in out.replace("would shift", "")  # all four files were skipped
    assert "rewrote 0 record(s)" in out
    _verify(verify, root, DEMO)


def test_a_crash_before_the_manifest_is_repaired_by_the_next_run(shift, verify, root, tmp_path, monkeypatch) -> None:
    demos = _demos_file(tmp_path, DEMO)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(shift, "write_manifest", boom)
    assert _run(shift, root, demos, "--apply") == 1  # the files are shifted, the manifest is stale
    monkeypatch.undo()
    assert all(read_episode(p)[2]["label_shift_frames"] == 1 for p in _files(root, DEMO))
    assert _run(shift, root, demos, "--apply") == 0
    _verify(verify, root, DEMO)
    assert all(read_episode(p)[2]["n_frames"] == N - 1 for p in _files(root, DEMO))  # shifted once, not twice


def test_no_temp_files_are_left_and_a_failed_write_keeps_the_original(shift, root, tmp_path, monkeypatch) -> None:
    path = min((root / "hdf5" / KIND / DEMO).glob("*.hdf5"))
    before = sha256_file(path)

    def boom(_path):
        raise OSError("fsync failed")

    monkeypatch.setattr(shift, "_fsync_path", boom)
    with pytest.raises(OSError, match="fsync"):
        shift.shift_file(path, apply=True)
    assert sha256_file(path) == before and not list(root.rglob("*.tmp"))
    monkeypatch.undo()
    assert shift.shift_file(path, apply=True) == "shifted" and not list(root.rglob("*.tmp"))


def test_unknown_demo_and_one_frame_episodes_fail(shift, root, tmp_path, capsys) -> None:
    assert _run(shift, root, _demos_file(tmp_path, "nope"), "--apply") == 1
    assert "no directory" in capsys.readouterr().out
    _pack_demo(root, "dm_tiny", n_episodes=1, data=[(_frames(1, 0), _actions(1, 0))])
    assert _run(shift, root, _demos_file(tmp_path, "dm_tiny"), "--apply") == 1
    assert "nothing is left" in capsys.readouterr().out
    assert read_episode(next((root / "hdf5" / KIND / "dm_tiny").glob("*.hdf5")))[2]["n_frames"] == 1


def test_demos_file_needs_at_least_one_id(shift, root, tmp_path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing\n")
    assert _run(shift, root, empty, "--apply") == 2
    assert shift.read_demo_ids(_demos_file(tmp_path, "a", "b")) == ["a", "b"]


# --- safety rails ----------------------------------------------------------------------------------------------


def _set_attr(path: Path, key: str, value) -> None:
    with h5py.File(path, "r+") as f:
        f.attrs[key] = value


def test_apply_needs_a_build(shift, root, tmp_path, capsys) -> None:
    before = _digests(root, DEMO)
    assert shift.main(["--root", str(root), "--demos", str(_demos_file(tmp_path, DEMO)), "--apply"]) == 2
    assert "--build" in capsys.readouterr().err and _digests(root, DEMO) == before


def test_files_of_another_build_are_refused(shift, root, tmp_path, capsys) -> None:
    before = _digests(root, DEMO)
    assert _run(shift, root, _demos_file(tmp_path, DEMO), "--apply", "--build", "26445667") == 1
    out = capsys.readouterr().out
    assert "cs2_build is" in out and "not 26445667" in out and "refusing" in out
    assert _digests(root, DEMO) == before


def test_files_packed_after_the_fix_are_refused(shift, verify, root, tmp_path, capsys) -> None:
    path = min((root / "hdf5" / KIND / DEMO).glob("*.hdf5"))
    _set_attr(path, "render_lag_ticks", 0.2)
    before = sha256_file(path)
    assert _run(shift, root, _demos_file(tmp_path, DEMO), "--apply") == 1
    assert "render_lag_ticks" in capsys.readouterr().out
    assert sha256_file(path) == before and "label_shift_frames" not in read_episode(path)[2]
    with pytest.raises(shift.ShiftError, match="render_lag_ticks"):
        shift.shift_file(path, apply=True, build="26598758")


def test_a_failing_file_still_gets_the_manifest_synced_for_the_files_shifted_before_it(
    shift, verify, root, tmp_path, capsys
) -> None:
    low = sorted((root / "hdf5" / KIND / DEMO).glob("*.hdf5"))
    _set_attr(low[1], "render_lag_ticks", 0.0)  # the second low-res file is refused, the first is shifted before it
    assert _run(shift, root, _demos_file(tmp_path, DEMO), "--apply") == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "rewrote 2 record(s)" in out  # the shifted file and the one the test edited
    assert read_episode(low[0])[2]["n_frames"] == N - 1 and read_episode(low[1])[2]["n_frames"] == N
    _verify(verify, root, DEMO)  # the manifest matches every file as it is now


# --- the pooled lag report --------------------------------------------------------------------------------------


def _turning_episode(ticks_ahead: int, start_tick: int) -> tuple[np.ndarray, np.ndarray]:
    """Frames that show the demo ``ticks_ahead`` ticks after their nominal tick, and the labels of the nominal ones."""
    frames, yaw, _ = synthetic_render(ticks_ahead, start_tick=start_tick)
    n = len(frames) - 1
    t = start_tick + TPF * np.arange(n + 1)
    right_deg = -(yaw.reindex(t[1:]).to_numpy() - yaw.reindex(t[:-1]).to_numpy())  # +yaw is a turn to the left
    actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
    actions[np.arange(n), SPEC.mouse_x_slice.start + bin_index(right_deg, SPEC.mouse_x_centers_deg)] = 1
    actions[np.arange(n), SPEC.mouse_y_slice.start + SPEC.mouse_y_units.index(0.0)] = 1
    return frames[:n], actions


def _best_lags(shift: ModuleType, root: Path, demos: Path, capsys) -> dict[tuple[str, str], int]:
    assert shift.main(["--root", str(root), "--demos", str(demos), "--report"]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()[1:]]
    return {(r[0], r[1]): int(r[-1]) for r in rows}


def test_report_shows_lag_plus_one_before_and_zero_after(shift, tmp_path, capsys) -> None:
    root = tmp_path / "synced"
    _pack_demo(root, DEMO, build=26598758, data=[_turning_episode(4, 1000 + 1000 * i) for i in range(3)])
    _pack_demo(root, OTHER, build=26445667, data=[_turning_episode(0, 1000 + 1000 * i) for i in range(3)])
    demos = _demos_file(tmp_path, DEMO)

    before = _best_lags(shift, root, demos, capsys)
    assert before == {("26445667", "other"): 0, ("26598758", "listed"): 1}
    assert _run(shift, root, demos, "--apply") == 0
    capsys.readouterr()
    after = _best_lags(shift, root, demos, capsys)
    assert after == {("26445667", "other"): 0, ("26598758", "listed"): 0}


def test_report_without_episodes_fails(shift, tmp_path, capsys) -> None:
    assert shift.main(["--root", str(tmp_path / "empty"), "--report"]) == 1
    assert "no episodes" in capsys.readouterr().out
