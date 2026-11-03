"""The no-CS2 fixture demo (#38): ``scripts/fixture_demo.py`` runs the real extract and pack stages on bundled tables."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cs2wm.actions import ActionSpec, decode
from cs2wm.config import load_config
from cs2wm.pipeline.pack import read_episode, sha256_file

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fixture_demo.py"
DEMO = "fixture-0001"


def run_script(root: Path) -> subprocess.CompletedProcess[str]:
    """Run the script as a user would, with the checkout importable even when ``cs2wm`` is not installed."""
    env = {**os.environ, "PYTHONPATH": str(SCRIPT.parent.parent)}
    cmd = [sys.executable, str(SCRIPT), "--root", str(root)]
    return subprocess.run(cmd, check=False, capture_output=True, text=True, env=env, timeout=120)


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, subprocess.CompletedProcess[str]]:
    root = tmp_path_factory.mktemp("fixture-demo") / "dcs2"
    return root, run_script(root)


def test_script_is_executable_and_succeeds(demo_run) -> None:
    _, proc = demo_run
    assert os.access(SCRIPT, os.X_OK) and SCRIPT.read_text().startswith("#!/usr/bin/env python3")
    assert proc.returncode == 0, proc.stderr
    assert "OK: 4 manifest record(s) verified" in proc.stdout
    assert "2 episode(s) packed" in proc.stdout and "keys=w" in proc.stdout


def test_hdf5_files_and_manifest_verify(demo_run) -> None:
    root, _ = demo_run
    low, full = root / "hdf5" / "dm" / DEMO, root / "hdf5_fullres" / "dm" / DEMO
    names = [f"dm-{DEMO}-s000-c00.hdf5", f"dm-{DEMO}-s000-c01.hdf5"]
    assert sorted(p.name for p in low.glob("*.hdf5")) == names
    assert sorted(p.name for p in full.glob("*.hdf5")) == names
    records = [json.loads(line) for line in (low / "manifest.jsonl").read_text().splitlines()]
    assert len(records) == 4 and all(r["frames"] == 12 for r in records)
    for rec in records:
        path = full / rec["file"].removeprefix("fullres/") if rec["file"].startswith("fullres/") else low / rec["file"]
        assert sha256_file(path) == rec["sha256"], rec["file"]
    assert not (root / "frames" / DEMO).exists() and not (root / "labels" / DEMO).exists()  # pack cleaned up


def test_actions_decode_to_what_the_fixture_does(demo_run) -> None:
    root, _ = demo_run
    spec = ActionSpec.from_config(load_config())
    low = root / "hdf5" / "dm" / DEMO
    frames, actions, attrs = read_episode(low / f"dm-{DEMO}-s000-c00.hdf5")
    assert frames.shape == (12, 30, 56, 3) and actions.shape == (12, 51)
    assert (attrs["player"], attrs["demo_id"], attrs["kind"], attrs["start_tick"]) == ("Bot A", DEMO, "dm", 10)
    first = decode(actions[0], spec)
    assert first.keys == ("w",) and first.lmb  # walking forward, firing at tick 13 (bot B's shot at 20 must not leak)
    assert decode(actions[7], spec).keys == ("w", "r")  # reload at tick 40
    assert abs(first.mouse_x_deg - 0.55) < 0.1  # yaw turns -0.55 deg per frame = mouse right

    _, actions2, attrs2 = read_episode(low / f"dm-{DEMO}-s000-c01.hdf5")
    assert attrs2["start_tick"] == 58
    assert {"w", "d"} <= set(decode(actions2[0], spec).keys)  # strafing right from tick 60
    assert "space" in decode(actions2[2], spec).keys and decode(actions2[7], spec).lmb
    assert np.isfinite(actions2).all()


def test_refuses_an_existing_root(tmp_path: Path) -> None:
    proc = run_script(tmp_path)
    assert proc.returncode == 2 and "already exists" in proc.stderr
