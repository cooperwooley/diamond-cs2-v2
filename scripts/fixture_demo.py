#!/usr/bin/env python3
"""No-CS2 demo of the data format: run the real ``extract`` and ``pack`` stages on a tiny synthetic demo.

``tests/fixtures/demo_fixture/`` holds what ``segment`` and demoparser2 would have produced for a 120-tick demo:
``render_plan.json`` (the output of the real ``segment`` stage), ``ticks.csv`` and ``events/*.csv`` (the tables
demoparser2 returns), and ``demo.json`` (the sidecar ``generate`` writes next to a demo). No ``.dem`` file is
bundled, so a small fake parser that reads the CSVs is injected through the ``cs2wm.demo._make_parser`` seam. The
rendered frames are drawn here, deterministically (stand-ins for what HLAE would capture). Everything between
those inputs and the HDF5 files is the real stage code: ``extract_actions.run`` labels the episodes and
``pack.run`` writes, verifies and manifests the HDF5 files.

Usage::

    python scripts/fixture_demo.py [--root DIR]

Without ``--root`` a temp dir is used and removed afterwards. The summary shows episodes, frames, decoded actions,
file attrs and the manifest. Exits 1 if the packed files do not verify.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from cs2wm import demo
from cs2wm.actions import ActionSpec, decode
from cs2wm.config import load_config
from cs2wm.pipeline import extract_actions, pack
from cs2wm.pipeline.segment import read_render_plan

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "demo_fixture"
DEMO_ID = "fixture-0001"
KIND = "dm"


class CsvParser:
    """Stands in for ``demoparser2.DemoParser``: serves the fixture's CSV tables."""

    def __init__(self, fixture: Path):
        self._ticks = pd.read_csv(fixture / "ticks.csv")
        self._events = {p.stem: pd.read_csv(p) for p in sorted((fixture / "events").glob("*.csv"))}
        self._header = json.loads((fixture / "demo.json").read_text(encoding="utf-8"))["header"]

    def parse_header(self) -> dict[str, str]:
        return self._header

    def parse_ticks(self, wanted_props: Sequence[str], **_: object) -> pd.DataFrame:
        keep = {"tick", "name", "steamid", *wanted_props}
        return self._ticks[[c for c in self._ticks.columns if c in keep]].copy()  # props the table lacks are omitted

    def list_game_events(self) -> list[str]:
        return list(self._events)

    def parse_events(self, names: Sequence[str], **_: object) -> list[tuple[str, pd.DataFrame]]:
        return [(n, self._events[n]) for n in names]


def draw_frame(index: int, size: tuple[int, int]) -> np.ndarray:
    """A BGR test frame: a gradient, a bar that moves with the frame index, and the index as text."""
    h, w = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[..., 0] = np.linspace(40, 200, w, dtype=np.uint8)[None, :]
    frame[..., 1] = np.linspace(60, 160, h, dtype=np.uint8)[:, None]
    x = 10 + 9 * index
    cv2.rectangle(frame, (x, 100), (x + 12, 140), (255, 255, 255), -1)
    cv2.putText(frame, f"frame {index}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def stage_inputs(cfg: DictConfig) -> None:
    """Put the fixture where the stages look for it: plan, sidecar, and rendered frames with their ``render.json``."""
    plan_path = Path(cfg.paths.plans) / DEMO_ID / "render_plan.json"
    plan_path.parent.mkdir(parents=True)
    shutil.copy(FIXTURE_DIR / "render_plan.json", plan_path)
    sidecar = Path(cfg.paths.demos) / KIND / f"{DEMO_ID}.json"
    sidecar.parent.mkdir(parents=True)
    shutil.copy(FIXTURE_DIR / "demo.json", sidecar)

    plan = read_render_plan(plan_path)
    size = (int(cfg.resolution.render[0]), int(cfg.resolution.render[1]))
    frames_dir = Path(cfg.paths.frames) / DEMO_ID
    segments = []
    for seg in plan["segments"]:
        seg_dir = frames_dir / f"seg{seg['segment_idx']:03d}"
        seg_dir.mkdir(parents=True)
        for i in range(seg["rendered_frames"]):
            cv2.imwrite(str(seg_dir / f"{i:06d}.png"), draw_frame(i, size))
        segments.append(
            {
                "segment_idx": seg["segment_idx"],
                "dir": seg_dir.name,
                "frames": seg["rendered_frames"],
                "episodes": [
                    {"episode_id": e["episode_id"], "first_frame": e["first_frame"], "n_frames": e["n_frames"]}
                    for e in seg["episodes"]
                ],
            }
        )
    render_json = {
        "demo_id": DEMO_ID,
        "plan_sha256": pack.sha256_file(plan_path),
        "size": list(size),
        "segments": segments,
    }
    (frames_dir / "render.json").write_text(json.dumps(render_json, indent=2) + "\n", encoding="utf-8")


def run_stages(root: Path) -> DictConfig:
    """Run extract and pack on the fixture under ``root`` and return the config used."""
    cfg = load_config(
        overrides=[f"paths.root={root}", "episodes.min_frames=4", "episodes.max_frames=12", "pack.fullres_mod=1"]
    )
    stage_inputs(cfg)
    real_make_parser = demo._make_parser
    demo._make_parser = lambda path: CsvParser(FIXTURE_DIR)
    try:
        qa = extract_actions.run(Path(cfg.paths.demos) / KIND / f"{DEMO_ID}.dem", cfg)
    finally:
        demo._make_parser = real_make_parser
    failed = [q for q in qa if not q.passed]
    if failed:
        raise RuntimeError(f"fixture episodes failed QA: {[(q.episode_id, q.reasons) for q in failed]}")
    pack.run(DEMO_ID, cfg)
    return cfg


def summarize(cfg: DictConfig) -> bool:
    """Read the packed files back, print a summary and return whether every manifest checksum matches."""
    spec = ActionSpec.from_config(cfg)
    hdf5_dir = Path(cfg.paths.hdf5) / KIND / DEMO_ID
    fullres_dir = Path(cfg.paths.hdf5_fullres) / KIND / DEMO_ID
    records = [json.loads(line) for line in (hdf5_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    files = sorted(hdf5_dir.glob("*.hdf5"))
    print(
        f"demo {DEMO_ID}: {len(files)} episode(s) packed from {len(read_render_plan(FIXTURE_DIR / 'render_plan.json')['segments'])} segment(s)"
    )

    for path in files:
        frames, actions, attrs = pack.read_episode(path)
        print(f"\n{path.name}: frames {frames.shape} uint8 BGR, actions {actions.shape} float32")
        print("  attrs: " + ", ".join(f"{k}={attrs[k]}" for k in ("player", "start_tick", "cs2_build", "encoding")))
        decoded = [decode(a, spec) for a in actions]
        notable = [k for k, a in enumerate(decoded) if a.lmb or a.rmb or set(a.keys) - set("wasd")]
        for k in sorted({0, *notable[:3]}):
            a = decoded[k]
            keys = "+".join(a.keys) or "-"
            clicks = "+".join(n for n, on in (("lmb", a.lmb), ("rmb", a.rmb)) if on) or "-"
            print(
                f"  frame {k:2d}: keys={keys:<10} click={clicks:<4} mouse=({a.mouse_x_deg:+.2f}, {a.mouse_y_deg:+.2f}) deg"
            )

    print("\nmanifest.jsonl:")
    ok = True
    for rec in records:
        name = rec["file"]
        path = fullres_dir / name.removeprefix("fullres/") if name.startswith("fullres/") else hdf5_dir / name
        match = path.is_file() and pack.sha256_file(path) == rec["sha256"]
        ok &= match
        print(f"  {name:<40} {rec['frames']:>3} frames  sha256 {rec['sha256'][:12]}  {'ok' if match else 'MISMATCH'}")
    print(f"\n{'OK' if ok else 'FAILED'}: {len(records)} manifest record(s) verified")
    return ok


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, help="keep the data root here (must not exist yet); default: a temp dir")
    args = p.parse_args(argv)
    if args.root is not None:
        if args.root.exists():
            p.error(f"{args.root} already exists")
        return 0 if summarize(run_stages(args.root)) else 1
    with tempfile.TemporaryDirectory(prefix="cs2wm-fixture-") as tmp:
        return 0 if summarize(run_stages(Path(tmp) / "dcs2")) else 1


if __name__ == "__main__":
    sys.exit(main())
