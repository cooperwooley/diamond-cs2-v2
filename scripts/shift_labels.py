#!/usr/bin/env python3
"""Shift the labels of already packed episodes by one frame, to fix the Nov 11 render offset (#88).

The demos rendered on CS2 build 26598758 with ``render.tick_offset = 5`` show frame ``k`` at about ``T_k + 4.5`` ticks,
one frame ahead of the labels. Frame ``k`` therefore shows what the labels call frame ``k + 1``. Dropping the last
frame and the first action fixes it::

    frames[:-1], actions[1:], start_tick += ticks_per_frame, n_frames -= 1, attr label_shift_frames = 1

This is exactly what re-extracting the labels from ``T0 + 4`` gives: every label depends only on the ticks of its own
window ``(T_k, T_{k+1}]``, so the labels of the kept frames are the same numbers, and frame ``k`` of the shifted file is
frame ``k`` of the old one. Nothing is re-rendered and no pixel is touched: the full-res JPEG bytes are copied as stored.

For each demo id in ``--demos`` (one per line, ``#`` comments allowed) the low-res episodes under ``<root>/hdf5`` and
the full-res ones under ``<root>/hdf5_fullres`` are rewritten, then the demo's ``manifest.jsonl`` gets the new
``frames`` and ``sha256`` of every file, so ``scripts/verify_manifests.py`` passes. Files that already carry
``label_shift_frames`` are skipped, so a re-run (or a run after a crash) does the rest; the manifest is always brought
in line with the files. Every file is written to a temp file, fsynced and renamed, like ``pack``.

The default is a dry run that prints what would change; ``--apply`` writes. ``--report`` prints the pooled frame-level
lag (``corr(-dx[t], mouse_x_deg[t + lag])`` of the low-res frames) per CS2 build and for listed vs other demos, to
show before and after: the listed demos should peak at +1 before and 0 after, the others at 0.

Safety: ``--apply`` needs ``--build`` (``26598758``). A file whose ``cs2_build`` attr differs, or that already has
``render_lag_ticks`` (packed after the ``tick_offset`` fix, so it is aligned), is refused and reported as a problem, never
shifted. If a file of a demo fails, the files before it stay shifted and the demo's manifest is still brought in line.

On the Mill, take a hardlink backup first (``cp -al``); that is safe because every write here is a temp file renamed over
the original, so the backup keeps the old inode and content.

After the shift, the PC's manifests for these demos are stale. Every supervisor sync rsyncs the PC's whole ``hdf5/``
tree, manifests included, and verifies every demo that has a local manifest, so a stale one would overwrite the rewritten
Mill manifest. Before the next sync move the PC copies out of the tree, to
``<data>/backup/manifests_pre88/<kind>/<demo_id>/manifest.jsonl``; the demos stay ``synced`` in the pipeline state.

Not touched: ``qa/<demo>.jsonl`` (it records the planned episode frame counts, not the packed ones).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from cs2wm.actions import ActionSpec
from cs2wm.config import load_config
from cs2wm.eval.render_offset import MAX_SHIFT_FRAC
from cs2wm.eval.rotation import estimate_shift_phasecorr
from cs2wm.pipeline.pack import _fsync_path, sha256_file, write_manifest

SHIFT_ATTR = "label_shift_frames"
LAG_ATTR = "render_lag_ticks"  # written by pack since #88, i.e. packed after the tick_offset fix
REPORT_LAGS = (-2, -1, 0, 1, 2)
MAX_MOUSE_DEG = 15.0  # report: pairs with a larger turn per frame are cuts or glitches (the #73 rule)
_TRANSIENT_SUFFIXES = (".staging", ".old", ".deleting")


class ShiftError(RuntimeError):
    """An episode file cannot be shifted."""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="synced data root (holds hdf5/ and hdf5_fullres/)")
    p.add_argument("--demos", help="file with one demo id per line: the demos to shift")
    p.add_argument("--apply", action="store_true", help="write the changes (default: dry run); needs --build")
    p.add_argument(
        "--build", help="CS2 build the demos were rendered on (26598758); files of any other build are refused"
    )
    p.add_argument("--report", action="store_true", help="print the pooled frame-level lag instead of shifting")
    return p


def read_demo_ids(path: Path) -> list[str]:
    """Demo ids of a ``--demos`` file: one per line, blank lines and ``#`` comments ignored."""
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def _demo_dirs(hdf5: Path, fullres: Path, demo_id: str) -> list[tuple[Path, Path]]:
    """``(low-res dir, full-res dir)`` of a demo, one per kind that has it."""
    found = [d for d in sorted(hdf5.glob(f"*/{demo_id}")) if d.is_dir() and not d.name.endswith(_TRANSIENT_SUFFIXES)]
    return [(d, fullres / d.parent.name / demo_id) for d in found]


def _h5_attrs(f: h5py.File) -> dict[str, Any]:
    return {k: f.attrs[k] for k in f.attrs}


def shift_file(path: Path, apply: bool, build: str | None = None) -> str:
    """Shift one episode file by one frame (see the module docstring).

    Args:
        path: Low-res or full-res episode HDF5.
        apply: Write the file; False only reports what would happen.
        build: Required ``cs2_build`` attr; a file of another build is refused. None skips the check (dry runs).

    Returns:
        ``"skipped"`` if the file already has ``label_shift_frames``, else ``"shifted"`` (or ``"would shift"``).

    Raises:
        ShiftError: If the file has fewer than 2 frames or lacks a frame or an attr, is of another CS2 build than
            ``build``, or already has ``render_lag_ticks`` (it was packed after the fix and must never be shifted).
    """
    path = Path(path)
    with h5py.File(path, "r") as src:
        attrs = _h5_attrs(src)
        if SHIFT_ATTR in attrs:
            return "skipped"
        if LAG_ATTR in attrs:
            raise ShiftError(f"{path}: has {LAG_ATTR} (packed after the tick_offset fix), refusing to shift it")
        if build is not None and str(attrs.get("cs2_build")) != str(build):
            raise ShiftError(f"{path}: cs2_build is {attrs.get('cs2_build')!r}, not {build}, refusing to shift it")
        try:
            n, start = int(attrs["n_frames"]), int(attrs["start_tick"])
            ticks = int(attrs["tickrate"]) // int(attrs["fps"])
        except KeyError as e:
            raise ShiftError(f"{path}: missing attr {e}") from e
        if n < 2:
            raise ShiftError(f"{path}: {n} frame(s), nothing is left after the shift")
        if not apply:
            return "would shift"
        tmp = path.with_name(path.name + ".tmp")
        try:
            with h5py.File(tmp, "w") as dst:
                for i in range(n - 1):
                    for name_src, name_dst in ((f"frame_{i}_x", f"frame_{i}_x"), (f"frame_{i + 1}_y", f"frame_{i}_y")):
                        if name_src not in src:
                            raise ShiftError(f"{path}: missing dataset {name_src!r}")
                        dst.create_dataset(name_dst, data=src[name_src][()])  # JPEG bytes stay as stored
                for key, value in attrs.items():
                    dst.attrs[key] = value
                dst.attrs["n_frames"] = n - 1
                dst.attrs["start_tick"] = start + ticks
                dst.attrs[SHIFT_ATTR] = 1
            _fsync_path(tmp)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)
    return "shifted"


def _manifest_target(low_dir: Path, full_dir: Path, name: str) -> Path:
    return full_dir / name.removeprefix("fullres/") if name.startswith("fullres/") else low_dir / name


def sync_manifest(low_dir: Path, full_dir: Path, apply: bool, pending: frozenset[Path] = frozenset()) -> int:
    """Bring a demo's ``manifest.jsonl`` in line with its files (``frames`` from ``n_frames``, and ``sha256``).

    Args:
        low_dir: ``hdf5/<kind>/<demo_id>``.
        full_dir: ``hdf5_fullres/<kind>/<demo_id>``.
        apply: Write the manifest; False only counts.
        pending: Files a dry run reports as "would shift"; their records count as changed.

    Returns:
        The number of records that differ from their file.

    Raises:
        ShiftError: If a listed file is missing.
    """
    manifest = low_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise ShiftError(f"{manifest} is missing")
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    for rec in records:
        path = _manifest_target(low_dir, full_dir, str(rec["file"]))
        if not path.is_file():
            raise ShiftError(f"{path} is listed in {manifest} but missing")
        with h5py.File(path, "r") as f:
            frames = int(f.attrs["n_frames"])
        sha = sha256_file(path)
        if path in pending:
            changed += 1
        elif rec.get("frames") != frames or rec.get("sha256") != sha:
            rec["frames"], rec["sha256"] = frames, sha
            changed += 1
    if changed and apply:
        write_manifest(manifest, records)
    return changed


def shift_demos(root: Path, demo_ids: Sequence[str], apply: bool, build: str | None = None) -> int:
    """Shift every episode of the listed demos; prints one line per file and a summary.

    Returns:
        0 on success, 1 if a demo was not found or a file could not be shifted.
    """
    cfg = load_config(overrides=[f"paths.root={root}"])
    hdf5, fullres = Path(cfg.paths.hdf5), Path(cfg.paths.hdf5_fullres)
    counts: dict[str, int] = defaultdict(int)
    failed = 0
    for demo_id in demo_ids:
        dirs = _demo_dirs(hdf5, fullres, demo_id)
        if not dirs:
            print(f"FAIL {demo_id}: no directory under {hdf5}/*/")
            failed += 1
            continue
        for low_dir, full_dir in dirs:
            files = sorted(low_dir.glob("*.hdf5")) + (sorted(full_dir.glob("*.hdf5")) if full_dir.is_dir() else [])
            pending: set[Path] = set()
            problem = False
            try:
                for path in files:
                    status = shift_file(path, apply, build)
                    counts[status] += 1
                    print(f"{status:<11} {path.relative_to(hdf5.parent)}")
                    if status == "would shift":
                        pending.add(path)
            except (ShiftError, OSError, ValueError, KeyError) as e:
                print(f"FAIL {demo_id}: {e}")
                failed, problem = failed + 1, True
            try:  # also after a failure: the manifest must match the files that were shifted before it
                stale = sync_manifest(low_dir, full_dir, apply, frozenset(pending))
            except (ShiftError, OSError, ValueError, KeyError) as e:
                if not problem:
                    print(f"FAIL {demo_id}: {e}")
                    failed += 1
                continue
            verb = "rewrote" if apply else "would rewrite"
            print(f"manifest    {demo_id}: {verb} {stale} record(s)")
    summary = ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "no files"
    print(f"{'APPLIED' if apply else 'DRY RUN (pass --apply to write)'}: {summary}; {failed} problem(s)")
    return 1 if failed else 0


def _episode_pairs(path: Path, spec: ActionSpec) -> tuple[np.ndarray, np.ndarray, str]:
    """``(-dx, mouse_x_deg)`` of a low-res file and its CS2 build."""
    with h5py.File(path, "r") as f:
        n = int(f.attrs["n_frames"])
        frames = [f[f"frame_{i}_x"][()] for i in range(n)]
        actions = np.stack([f[f"frame_{i}_y"][()] for i in range(n)])
        build = str(f.attrs.get("cs2_build", ""))
    mouse_x = spec.mouse_x_centers_deg[np.argmax(actions[:, spec.mouse_x_slice], axis=1)]
    dx = np.array([estimate_shift_phasecorr(a, b).dx for a, b in itertools.pairwise(frames)])
    return -dx, mouse_x, build


def report(root: Path, listed: set[str]) -> int:
    """Print the pooled frame-level lag per ``(cs2_build, listed or not)`` group.

    ``r(lag)`` is the correlation of ``-dx[t]`` (turn right = content moves left) with ``mouse_x_deg[t + lag]`` over
    every low-res frame pair of the group; the best lag is the highest r. Aligned data peaks at 0. Pairs with a turn of
    ``MAX_MOUSE_DEG`` or more, or a shift of over ``MAX_SHIFT_FRAC`` of the frame width (a spectator cut makes phase
    correlation return a wrapped shift in the hundreds of pixels), are left out; they would swamp the correlation.
    """
    cfg = load_config(overrides=[f"paths.root={root}"])
    spec = ActionSpec.from_config(cfg)
    max_shift = MAX_SHIFT_FRAC * int(cfg.resolution.low_res[1])
    pools: dict[tuple[str, bool], dict[int, list[tuple[np.ndarray, np.ndarray]]]] = defaultdict(
        lambda: {lag: [] for lag in REPORT_LAGS}
    )
    episodes: dict[tuple[str, bool], int] = defaultdict(int)
    files = [
        p for p in sorted(Path(cfg.paths.hdf5).glob("*/*/*.hdf5")) if not p.parent.name.endswith(_TRANSIENT_SUFFIXES)
    ]
    for path in files:
        neg_dx, mouse_x, build = _episode_pairs(path, spec)
        key = (build or "unknown", path.parent.name in listed)
        episodes[key] += 1
        t = np.arange(len(neg_dx))
        for lag in REPORT_LAGS:
            ok = (t + lag >= 0) & (t + lag < len(mouse_x))
            ok[ok] = (np.abs(mouse_x[t[ok] + lag]) < MAX_MOUSE_DEG) & (np.abs(neg_dx[ok]) <= max_shift)
            pools[key][lag].append((neg_dx[ok], mouse_x[t[ok] + lag]))
    if not files:
        print(f"FAILED: no episodes under {cfg.paths.hdf5}")
        return 1
    print(
        f"{'cs2_build':<12}{'demos':<8}{'episodes':>9}  "
        + " ".join(f"{f'r({lag:+d})':>7}" for lag in REPORT_LAGS)
        + "  best"
    )
    for key in sorted(pools):
        r = {}
        for lag in REPORT_LAGS:
            x = np.concatenate([p[0] for p in pools[key][lag]])
            y = np.concatenate([p[1] for p in pools[key][lag]])
            r[lag] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")
        best = max((lag for lag in REPORT_LAGS if np.isfinite(r[lag])), key=r.get, default=None)
        cells = " ".join(f"{r[lag]:>+7.2f}" for lag in REPORT_LAGS)
        label = "listed" if key[1] else "other"
        print(f"{key[0]:<12}{label:<8}{episodes[key]:>9}  {cells}  {'n/a' if best is None else f'{best:+d}'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the script.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: 0 on success, 1 on a problem, 2 on a usage error.
    """
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    demo_ids = read_demo_ids(Path(args.demos)) if args.demos else []
    if args.report:
        return report(root, set(demo_ids))
    if not demo_ids:
        print("error: --demos FILE with at least one demo id is required to shift", file=sys.stderr)
        return 2
    if args.apply and not args.build:
        print("error: --apply needs --build (the CS2 build of the demos, e.g. 26598758)", file=sys.stderr)
        return 2
    return shift_demos(root, demo_ids, args.apply, args.build)


if __name__ == "__main__":
    raise SystemExit(main())
