#!/usr/bin/env python3
"""Re-check every sha256 in each ``manifest.jsonl`` under an HDF5 root (#17); used by #19, #24, #25.

Each record's ``file`` is relative to its manifest's directory. Prints one line per manifest and a final
``OK``/``FAILED`` summary; exits 1 on any mismatch, missing file, unreadable record or manifest, empty manifest
(nothing was packed), or if no manifest is found.
``fullres/<name>`` records (the full-res twin of a low-res file) are checked against ``--fullres-root`` and counted
as skipped without it. ``.hdf5`` files that no record lists are reported as a warning only.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cs2wm.pipeline.pack import sha256_file


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="HDF5 root ($S/hdf5)")
    p.add_argument("--fullres-root", help="full-res HDF5 root ($S/hdf5_fullres), to check fullres/<name> records")
    return p


# pack stages, swaps and deletes whole demo dirs under these suffixes; files inside them are never part of the dataset
_TRANSIENT_DIR_SUFFIXES = (".staging", ".old", ".deleting")


def _in_transient_dir(path: Path, root: Path) -> bool:
    return any(part.endswith(_TRANSIENT_DIR_SUFFIXES) for part in path.relative_to(root).parts[:-1])


def check_manifest(manifest: Path, root: Path, fullres_root: Path | None) -> tuple[int, int, list[str], list[str]]:
    """Check one manifest.

    Args:
        manifest: A ``manifest.jsonl``.
        root: The HDF5 root it lives under.
        fullres_root: Full-res root, or None to skip ``fullres/`` records.

    Returns:
        ``(checked, skipped, problems, unlisted)``: files verified OK, full-res records skipped, one message per
        failure, and ``.hdf5`` files next to the manifest that no record lists.
    """
    checked, skipped = 0, 0
    problems: list[str] = []
    listed: set[Path] = set()
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError) as e:
        return 0, 0, [f"{manifest}: unreadable manifest ({e})"], []
    if not any(line.strip() for line in lines):
        return 0, 0, [f"{manifest}: empty manifest (no records)"], []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            name, expected = str(record["file"]), str(record["sha256"])
        except (ValueError, KeyError, TypeError) as e:
            problems.append(f"{manifest}:{lineno}: unreadable record ({e!r})")
            continue
        base = manifest.parent
        if name.startswith("fullres/"):
            if fullres_root is None:
                skipped += 1
                continue
            base, name = fullres_root / manifest.parent.relative_to(root), name.removeprefix("fullres/")
        path = base / name
        if base.resolve() not in path.resolve().parents:
            problems.append(f"{manifest}:{lineno}: {record['file']!r} points outside the manifest directory")
            continue
        listed.add(path)
        if not path.is_file():
            problems.append(f"MISSING {path}")
        elif (actual := sha256_file(path)) != expected:
            problems.append(f"MISMATCH {path}: manifest {expected[:12]}..., file {actual[:12]}...")
        else:
            checked += 1
    unlisted = [str(p) for p in sorted(manifest.parent.glob("*.hdf5")) if p not in listed]
    return checked, skipped, problems, unlisted


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the manifests.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        0 if every checksum matches, else 1.
    """
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    fullres_root = Path(args.fullres_root) if args.fullres_root else None
    manifests = [m for m in sorted(root.rglob("manifest.jsonl")) if not _in_transient_dir(m, root)]
    if not manifests:
        print(f"FAILED: no manifest.jsonl under {root}")
        return 1
    total_checked, total_skipped, total_problems = 0, 0, 0
    for manifest in manifests:
        checked, skipped, problems, unlisted = check_manifest(manifest, root, fullres_root)
        status = "FAIL" if problems else "ok"
        print(f"{status:<4} {manifest.parent.relative_to(root)}: {checked} files verified")
        for msg in problems:
            print(f"     {msg}")
        for path in unlisted:
            print(f"     warning: not in manifest: {path}")
        total_checked, total_skipped, total_problems = (
            total_checked + checked,
            total_skipped + skipped,
            total_problems + len(problems),
        )
    note = f" ({total_skipped} full-res records skipped, no --fullres-root)" if total_skipped else ""
    if total_problems:
        print(f"FAILED: {total_problems} problems; {total_checked} files verified in {len(manifests)} manifests{note}")
        return 1
    print(f"OK: {total_checked} files verified in {len(manifests)} manifests{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
