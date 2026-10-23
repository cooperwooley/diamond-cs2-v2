#!/usr/bin/env python3
"""Convert synced low-res HDF5 episodes into DIAMOND ``.pt`` train/test datasets (#17, runs on the Mill).

Output: ``<out>/low_res/{train,test}`` (DIAMOND ``Dataset`` episodes plus ``info.pt``), ``<out>/split.json``
(demo -> split) and ``<out>/dataset_meta.json`` (seed, flags, episode table and source manifest checksums).

The split is by demo, so no demo is in both splits, and it is deterministic for a seed. Each kind is split on its own
(``ceil(test_frac * n)`` demos to test, always keeping one demo in train). Smoke demos are the H2 evaluation set: at
least 30% of them, and at least 3, go to test. With fewer than 3 smoke demos all of them go to test (and a warning
says train has none), because an H2 evaluation without smoke test data is worse than a train set without smoke.

``--shuffle-actions`` gives each train episode the action sequence of a different train episode (a seeded
derangement, tiled or cropped to the episode's length); test keeps the true actions.

Frames are saved as uint8 directly (not through ``Episode.save``, whose float round trip truncates 63 of the 256
values), so the stored pixels equal the HDF5 pixels exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from cs2wm.pipeline.pack import read_episode, sha256_file

REPO_ROOT = Path(__file__).resolve().parent.parent
# src/ is v1/DIAMOND code with flat imports (``from data import ...``).
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import Dataset, Episode

SMOKE_KIND = "smoke"
SMOKE_MIN_FRAC = Fraction(3, 10)
SMOKE_MIN_DEMOS = 3
SPLITS = ("train", "test")


@dataclass(frozen=True)
class EpisodeRef:
    """One packed episode, identified from its HDF5 attrs.

    Attributes:
        path: Low-res HDF5 file.
        episode_id: Episode id (the ``episode_id`` attr).
        demo_id: Demo the episode was cut from.
        kind: ``dm``, ``human`` or ``smoke``.
        n_frames: Number of frames.
    """

    path: Path
    episode_id: str
    demo_id: str
    kind: str
    n_frames: int


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", required=True, help="root of the synced low-res HDF5 tree ($S/hdf5)")
    p.add_argument("--out", required=True, help="output dataset directory ($S/datasets/<name>)")
    p.add_argument("--kinds", default="dm,human,smoke", help="comma-separated episode kinds to include")
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-actions", action="store_true", help="build the shuffled-action control (train only)")
    return p


def _py(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def discover(hdf5_root: Path, kinds: Sequence[str]) -> list[EpisodeRef]:
    """List the episodes of the requested kinds under an HDF5 root, sorted by episode id.

    The kind, demo and episode id come from the file attrs, not from the path.

    Args:
        hdf5_root: ``$S/hdf5``.
        kinds: Kinds to keep.

    Returns:
        The episodes.

    Raises:
        ValueError: If a file lacks an attr, an episode id repeats, or a demo appears under two kinds.
    """
    episodes: dict[str, EpisodeRef] = {}
    demo_kind: dict[str, str] = {}
    for path in sorted(Path(hdf5_root).rglob("*.hdf5")):
        with h5py.File(path, "r") as f:
            missing = [k for k in ("episode_id", "demo_id", "kind", "n_frames") if k not in f.attrs]
            if missing:
                raise ValueError(f"{path}: missing attrs {missing}")
            ref = EpisodeRef(
                path,
                str(_py(f.attrs["episode_id"])),
                str(_py(f.attrs["demo_id"])),
                str(_py(f.attrs["kind"])),
                int(_py(f.attrs["n_frames"])),
            )
        if ref.kind not in kinds:
            continue
        if ref.episode_id in episodes:
            raise ValueError(f"duplicate episode_id {ref.episode_id!r}: {episodes[ref.episode_id].path} and {path}")
        if demo_kind.setdefault(ref.demo_id, ref.kind) != ref.kind:
            raise ValueError(f"demo {ref.demo_id!r} appears under kinds {demo_kind[ref.demo_id]!r} and {ref.kind!r}")
        episodes[ref.episode_id] = ref
    return sorted(episodes.values(), key=lambda e: e.episode_id)


def _ceil_frac(frac: float, n: int) -> int:
    """``ceil(frac * n)`` on the decimal value of ``frac``, so ties (0.3 * 10) are not decided by float rounding."""
    return math.ceil(Fraction(str(frac)) * n)


def n_test_demos(kind: str, n: int, test_frac: float) -> int:
    """Number of a kind's demos that go to test.

    Args:
        kind: Episode kind.
        n: Number of demos of that kind.
        test_frac: Requested test fraction.

    Returns:
        ``ceil(test_frac * n)`` capped at ``n - 1`` (so train keeps one demo); for smoke, at least
        ``max(ceil(0.3 n), 3)`` capped at ``n`` (all of them when there are fewer than 3).
    """
    if kind == SMOKE_KIND:
        return min(n, max(_ceil_frac(test_frac, n), math.ceil(SMOKE_MIN_FRAC * n), SMOKE_MIN_DEMOS))
    return min(_ceil_frac(test_frac, n), n - 1)


def split_demos(demo_kinds: dict[str, str], test_frac: float, seed: int) -> dict[str, str]:
    """Assign every demo to train or test, per kind, deterministically.

    Args:
        demo_kinds: Demo id -> kind.
        test_frac: Requested test fraction per kind.
        seed: Split seed.

    Returns:
        Demo id -> ``"train"`` or ``"test"``.
    """
    split: dict[str, str] = {}
    for kind in sorted(set(demo_kinds.values())):
        demos = sorted(d for d, k in demo_kinds.items() if k == kind)
        # A string seed is hashed with sha512 by random.Random, so it is stable across runs and platforms.
        random.Random(f"{seed}:split:{kind}").shuffle(demos)
        n_test = n_test_demos(kind, len(demos), test_frac)
        if kind == SMOKE_KIND and len(demos) < SMOKE_MIN_DEMOS:
            print(f"WARNING: only {len(demos)} smoke demos (< {SMOKE_MIN_DEMOS}): all go to test, train has no smoke")
        for i, demo in enumerate(demos):
            split[demo] = "test" if i < n_test else "train"
    return split


def derangement(n: int, seed: int) -> list[int]:
    """A seeded permutation of ``range(n)`` without fixed points (Sattolo's algorithm: one n-cycle).

    Args:
        n: Size, at least 2.
        seed: Seed.

    Returns:
        ``p`` with ``p[i] != i`` for all ``i``.

    Raises:
        ValueError: If ``n < 2``, where no derangement exists.
    """
    if n < 2:
        raise ValueError(f"a derangement needs at least 2 items, got {n}")
    rng = random.Random(f"{seed}:derangement")
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randrange(i)
        p[i], p[j] = p[j], p[i]
    return p


def tile_to_length(actions: np.ndarray, n: int) -> np.ndarray:
    """Repeat (tile) or cut (crop) an action sequence to ``n`` rows.

    Args:
        actions: ``[m, 51]`` actions, ``m >= 1``.
        n: Target length.

    Returns:
        ``[n, 51]`` actions.
    """
    return actions[np.arange(n) % len(actions)]


def _read_actions(path: Path) -> np.ndarray:
    """Read only the labels of an episode (``frame_{i}_y``), for the shuffle sources."""
    with h5py.File(path, "r") as f:
        n = int(_py(f.attrs["n_frames"]))
        return np.stack([f[f"frame_{i}_y"][:] for i in range(n)]).astype(np.float32, copy=False)


def _save_episode(dataset: Dataset, ref: EpisodeRef, frames_bgr: np.ndarray, actions: np.ndarray) -> int:
    """Register an episode in ``dataset`` and write its ``.pt`` file with exact uint8 pixels."""
    n = len(frames_bgr)
    obs = torch.from_numpy(np.ascontiguousarray(frames_bgr[..., ::-1])).permute(0, 3, 1, 2).contiguous()  # RGB CHW
    act = torch.from_numpy(np.ascontiguousarray(actions, dtype=np.float32))
    rew = torch.zeros(n)
    end = torch.zeros(n, dtype=torch.uint8)
    trunc = torch.zeros(n, dtype=torch.uint8)
    # Same key as the full-res loader (parent directory name / file name); episode_id is for provenance.
    info = {
        "original_file_id": f"{ref.demo_id}/{ref.episode_id}.hdf5",
        "episode_id": ref.episode_id,
        "demo_id": ref.demo_id,
        "kind": ref.kind,
    }
    # The dataset is built with save_on_disk=False: add_episode only updates the counters here (it ignores the pixel
    # values), and the lossless file is written below in the layout Episode.load expects.
    obs_float = obs.float().div(255).mul(2).sub(1)
    episode_id = dataset.add_episode(Episode(obs_float, act, rew, end, trunc, info))
    path = dataset._get_episode_path(episode_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"obs": obs, "act": act, "rew": rew, "end": end, "trunc": trunc, "info": info}, tmp)
    tmp.rename(path)
    return episode_id


def _manifest_checksums(hdf5_root: Path) -> dict[str, str]:
    return {str(p.relative_to(hdf5_root)): sha256_file(p) for p in sorted(Path(hdf5_root).rglob("manifest.jsonl"))}


def convert(
    hdf5_root: Path,
    out: Path,
    kinds: Sequence[str],
    test_frac: float,
    seed: int,
    shuffle_actions: bool,
) -> dict[str, Any]:
    """Assemble the dataset.

    Args:
        hdf5_root: HDF5 root.
        out: Output dataset directory; ``low_res`` must not exist yet.
        kinds: Kinds to include.
        test_frac: Requested test fraction per kind.
        seed: Split and derangement seed.
        shuffle_actions: Build the shuffled-action control.

    Returns:
        The contents of ``dataset_meta.json``.

    Raises:
        SystemExit: If there is nothing to convert, train or test would be empty, the output exists, or
            ``--shuffle-actions`` has fewer than 2 train episodes.
    """
    hdf5_root, out = Path(hdf5_root), Path(out)
    if (out / "low_res").exists():
        raise SystemExit(f"{out / 'low_res'} already exists; remove it or choose another --out")
    episodes = discover(hdf5_root, kinds)
    if not episodes:
        raise SystemExit(f"no episodes of kinds {list(kinds)} under {hdf5_root}")
    split = split_demos({e.demo_id: e.kind for e in episodes}, test_frac, seed)
    by_split = {s: [e for e in episodes if split[e.demo_id] == s] for s in SPLITS}
    for s, eps in by_split.items():
        if not eps:
            raise SystemExit(f"the {s} split is empty ({len(episodes)} episodes, {len(split)} demos); add demos")
    sources: dict[str, EpisodeRef] = {}
    if shuffle_actions:
        n_train = len(by_split["train"])
        if n_train < 2:
            raise SystemExit(f"--shuffle-actions needs at least 2 train episodes to swap, got {n_train}")
        perm = derangement(n_train, seed)
        sources = {e.episode_id: by_split["train"][perm[i]] for i, e in enumerate(by_split["train"])}

    table: list[dict[str, Any]] = []
    for s in SPLITS:
        dataset = Dataset(out / "low_res" / s, None, f"{s}_dataset", save_on_disk=False)
        for ref in by_split[s]:
            frames, actions, _ = read_episode(ref.path)
            if ref.episode_id in sources:
                actions = tile_to_length(_read_actions(sources[ref.episode_id].path), len(frames))
            index = _save_episode(dataset, ref, frames, actions)
            table.append(
                {
                    "split": s,
                    "index": index,
                    "episode_id": ref.episode_id,
                    "demo_id": ref.demo_id,
                    "kind": ref.kind,
                    "frames": ref.n_frames,
                    "actions_from": sources[ref.episode_id].episode_id if ref.episode_id in sources else ref.episode_id,
                }
            )
        dataset.save_to_default_path()

    (out / "split.json").write_text(json.dumps(dict(sorted(split.items())), indent=2) + "\n")
    meta = {
        "hdf5_root": str(hdf5_root),
        "kinds": list(kinds),
        "test_frac": test_frac,
        "seed": seed,
        "shuffle_actions": shuffle_actions,
        "source_manifests": _manifest_checksums(hdf5_root),
        "episodes": table,
    }
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    """Run the conversion.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    args = build_parser().parse_args(argv)
    kinds = [k for k in args.kinds.split(",") if k]
    meta = convert(Path(args.hdf5), Path(args.out), kinds, args.test_frac, args.seed, args.shuffle_actions)
    for s in SPLITS:
        rows = [r for r in meta["episodes"] if r["split"] == s]
        demos = {r["demo_id"] for r in rows}
        print(f"{s}: {len(rows)} episodes, {sum(r['frames'] for r in rows)} frames, {len(demos)} demos")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
