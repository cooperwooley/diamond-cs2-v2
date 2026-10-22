#!/usr/bin/env python3
"""Convert synced low-res HDF5 episodes into DIAMOND ``.pt`` train/test datasets (#17, runs on the Mill).

Output: ``<out>/low_res/{train,test}`` (DIAMOND ``Dataset`` episodes plus ``info.pt``) and ``<out>/split.json``
(demo -> split). The split is by demo (no demo in both splits), deterministic for a seed, and puts at least 30% of
smoke demos (and at least 3) in test for H2. ``--shuffle-actions`` gives each train episode the action sequence of a
different train episode (seeded derangement, tiled or cropped to length); test keeps the true actions.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the conversion.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    build_parser().parse_args(argv)
    # TODO(#17): split by demo, write DIAMOND episodes, optional derangement of train actions, write split.json.
    raise NotImplementedError("TODO(#17): hdf5_to_diamond")


if __name__ == "__main__":
    raise SystemExit(main())
