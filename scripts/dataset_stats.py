"""Dataset statistics (#17, #24, #25, #26).

Two modes:

- ``dataset_stats.py <dataset_dir> [--json out.json]``: hours and frames per kind and split, episode-length
  histogram, per-slot label rates, mouse-bin histograms and the fraction of all-zero actions (#17, #25).
- ``dataset_stats.py --hdf5 <root> --summary``: one-line running total of hours on the Mill, used by the nightly
  paste-back in #24.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", help="assembled dataset directory ($S/datasets/<name>)")
    p.add_argument("--hdf5", help="HDF5 root for --summary ($S/hdf5)")
    p.add_argument("--summary", action="store_true", help="print only the hours total of --hdf5")
    p.add_argument("--json", help="also write the stats to this file")
    p.add_argument("--figure", help="write the stats figure to this file (#26)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Print the stats.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    args = build_parser().parse_args(argv)
    if args.summary and not args.hdf5:
        raise SystemExit("--summary needs --hdf5")
    if not args.summary and not args.dataset:
        raise SystemExit("give a dataset directory, or --hdf5 with --summary")
    # TODO(#17): compute and print the stats; TODO(#24): the --hdf5 --summary running total.
    raise NotImplementedError("TODO(#17): dataset_stats")


if __name__ == "__main__":
    raise SystemExit(main())
