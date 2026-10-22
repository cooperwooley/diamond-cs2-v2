"""Re-check every sha256 in each ``manifest.jsonl`` under an HDF5 root (#17); used by #19, #24, #25.

Prints one line per demo and a final ``OK``/``FAILED`` summary; exits 1 on any mismatch or missing file.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="HDF5 root ($S/hdf5)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the manifests.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        0 if every checksum matches, else 1.
    """
    build_parser().parse_args(argv)
    # TODO(#17): walk manifest.jsonl files and compare sha256 of each listed file.
    raise NotImplementedError("TODO(#17): verify_manifests")


if __name__ == "__main__":
    raise SystemExit(main())
