"""Regenerate every report table and figure from ``results/*.json`` (#36).

Tables are booktabs ``.tex`` files with macros for best values; missing inputs yield a marked ``\\textbf{TBD}``
placeholder so the report compiles at any stage. Output is deterministic (byte-identical across runs).
"""

from __future__ import annotations

from pathlib import Path


def build_tables(results_dir: Path, out_dir: Path) -> list[Path]:
    """Write ``dataset, label_accuracy, h1, h2, carried, play_fps`` tables to ``out_dir/tables/``.

    Args:
        results_dir: Directory with the result JSONs.
        out_dir: Report directory.

    Returns:
        Paths of the written ``.tex`` files.
    """
    # TODO(#36): one builder per table; TBD placeholder when an input is missing.
    raise NotImplementedError("TODO(#36): build_tables")


def build_figures(results_dir: Path, out_dir: Path) -> list[Path]:
    """Write the PDF figures to ``out_dir/figures/``: H1 scatter and curves, H2 triptychs, loss curves, rollout
    strips, dataset stats.

    Args:
        results_dir: Directory with the result JSONs.
        out_dir: Report directory.

    Returns:
        Paths of the written figures.
    """
    raise NotImplementedError("TODO(#36): build_figures")


def run(results_dir: Path, out_dir: Path) -> None:
    """CLI entry: ``build_tables`` then ``build_figures``.

    Args:
        results_dir: Directory with the result JSONs.
        out_dir: Report directory.
    """
    raise NotImplementedError("TODO(#36): report.latex.run")
