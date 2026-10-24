"""Regenerate every report table, figure and number macro from ``results/*.json`` (#36).

``cs2wm report-assets --results results --out report`` writes

* ``<out>/tables/{dataset,label_accuracy,h1,h2,carried,play_fps}.tex``: each a booktabs ``tabular`` only (the report adds
  the float, caption and label), with the best value per column in bold where that is meaningful;
* ``<out>/tables/macros.tex``: a ``\\newcommand`` (ending in ``\\xspace``) for each ``\\TBD...`` placeholder of the
  report's ``placeholders.tex`` whose value can be computed. A macro whose input is missing is left undefined, so the
  report's ``\\providecommand`` placeholder shows;
* ``<out>/figures/{dataset_stats,h1_scatter,h1_per_bin,h2_triptych,loss_curves,rollout_strip}.pdf``.

A missing or unreadable input never raises: its table becomes a single ``\\textbf{TBD}`` cell (with a comment naming the
file and issue), its macros stay undefined and its figure is not written (a stale one is removed), so the report always
compiles. The ``.tex`` output is deterministic (byte-identical across runs: no timestamps, fixed number formatting,
JSON order preserved); PDFs are made reproducible as far as matplotlib allows.

Inputs (all under ``results/`` unless noted)
--------------------------------------------
``dataset_stats.json`` (``--stats`` overrides; ``scripts/dataset_stats.py --json``, #26), ``label_accuracy.json`` (#18),
``h1/summary.json`` and ``h1/<model>_frames.csv`` (#32), ``h2/summary.json`` and ``h2/<model>_triptychs.{pdf,png}``
(#33), ``carried/summary.json`` and ``carried/rollout_strip_NN.png`` (#34), ``upsampler_check.json`` (#27),
``play_fps.json`` (#28).
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import zlib
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

TABLE_NAMES = ("dataset", "label_accuracy", "h1", "h2", "carried", "play_fps")
SPLITS = ("train", "test")
FIGURE_NAMES = ("dataset_stats", "h1_scatter", "h1_per_bin", "h2_triptych", "loss_curves", "rollout_strip")

# What each figure is built from, for the message when it is removed.
FIGURE_INPUTS = {
    "dataset_stats": "dataset_stats.json",
    "h1_scatter": "h1/summary.json scatter data or h1/<model>_frames.csv",
    "h1_per_bin": "h1/summary.json",
    "h2_triptych": "h2/<model>_triptychs.pdf or .png",
    "loss_curves": "carried/summary.json",
    "rollout_strip": "carried/rollout_strip_NN.png",
}

# Input file (relative to the results directory) and the issue that produces it.
STATS_FILE, STATS_ISSUE = "dataset_stats.json", "#25"
LABEL_FILE, LABEL_ISSUE = "label_accuracy.json", "#18"
H1_FILE, H1_ISSUE = "h1/summary.json", "#32"
H2_FILE, H2_ISSUE = "h2/summary.json", "#33"
CARRIED_FILE, CARRIED_ISSUE = "carried/summary.json", "#34"
UPS_FILE, UPS_ISSUE = "upsampler_check.json", "#27"
FPS_FILE, FPS_ISSUE = "play_fps.json", "#28"

MODEL_ORDER = ("v2", "v1", "shuffled")  # the report's order; any other model follows alphabetically
# Null controls of H1: never bold, so a low-drift or high-r control cannot look like a result.
H1_NULL_MODELS = ("v1", "shuffled")
# Headline configuration of the play benchmark: the play loop's default (upsampler, 1 denoising step, fp32), then bf16.
FPS_HEADLINE = ("upsampler_1step_fp32", "upsampler_1step_bf16")
# Condition of the H2 macros: teacher-forced single-step predictions.
H2_HEADLINE = "single"
MISSING = "--"
SOURCE_DATE_EPOCH = "0"

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
SPLIT_COLORS = ("#CC79A7", "#E69F00")  # train, test of the dataset figure; distinct from the model colours
FIG_FONT_SIZE = 7


# --- reading inputs and formatting -----------------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object; None if the file is absent, unreadable or not an object."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _get(obj: Any, *keys: str | int) -> Any:
    """Follow ``keys`` through nested dicts and lists; None as soon as one is missing."""
    for key in keys:
        try:
            obj = obj[key]
        except (KeyError, IndexError, TypeError):
            return None
    return obj


def _finite(x: Any) -> float | None:
    """``x`` as a float if it is a finite real number, else None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def _num(x: Any, digits: int) -> str:
    """Fixed-point text; ``--`` for a missing value; a minus that works in text and math mode; never ``-0.00``."""
    v = _finite(x)
    if v is None:
        return MISSING
    s = f"{v:.{digits}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return "\\ensuremath{-}" + s[1:] if s.startswith("-") else s


def _int(x: Any) -> str:
    """Integer text with ``{,}`` thousands separators."""
    v = _finite(x)
    return MISSING if v is None else f"{round(v):,}".replace(",", "{,}")


def _pct(x: Any, digits: int = 1) -> str:
    """A fraction as a percentage number (no percent sign)."""
    v = _finite(x)
    return MISSING if v is None else _num(100 * v, digits)


def _percent(x: Any) -> float | None:
    """A fraction as a percentage value (None if missing)."""
    v = _finite(x)
    return None if v is None else 100 * v


_TEX_SPECIAL = {
    "\\": "\\textbackslash{}",
    "_": "\\_",
    "#": "\\#",
    "%": "\\%",
    "&": "\\&",
    "$": "\\$",
    "{": "\\{",
    "}": "\\}",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
}


def _tex_escape(text: Any) -> str:
    """Escape free text (model, dataset, group and script names) for a LaTeX table cell or comment line."""
    return re.sub(r"[\\_#%&${}~^]", lambda m: _TEX_SPECIAL[m[0]], " ".join(str(text).split()))


def _model_color(model: str) -> str:
    """A model's colour from its place in the canonical order, so it is the same in every figure."""
    if model in MODEL_ORDER:
        return COLORS[MODEL_ORDER.index(model)]
    return COLORS[len(MODEL_ORDER) + zlib.crc32(model.encode()) % (len(COLORS) - len(MODEL_ORDER))]


def _bold(text: str, on: bool) -> str:
    return f"\\textbf{{{text}}}" if on else text


def _ordered(names: Sequence[str]) -> list[str]:
    """Model names in the report's order (v2, v1, shuffled), others alphabetically after them."""
    known = [n for n in MODEL_ORDER if n in names]
    return known + sorted(n for n in names if n not in MODEL_ORDER)


def _best(
    values: Sequence[float | None], digits: int, pick: Callable[[Decimal], Decimal], min_count: int = 2
) -> set[int]:
    """Indices holding the best value, judged on the displayed value in exact decimal.

    Args:
        values: One value per row (None where missing or not eligible).
        digits: Decimals shown, so ties in the table are all bold (0.90 and 1.10 are equally far from 1).
        pick: Maps a value to a score; the lowest score wins (``-v`` for larger-is-better, ``abs(v - 1)`` for slopes).
        min_count: Fewest values that make a comparison; below it nothing is bold.
    """
    scored = {i: pick(Decimal(f"{v:.{digits}f}")) for i, v in enumerate(values) if v is not None}
    if len(scored) < min_count:
        return set()
    top = min(scored.values())
    return {i for i, s in scored.items() if s == top}


def _tabular(spec: str, lines: Sequence[str]) -> str:
    """A booktabs tabular; ``lines`` are the complete lines of the body (rules or ``a & b \\\\`` rows)."""
    return "\n".join([f"\\begin{{tabular}}{{{spec}}}", "\\toprule", *lines, "\\bottomrule", "\\end{tabular}"]) + "\n"


def _row(cells: Sequence[str]) -> str:
    return " & ".join(cells) + " \\\\"


def tbd_table(missing_file: str, issue: str) -> str:
    """A table whose only cell is ``\\textbf{TBD}``, with a comment naming the missing input.

    Args:
        missing_file: Path of the missing input, relative to the results directory.
        issue: The issue that produces it.
    """
    return (
        f"% TBD: results/{missing_file} is missing or unusable ({issue}); rerun `cs2wm report-assets` once it exists.\n"
        + _tabular("c", ["\\textbf{TBD} \\\\"])
    )


# --- tables ----------------------------------------------------------------------------------------------------------


def dataset_table(stats: dict[str, Any]) -> str | None:
    """Episodes, frames and hours per kind and split, then totals (``dataset_stats.json``).

    Args:
        stats: ``scripts/dataset_stats.py --json`` output (``by_kind``, ``train``, ``test``, ``total``).

    Returns:
        The tabular, or None if the totals are missing.
    """
    if not all(isinstance(_get(stats, s), dict) for s in ("train", "test", "total")):
        return None
    lines = [_row(["Kind", "Split", "Episodes", "Frames", "Hours"]), "\\midrule"]
    by_kind = _get(stats, "by_kind") or {}
    for i, kind in enumerate(sorted(by_kind)):
        if i:
            lines.append("\\addlinespace[2pt]")
        for split in ("train", "test"):
            v = by_kind[kind].get(split) or {}
            lines.append(
                _row(
                    [
                        f"\\texttt{{{_tex_escape(kind)}}}",
                        _tex_escape(split),
                        _int(v.get("episodes")),
                        _int(v.get("frames")),
                        _num(v.get("hours"), 1),
                    ]
                )
            )
    if by_kind:
        lines.append("\\midrule")
    for name, label in (("train", "train"), ("test", "test"), ("total", "all")):
        v = stats[name]
        lines.append(_row(["Total", label, _int(v.get("episodes")), _int(v.get("frames")), _num(v.get("hours"), 1)]))
    return _tabular("llrrr", lines)


GROUP_LABELS = {
    "wasd": "WASD",
    "jump": "jump",
    "crouch": "crouch",
    "fire": "fire",
    "reload": "reload",
    "switch": "weapon switch",
    "mouse_x_sign": "mouse-$x$ sign",
    "mouse_x_bin": "mouse-$x$ bin $\\pm1$",
    "mouse_y": "mouse-$y$",
}


def label_accuracy_table(acc: dict[str, Any]) -> str | None:
    """Per-group judged rows, correct rows and accuracy (``label_accuracy.json``).

    Args:
        acc: ``score_review_sheet`` output (``groups`` of ``n, correct, accuracy, min_accuracy``).

    Returns:
        The tabular, or None if there are no groups.
    """
    groups = _get(acc, "groups")
    if not isinstance(groups, dict) or not groups:
        return None
    lines = [_row(["Group", "Judged", "Correct", "Acc.\\ (\\%)", "Req.\\ (\\%)"]), "\\midrule"]
    for name, g in groups.items():
        label = GROUP_LABELS.get(name) or _tex_escape(name)
        lines.append(
            _row(
                [
                    label,
                    _int(_get(g, "n")),
                    _int(_get(g, "correct")),
                    _pct(_get(g, "accuracy")),
                    _pct(_get(g, "min_accuracy"), 0),
                ]
            )
        )
    return _tabular("lrrrr", lines)


def _h1_axis(summary: dict[str, Any], model: str, axis: str) -> dict[str, Any]:
    return _get(summary, "results", model, "fullres", axis) or {}


def h1_table(summary: dict[str, Any]) -> str | None:
    """Yaw and pitch Pearson r, slope and sign accuracy per model, plus no-op yaw drift (``h1/summary.json``, 150x280).

    The best model per column is bold among the controllable models only (``H1_NULL_MODELS`` are never bold):
    highest r and sign accuracy, slope closest to 1, lowest drift.

    Args:
        summary: ``run_h1`` summary.

    Returns:
        The tabular, or None if there are no model results.
    """
    results = _get(summary, "results")
    if not isinstance(results, dict) or not results:
        return None
    models = _ordered(list(results))
    cols: list[tuple[str, str, int, Callable[[Decimal], Decimal], Callable[[str], Any]]] = []
    for axis in ("yaw", "pitch"):
        cols += [
            (axis, "r", 2, lambda v: -v, lambda m, a=axis: _finite(_h1_axis(summary, m, a).get("pearson_r"))),
            (axis, "slope", 2, lambda v: abs(v - 1), lambda m, a=axis: _finite(_h1_axis(summary, m, a).get("slope"))),
            (
                axis,
                "sign",
                1,
                lambda v: -v,
                lambda m, a=axis: _percent(_h1_axis(summary, m, a).get("sign_accuracy")),
            ),
        ]
    cols.append(
        (
            "drift",
            "drift",
            3,
            lambda v: v,
            lambda m: _finite(_get(results, m, "fullres", "noop_drift", "mean_abs_dyaw_deg", "mean")),
        )
    )
    table = [[col[4](m) for col in cols] for m in models]
    controls = [m for m in models if m in H1_NULL_MODELS]
    winners = [
        _best(
            [None if m in H1_NULL_MODELS else row[j] for m, row in zip(models, table)],
            cols[j][2],
            cols[j][3],
            min_count=1,
        )
        for j in range(len(cols))
    ]
    lines = [
        "& \\multicolumn{3}{c}{Yaw} & \\multicolumn{3}{c}{Pitch} & no-op yaw \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        _row(["Model", "$r$", "slope", "sign (\\%)", "$r$", "slope", "sign (\\%)", "drift ($^\\circ$/frame)"]),
        "\\midrule",
    ]
    for i, m in enumerate(models):
        cells = [_num(table[i][j], cols[j][2]) for j in range(len(cols))]
        lines.append(_row([_tex_escape(m), *(_bold(c, i in winners[j]) for j, c in enumerate(cells))]))
    note = ""
    if controls:
        note = f"% Bold marks the best controllable model per column; null controls ({', '.join(map(_tex_escape, controls))}) are never bold.\n"
    return note + _tabular(
        "@{}l@{\\hspace{5pt}}r@{\\hspace{3pt}}r@{\\hspace{3pt}}r@{\\hspace{6pt}}r@{\\hspace{3pt}}r@{\\hspace{3pt}}r@{\\hspace{6pt}}r@{}",
        lines,
    )


def h2_table(summary: dict[str, Any]) -> str | None:
    """LPIPS and SSIM inside smoke vs on static geometry per model and condition (``h2/summary.json``).

    Tolerates missing models, conditions and metrics (shown as ``--``). Within each condition the best model per
    column is bold (lowest LPIPS, highest SSIM).

    Args:
        summary: ``run_h2`` summary (``results[model][condition][region][metric].mean``).

    Returns:
        The tabular, or None if there are no model results.
    """
    results = _get(summary, "results")
    if not isinstance(results, dict) or not results:
        return None
    models = _ordered(list(results))
    titles = {"single": "Single step (teacher-forced)", "rollout": "Autoregressive rollout"}
    conditions = [c for c in titles if any(isinstance(_get(results, m, c), dict) for m in models)]
    conditions += sorted({c for m in models for c in (results[m] or {}) if c not in titles})
    if not conditions:
        return None
    cols = [("smoke", "lpips"), ("static", "lpips"), ("smoke", "ssim"), ("static", "ssim")]
    lines = [
        "& \\multicolumn{2}{c}{LPIPS} & \\multicolumn{2}{c}{SSIM} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        _row(["Model", "smoke", "static", "smoke", "static"]),
    ]
    for cond in conditions:
        present = [m for m in models if isinstance(_get(results, m, cond), dict)]
        values = [[_finite(_get(results, m, cond, r, k, "mean")) for r, k in cols] for m in present]
        winners = [
            _best([row[j] for row in values], 3, (lambda v: v) if cols[j][1] == "lpips" else (lambda v: -v))
            for j in range(len(cols))
        ]
        lines += ["\\midrule", f"\\multicolumn{{5}}{{l}}{{\\emph{{{titles.get(cond) or _tex_escape(cond)}}}}} \\\\"]
        for i, m in enumerate(present):
            lines.append(
                _row([_tex_escape(m), *(_bold(_num(values[i][j], 3), i in winners[j]) for j in range(len(cols)))])
            )
    return _tabular("@{}lrrrr@{}", lines)


def carried_table(summary: dict[str, Any]) -> str | None:
    """FID per model and test set (``carried/summary.json`` ``fid.rows``).

    Mean +- std over seeds when there are several. The lowest FID is bold within each test set that has two or more
    rows, because FIDs across test sets are not comparable.

    Args:
        summary: ``run_carried`` summary.

    Returns:
        The tabular, or None if there are no FID rows.
    """
    rows = _get(summary, "fid", "rows")
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        return None
    set_names = {"v2_test": "v2", "v1_test": "v1"}
    source_names = {"gt": "GT", "zeros": "zero"}
    winners: set[int] = set()
    for ds in dict.fromkeys(r.get("dataset") for r in rows):
        idx = [i for i, r in enumerate(rows) if r.get("dataset") == ds]
        winners |= {idx[k] for k in _best([_finite(rows[i].get("fid")) for i in idx], 1, lambda v: v)}
    lines = [_row(["Model", "Test set", "Actions", "FID"]), "\\midrule"]
    for i, r in enumerate(rows):
        fid = _bold(_num(r.get("fid"), 1), i in winners)
        if _finite(r.get("fid_std")) is not None:
            fid += f" $\\pm$ {_num(r['fid_std'], 1)}"
        source = r.get("action_source")
        lines.append(
            _row(
                [
                    _tex_escape(r.get("model", MISSING)),
                    set_names.get(r.get("dataset")) or _tex_escape(r.get("dataset", MISSING)),
                    source_names.get(source) or _tex_escape(source if source is not None else MISSING),
                    fid,
                ]
            )
        )
    return _tabular("lllr", lines)


_FPS_KEY = re.compile(r"^(denoiser|upsampler)_(\d+)step_(fp32|bf16)$")


def play_fps_table(report: dict[str, Any]) -> str | None:
    """Frames per second per configuration: denoiser or upsampler, 1 or 3 steps, fp32 or bf16 (``play_fps.json``).

    Skipped configurations show ``--``. The fastest configuration per precision column is bold.

    Args:
        report: ``loop.benchmark`` report.

    Returns:
        The tabular, or None if no configuration is recognised.
    """
    results = _get(report, "results")
    if not isinstance(results, dict):
        return None
    configs: dict[tuple[str, int], dict[str, float | None]] = {}
    for key, r in results.items():
        m = _FPS_KEY.match(key)
        if m:
            fps = _finite(_get(r, "fps")) if _get(r, "status") == "ok" else None
            configs.setdefault((m[1], int(m[2])), {})[m[3]] = fps
    if not configs:
        return None
    order = sorted(configs, key=lambda c: (c[0] != "denoiser", c[1]))
    precisions = ("fp32", "bf16")
    winners = [_best([configs[c].get(p) for c in order], 1, lambda v: -v) for p in precisions]
    lines = [_row(["Configuration", "fp32 (fps)", "bf16 (fps)"]), "\\midrule"]
    for i, c in enumerate(order):
        label = f"{'Denoiser + upsampler' if c[0] == 'upsampler' else 'Denoiser only'}, {c[1]} step{'s' if c[1] > 1 else ''}"
        lines.append(
            _row([label, *(_bold(_num(configs[c].get(p), 1), i in winners[j]) for j, p in enumerate(precisions))])
        )
    gpu = _get(report, "gpu") or _get(report, "device")
    comment = f"% Measured on: {_tex_escape(gpu)}.\n" if isinstance(gpu, str) else ""
    return comment + _tabular("lrr", lines)


@dataclass(frozen=True)
class _TableSpec:
    name: str
    file: str
    issue: str
    builder: Callable[[dict[str, Any]], str | None]


def _table_specs(stats_file: str) -> list[_TableSpec]:
    return [
        _TableSpec("dataset", stats_file, STATS_ISSUE, dataset_table),
        _TableSpec("label_accuracy", LABEL_FILE, LABEL_ISSUE, label_accuracy_table),
        _TableSpec("h1", H1_FILE, H1_ISSUE, h1_table),
        _TableSpec("h2", H2_FILE, H2_ISSUE, h2_table),
        _TableSpec("carried", CARRIED_FILE, CARRIED_ISSUE, carried_table),
        _TableSpec("play_fps", FPS_FILE, FPS_ISSUE, play_fps_table),
    ]


# --- macros ----------------------------------------------------------------------------------------------------------


@dataclass
class Inputs:
    """The loaded result JSONs; an entry is None if the file is missing or unreadable."""

    stats: dict[str, Any] | None
    label: dict[str, Any] | None
    h1: dict[str, Any] | None
    h2: dict[str, Any] | None
    carried: dict[str, Any] | None
    ups: dict[str, Any] | None
    fps: dict[str, Any] | None
    stats_file: str = STATS_FILE


def load_inputs(results_dir: Path, stats: Path | None = None) -> Inputs:
    """Load every result JSON.

    Args:
        results_dir: Directory with the result JSONs.
        stats: Dataset statistics JSON, default ``<results_dir>/dataset_stats.json``.
    """
    results_dir = Path(results_dir)
    stats_path = Path(stats) if stats is not None else results_dir / STATS_FILE
    return Inputs(
        stats=_load_json(stats_path),
        label=_load_json(results_dir / LABEL_FILE),
        h1=_load_json(results_dir / H1_FILE),
        h2=_load_json(results_dir / H2_FILE),
        carried=_load_json(results_dir / CARRIED_FILE),
        ups=_load_json(results_dir / UPS_FILE),
        fps=_load_json(results_dir / FPS_FILE),
        stats_file=str(stats) if stats is not None else STATS_FILE,
    )


def _h1_estimator(h1: dict[str, Any]) -> str:
    return _get(h1, "config", "estimator") or "phasecorr"


def _yaw_validation(h1: dict[str, Any], key: str) -> Any:
    return _get(h1, "estimator_validation", "resolutions", "150x280", "estimators", _h1_estimator(h1), "yaw", key)


def _fid_row(carried: dict[str, Any], model: str, dataset: str) -> Any:
    """FID of ``model`` on ``dataset``; with several action sources, the zero-action row (v1's training condition)."""
    rows = [r for r in _get(carried, "fid", "rows") or [] if isinstance(r, dict)]
    rows = [r for r in rows if r.get("model") == model and r.get("dataset") == dataset]
    rows.sort(key=lambda r: r.get("action_source") != ("zeros" if model == "v1" else "gt"))
    return _finite(rows[0].get("fid")) if rows else None


def _fps_headline(fps: dict[str, Any]) -> tuple[str, float] | None:
    """The headline configuration's name and fps: the first of ``FPS_HEADLINE`` that ran."""
    for name in FPS_HEADLINE:
        value = _finite(_get(fps, "results", name, "fps")) if _get(fps, "results", name, "status") == "ok" else None
        if value is not None:
            return name, value
    return None


def _label_min(label: dict[str, Any]) -> float | None:
    """Lowest accuracy over the groups that were judged (``n > 0``; unscored groups do not count)."""
    groups = _get(label, "groups")
    accs = [
        _finite(_get(g, "accuracy"))
        for g in (groups.values() if isinstance(groups, dict) else ())
        if (_finite(_get(g, "n")) or 0) > 0
    ]
    accs = [a for a in accs if a is not None]
    return min(accs) if accs else None


H2_CONDITION_NAMES = {"single": "single-step", "rollout": "rollout"}


def _h2_condition(h2: dict[str, Any]) -> str:
    name = H2_CONDITION_NAMES.get(H2_HEADLINE, H2_HEADLINE)
    steps = _finite(_get(h2, "config", "rollout_steps"))
    return f"{int(steps)}-step rollout" if H2_HEADLINE == "rollout" and steps else name


# name, human-readable source, value builder (None -> undefined). The names are those of the report's placeholders.tex.
_MACROS: list[tuple[str, str, Callable[[Inputs], str | None]]] = [
    (
        "TBDhours",
        "dataset_stats.json total.hours",
        lambda i: _opt(i.stats, lambda s: _num(_get(s, "total", "hours"), 1)),
    ),
    (
        "TBDtrainhours",
        "dataset_stats.json train.hours",
        lambda i: _opt(i.stats, lambda s: _num(_get(s, "train", "hours"), 1)),
    ),
    (
        "TBDtesthours",
        "dataset_stats.json test.hours",
        lambda i: _opt(i.stats, lambda s: _num(_get(s, "test", "hours"), 1)),
    ),
    (
        "TBDnumepisodes",
        "dataset_stats.json total.episodes",
        lambda i: _opt(i.stats, lambda s: _int(_get(s, "total", "episodes"))),
    ),
    (
        "TBDlabelmin",
        "label_accuracy.json min of groups.*.accuracy over the groups with n > 0 (percent)",
        lambda i: _opt(i.label, lambda s: _pct(_label_min(s))),
    ),
    (
        "TBDlabelxsign",
        "label_accuracy.json groups.mouse_x_sign.accuracy (percent)",
        lambda i: _opt(i.label, lambda s: _pct(_get(s, "groups", "mouse_x_sign", "accuracy"))),
    ),
    (
        "TBDestimatorR",
        "h1/summary.json estimator_validation.resolutions.150x280.estimators.<config.estimator>.yaw.r",
        lambda i: _opt(i.h1, lambda s: _num(_yaw_validation(s, "r"), 2)),
    ),
    (
        "TBDdegperpx",
        "h1/summary.json estimator_validation.resolutions.150x280.estimators.<config.estimator>.yaw.slope_deg_per_px",
        lambda i: _opt(i.h1, lambda s: _num(_yaw_validation(s, "slope_deg_per_px"), 3), "$^\\circ$/px"),
    ),
    (
        "TBDhOneRvTwo",
        "h1/summary.json results.v2.fullres.yaw.pearson_r",
        lambda i: _opt(i.h1, lambda s: _num(_h1_axis(s, "v2", "yaw").get("pearson_r"), 2)),
    ),
    (
        "TBDhOneRvOne",
        "h1/summary.json results.v1.fullres.yaw.pearson_r",
        lambda i: _opt(i.h1, lambda s: _num(_h1_axis(s, "v1", "yaw").get("pearson_r"), 2)),
    ),
    (
        "TBDhOneRshuf",
        "h1/summary.json results.shuffled.fullres.yaw.pearson_r",
        lambda i: _opt(i.h1, lambda s: _num(_h1_axis(s, "shuffled", "yaw").get("pearson_r"), 2)),
    ),
    (
        "TBDhOneSignVtwo",
        "h1/summary.json results.v2.fullres.yaw.sign_accuracy (percent)",
        lambda i: _opt(i.h1, lambda s: _pct(_h1_axis(s, "v2", "yaw").get("sign_accuracy")), "\\%"),
    ),
    (
        "TBDhTwoSmoke",
        f"h2/summary.json results.v2.{H2_HEADLINE}.smoke.lpips.mean",
        lambda i: _opt(i.h2, lambda s: _num(_get(s, "results", "v2", H2_HEADLINE, "smoke", "lpips", "mean"), 3)),
    ),
    (
        "TBDhTwoStatic",
        f"h2/summary.json results.v2.{H2_HEADLINE}.static.lpips.mean",
        lambda i: _opt(i.h2, lambda s: _num(_get(s, "results", "v2", H2_HEADLINE, "static", "lpips", "mean"), 3)),
    ),
    (
        "TBDfidVtwo",
        "carried/summary.json fid.rows[model=v2, dataset=v2_test].fid",
        lambda i: _opt(i.carried, lambda s: _num(_fid_row(s, "v2", "v2_test"), 1)),
    ),
    (
        "TBDfidVone",
        "carried/summary.json fid.rows[model=v1, dataset=v1_test].fid (zero-action row)",
        lambda i: _opt(i.carried, lambda s: _num(_fid_row(s, "v1", "v1_test"), 1)),
    ),
    (
        "TBDfidShift",
        "carried/summary.json fid.rows[model=v1, dataset=v2_test].fid (zero-action row)",
        lambda i: _opt(i.carried, lambda s: _num(_fid_row(s, "v1", "v2_test"), 1)),
    ),
    (
        "TBDupsVtwo",
        "upsampler_check.json v2.upsampler.lpips.mean",
        lambda i: _opt(i.ups, lambda s: _num(_get(s, "v2", "upsampler", "lpips", "mean"), 3)),
    ),
    (
        "TBDupsVone",
        "upsampler_check.json v1.upsampler.lpips.mean",
        lambda i: _opt(i.ups, lambda s: _num(_get(s, "v1", "upsampler", "lpips", "mean"), 3)),
    ),
    (
        "TBDupsBicubic",
        "upsampler_check.json v2.bicubic.lpips.mean",
        lambda i: _opt(i.ups, lambda s: _num(_get(s, "v2", "bicubic", "lpips", "mean"), 3)),
    ),
    (
        "TBDfps",
        "play_fps.json results.upsampler_1step_fp32.fps (else upsampler_1step_bf16)",
        lambda i: _opt(i.fps, lambda s: _num((_fps_headline(s) or (None, None))[1], 1)),
    ),
    (
        "TBDfpsgpu",
        "play_fps.json gpu",
        lambda i: _opt(
            i.fps, lambda s: _tex_escape(s["gpu"]) if isinstance(s.get("gpu"), str) and s["gpu"] else MISSING
        ),
    ),
    (
        "TBDhTwoCondition",
        f"constant H2_HEADLINE = {H2_HEADLINE!r} (the condition of TBDhTwoSmoke and TBDhTwoStatic)",
        lambda i: _opt(i.h2, _h2_condition),
    ),
]

# Placeholders that no result JSON carries, with the reason (listed in the macros.tex header).
UNCOMPUTABLE = {
    "TBDcsbuild": "CS2 build IDs are in the demo sidecars (#24), not in a results JSON",
    "TBDhlaever": "the HLAE release is pinned in the generate config (#14), not in a results JSON",
    "TBDnumdemos": "dataset_stats.json counts episodes, not demos (#24)",
    "TBDqadroprate": "the QA drop rate comes from the extract QA reports (#19), not from a results JSON",
    "TBDwasdlag": "label_accuracy.json has per-group accuracy only; the WASD lag is not measured (#18)",
    "TBDtrainwall": "no results JSON records the training wall-clock time (#29)",
    "TBDmark": "defined by placeholders.tex itself",
}


def _opt(data: dict[str, Any] | None, fn: Callable[[dict[str, Any]], str], suffix: str = "") -> str | None:
    """``fn(data)`` plus ``suffix``; None if the input is missing or the value could not be computed."""
    if data is None:
        return None
    value = fn(data)
    return None if value == MISSING else value + suffix


def _macro_notes(inputs: Inputs) -> list[str]:
    """Comment lines for macros whose value depends on a choice or a fallback."""
    notes = [
        f"%   TBDhTwoSmoke, TBDhTwoStatic: the {H2_HEADLINE!r} condition (see TBDhTwoCondition); the h2 table has both.",
        f"%   TBDlabelmin: lowest accuracy over the label groups with n > 0, of {', '.join(LABEL_GROUPS_NOTE)}.",
    ]
    headline = _fps_headline(inputs.fps) if inputs.fps is not None else None
    if headline is not None and headline[0] != FPS_HEADLINE[0]:
        notes.append(f"%   TBDfps: FELL BACK to {headline[0]} because {FPS_HEADLINE[0]} did not run.")
    return notes


LABEL_GROUPS_NOTE = ("wasd", "jump", "crouch", "fire", "reload", "switch", "mouse_x_sign", "mouse_x_bin", "mouse_y")


def build_macros(inputs: Inputs) -> str:
    """The text of ``tables/macros.tex``: a header listing every macro and its source, then the definitions.

    Args:
        inputs: The loaded result JSONs.
    """
    defined: list[tuple[str, str, str]] = []
    undefined: list[tuple[str, str]] = []
    for name, source, fn in _MACROS:
        value = fn(inputs)
        if value is None:
            undefined.append((name, source))
        else:
            defined.append((name, source, value))
    lines = [
        "% Generated by `cs2wm report-assets` from results/*.json. Do not edit by hand.",
        "% Each macro below is defined only when its source value exists; the others keep the \\providecommand",
        "% placeholder of placeholders.tex.",
        "%",
        "% Defined (macro: source):",
        *(f"%   \\{n}: {s}" for n, s, _ in defined),
        "% Notes:",
        *_macro_notes(inputs),
        "% Not defined (missing input):",
        *(f"%   \\{n}: {s}" for n, s in undefined),
        "% Not computable from any results JSON:",
        *(f"%   \\{n}: {why}" for n, why in sorted(UNCOMPUTABLE.items()) if n != "TBDmark"),
        "%",
        "\\RequirePackage{xspace}",
        "\\xspaceaddexceptions{\\,}% the report writes \\TBDfps\\,fps and \\TBDlabelmin\\,\\%",
        *(f"\\newcommand{{\\{n}}}{{{v}\\xspace}}" for n, _, v in defined),
    ]
    return "\n".join(lines) + "\n"


# --- figures ---------------------------------------------------------------------------------------------------------


@contextmanager
def _reproducible_pdf() -> Iterator[Any]:
    """Matplotlib (Agg, fixed fonts and sizes) with PDF output that does not depend on the time it is made."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    old_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH
    try:
        with matplotlib.rc_context(
            {
                "font.family": "DejaVu Sans",
                "font.size": FIG_FONT_SIZE,
                "axes.titlesize": FIG_FONT_SIZE,
                "axes.labelsize": FIG_FONT_SIZE,
                "legend.fontsize": FIG_FONT_SIZE - 1,
                "xtick.labelsize": FIG_FONT_SIZE - 1,
                "ytick.labelsize": FIG_FONT_SIZE - 1,
                "pdf.fonttype": 42,
                "axes.grid": True,
                "grid.alpha": 0.25,
            }
        ):
            yield plt
    finally:
        if old_epoch is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = old_epoch


def _save(fig: Any, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Title": None})
    plt.close(fig)


def fig_dataset_stats(stats: dict[str, Any], path: Path) -> bool:
    """Label rates and mouse-bin histograms, train vs test (``dataset_stats.json``).

    Returns:
        Whether a figure was written (False if the statistics lack the label rates).
    """
    splits = [s for s in SPLITS if isinstance(_get(stats, s, "label_rates"), dict)]
    if not splits:
        return False
    with _reproducible_pdf() as plt:
        fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.4), constrained_layout=True)
        names = list(stats[splits[0]]["label_rates"])
        pos = np.arange(len(names))
        width = 0.8 / len(splits)
        for i, split in enumerate(splits):
            rates = [_finite(stats[split]["label_rates"].get(n)) or 0.0 for n in names]
            axes[0].bar(
                pos + (i - (len(splits) - 1) / 2) * width,
                rates,
                width,
                label=split,
                color=SPLIT_COLORS[SPLITS.index(split)],
            )
        axes[0].set_xticks(pos, names, rotation=60)
        axes[0].set_ylabel("fraction of frames")
        axes[0].set_title("Label rates")
        axes[0].legend(frameon=False)
        for ax, axis in zip(axes[1:], ("x", "y")):
            hist = _get(stats, splits[0], f"mouse_{axis}_hist") or {}
            bins = np.arange(len(hist))
            for i, split in enumerate(splits):
                counts = np.array(list((_get(stats, split, f"mouse_{axis}_hist") or {}).values()), dtype=np.float64)
                if len(counts) != len(bins):
                    continue
                ax.bar(
                    bins + (i - (len(splits) - 1) / 2) * width,
                    counts / max(counts.sum(), 1),
                    width,
                    color=SPLIT_COLORS[SPLITS.index(split)],
                )
            ax.set_xticks(bins, [f"{float(k):g}" for k in hist], rotation=90)
            ax.set_yscale("log")
            ax.set_ylabel("fraction of frames")
            ax.set_title(f"Mouse {axis} bin (centre, degrees)")
        _save(fig, path)
    return True


def _read_frames_csv(path: Path) -> dict[str, np.ndarray] | None:
    """Commanded and estimated yaw/pitch (150x280) of the rows that enter the H1 statistics."""
    cols = ("cmd_yaw_deg", "cmd_pitch_deg", "est_yaw_deg", "est_pitch_deg")
    out: dict[str, list[float]] = {c: [] for c in cols}
    try:
        with Path(path).open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("in_stats") == "0":
                    continue
                for c in cols:
                    out[c].append(float(row[c]) if row.get(c) not in (None, "") else math.nan)
    except (OSError, ValueError, KeyError):
        return None
    return {c: np.array(v) for c, v in out.items()} if out[cols[0]] else None


def _scatter_from_summary(h1: dict[str, Any], model: str) -> dict[str, np.ndarray] | None:
    """The compact scatter block ``results.<model>.fullres.scatter.points`` (per context and script means)."""
    points = _get(h1, "results", model, "fullres", "scatter", "points")
    if not isinstance(points, list) or not points:
        return None
    cols = ("cmd_yaw_deg", "cmd_pitch_deg", "est_yaw_deg", "est_pitch_deg")
    keys = ("commanded_yaw_deg", "commanded_pitch_deg", "est_yaw_deg", "est_pitch_deg")
    out = {c: [] for c in cols}
    for pt in points:
        for c, k in zip(cols, keys):
            v = _finite(_get(pt, k))
            out[c].append(math.nan if v is None else v)
    return {c: np.array(v) for c, v in out.items()}


def fig_h1_scatter(h1: dict[str, Any], h1_dir: Path, path: Path) -> bool:
    """Commanded vs estimated rotation, all models overlaid.

    Plots ``results.<model>.fullres.scatter.points`` of the summary (the mean over the in-stats frame pairs of every
    context and script); if a model has none, falls back to its per-frame ``<model>_frames.csv`` (rows with
    ``in_stats`` 1).

    Returns:
        Whether a figure was written (False if no model has scatter data).
    """
    names = _ordered(sorted({*(_get(h1, "models") or {}), *(_get(h1, "results") or {})}))
    data = {m: _scatter_from_summary(h1, m) or _read_frames_csv(h1_dir / f"{m}_frames.csv") for m in names}
    data = {m: d for m, d in data.items() if d is not None}
    if not data:
        return False
    with _reproducible_pdf() as plt:
        fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0), constrained_layout=True)
        for ax, axis in zip(axes, ("yaw", "pitch")):
            lim = 0.0
            for m, d in data.items():
                cmd, est = d[f"cmd_{axis}_deg"], d[f"est_{axis}_deg"]
                ok = np.isfinite(cmd) & np.isfinite(est)
                ax.scatter(
                    cmd[ok], est[ok], s=2, alpha=0.4, color=_model_color(m), linewidths=0, rasterized=True, label=m
                )
                if ok.any():
                    lim = max(lim, float(np.abs(np.r_[cmd[ok], est[ok]]).max()))
            lim = max(lim * 1.05, 0.5)
            ax.plot([-lim, lim], [-lim, lim], color="0.3", lw=0.8, ls="--")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel(f"commanded {axis} ($^\\circ$/frame)")
            ax.set_ylabel(f"estimated {axis} ($^\\circ$/frame)")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, frameon=False, markerscale=4, loc="upper left")
        for lh in axes[0].get_legend().legend_handles:
            lh.set_alpha(1)
        _save(fig, path)
    return True


def fig_h1_per_bin(h1: dict[str, Any], path: Path) -> bool:
    """Per-bin mean estimated rotation with 95 % bootstrap CI against the commanded value (``results.*.per_bin``).

    Returns:
        Whether a figure was written (False if no model has per-bin data).
    """
    results = _get(h1, "results")
    if not isinstance(results, dict):
        return False
    series: dict[str, dict[str, list[tuple[float, float, float, float]]]] = {}
    for m in _ordered(list(results)):
        for axis in ("yaw", "pitch"):
            pts = []
            for b in _get(results, m, "fullres", "per_bin") or []:
                v = [
                    _finite(x)
                    for x in (_get(b, f"commanded_{axis}_deg"), *(_get(b, axis, k) for k in ("mean", "ci_lo", "ci_hi")))
                ]
                if _get(b, "kind") in (axis, "noop") and None not in v:
                    pts.append(tuple(v))
            if pts:
                series.setdefault(m, {})[axis] = sorted(pts)
    if not series:
        return False
    with _reproducible_pdf() as plt:
        fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0), constrained_layout=True)
        for ax, axis in zip(axes, ("yaw", "pitch")):
            lim = 0.1
            for m, per_axis in series.items():
                if axis not in per_axis:
                    continue
                x, y, lo, hi = (np.array(c) for c in zip(*per_axis[axis]))
                ax.errorbar(
                    x,
                    y,
                    yerr=np.vstack([y - lo, hi - y]),
                    marker="o",
                    ms=2,
                    lw=0.8,
                    capsize=1.5,
                    color=_model_color(m),
                    label=m,
                )
                lim = max(lim, float(np.abs(np.r_[x, lo, hi]).max()))
            lim *= 1.1
            ax.plot([-lim, lim], [-lim, lim], color="0.3", lw=0.8, ls="--")
            ax.set_xlabel(f"commanded {axis} ($^\\circ$/frame)")
            ax.set_ylabel(f"estimated {axis} ($^\\circ$/frame)")
        axes[0].legend(frameon=False, loc="upper left")
        _save(fig, path)
    return True


def _png_to_pdf(png: Path, path: Path, width_in: float = 7.0) -> bool:
    """Embed a PNG as a one-image PDF page of the PNG's aspect ratio."""
    import matplotlib.image as mpimg

    try:
        img = mpimg.imread(png)
    except (OSError, ValueError):
        return False
    h, w = img.shape[:2]
    with _reproducible_pdf() as plt:
        fig = plt.figure(figsize=(width_in, width_in * h / w))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(img, interpolation="lanczos")
        ax.axis("off")
        _save(fig, path)
    return True


def fig_h2_triptych(h2_dir: Path, path: Path) -> bool:
    """The H2 triptych figure of the producer (GT | generated | mask): v2's if present, else the first model's.

    Returns:
        Whether a figure was written (False if the producer wrote none).
    """
    for ext in ("pdf", "png"):
        names = sorted(h2_dir.glob(f"*_triptychs.{ext}"), key=lambda p: (p.name != f"v2_triptychs.{ext}", p.name))
        if not names:
            continue
        if ext == "pdf":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(names[0].read_bytes())
            return True
        return _png_to_pdf(names[0], path, width_in=3.4)
    return False


def fig_loss_curves(carried: dict[str, Any], path: Path) -> bool:
    """Test loss against epoch for v1 and v2 (``loss_curve`` of the carried summary).

    Returns:
        Whether a figure was written (False if neither model has a test-loss series).
    """
    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, epochs, losses in (
        (
            "v1",
            _get(carried, "loss_curve", "v1", "test", "epochs"),
            _get(carried, "loss_curve", "v1", "test", "denoiser"),
        ),
        ("v2", _get(carried, "loss_curve", "v2", "epochs"), _get(carried, "loss_curve", "v2", "test_loss")),
    ):
        if isinstance(epochs, list) and isinstance(losses, list) and len(epochs) == len(losses):
            pts = [(_finite(e), _finite(v)) for e, v in zip(epochs, losses)]
            pts = sorted((e, v) for e, v in pts if e is not None and v is not None)
            if pts:
                series.append((name, np.array([p[0] for p in pts]), np.array([p[1] for p in pts])))
    if not series:
        return False
    with _reproducible_pdf() as plt:
        fig, ax = plt.subplots(figsize=(3.4, 2.2), constrained_layout=True)
        for name, x, y in series:
            ax.plot(x, y, marker="o", ms=2.5, lw=1, color=_model_color(name), label=name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("test loss (denoiser MSE)")
        ax.legend(frameon=False)
        _save(fig, path)
    return True


def fig_rollout_strip(carried_dir: Path, path: Path) -> bool:
    """The first ``rollout_strip_NN.png`` of the carried eval, as a vector-page PDF.

    Returns:
        Whether a figure was written (False if the producer wrote no strip).
    """
    strips = sorted(carried_dir.glob("rollout_strip_*.png"))
    return bool(strips) and _png_to_pdf(strips[0], path)


# --- entry points ----------------------------------------------------------------------------------------------------


def build_tables(results_dir: Path, out_dir: Path, stats: Path | None = None) -> list[Path]:
    """Write ``dataset, label_accuracy, h1, h2, carried, play_fps`` tables and ``macros.tex`` to ``out_dir/tables/``.

    A table whose input is missing or unusable is written as a single ``\\textbf{TBD}`` cell.

    Args:
        results_dir: Directory with the result JSONs.
        out_dir: Report directory.
        stats: Dataset statistics JSON (default ``<results_dir>/dataset_stats.json``).

    Returns:
        Paths of the written ``.tex`` files (the tables, then ``macros.tex``).
    """
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    inputs = load_inputs(results_dir, stats)
    data = {
        "dataset": inputs.stats,
        "label_accuracy": inputs.label,
        "h1": inputs.h1,
        "h2": inputs.h2,
        "carried": inputs.carried,
        "play_fps": inputs.fps,
    }
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for spec in _table_specs(inputs.stats_file):
        text = spec.builder(data[spec.name]) if data[spec.name] is not None else None
        path = tables_dir / f"{spec.name}.tex"
        path.write_text(text if text is not None else tbd_table(spec.file, spec.issue))
        written.append(path)
    macros = tables_dir / "macros.tex"
    macros.write_text(build_macros(inputs))
    return [*written, macros]


def build_figures(results_dir: Path, out_dir: Path, stats: Path | None = None) -> list[Path]:
    """Write the PDF figures to ``out_dir/figures/``: dataset statistics, H1 scatter and per-bin curves, H2
    triptych, loss curves and the rollout strip.

    A figure whose input is missing is not written, and a stale one from an earlier run is removed, so the report
    shows its placeholder instead of out-of-date results.

    Args:
        results_dir: Directory with the result JSONs.
        out_dir: Report directory.
        stats: Dataset statistics JSON (default ``<results_dir>/dataset_stats.json``).

    Returns:
        Paths of the written figures.
    """
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    inputs = load_inputs(results_dir, stats)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    makers: dict[str, Callable[[Path], bool]] = {
        "dataset_stats": lambda p: inputs.stats is not None and fig_dataset_stats(inputs.stats, p),
        "h1_scatter": lambda p: inputs.h1 is not None and fig_h1_scatter(inputs.h1, results_dir / "h1", p),
        "h1_per_bin": lambda p: inputs.h1 is not None and fig_h1_per_bin(inputs.h1, p),
        "h2_triptych": lambda p: fig_h2_triptych(results_dir / "h2", p),
        "loss_curves": lambda p: inputs.carried is not None and fig_loss_curves(inputs.carried, p),
        "rollout_strip": lambda p: fig_rollout_strip(results_dir / "carried", p),
    }
    written = []
    for name in FIGURE_NAMES:
        path = fig_dir / f"{name}.pdf"
        if makers[name](path):
            written.append(path)
        elif path.exists():
            path.unlink()
            print(f"removed stale figures/{name}.pdf (missing {FIGURE_INPUTS[name]})")
    return written


def run(results_dir: Path, out_dir: Path, stats: Path | None = None) -> None:
    """CLI entry: ``build_tables`` then ``build_figures``; prints what was written and what is still TBD.

    Args:
        results_dir: Directory with the result JSONs.
        out_dir: Report directory.
        stats: Dataset statistics JSON (default ``<results_dir>/dataset_stats.json``).
    """
    inputs = load_inputs(results_dir, stats)
    if all(getattr(inputs, f) is None for f in ("stats", "label", "h1", "h2", "carried", "ups", "fps")):
        print(f"warning: no recognised result files in {results_dir}; every table is TBD", file=sys.stderr)
    tables = build_tables(results_dir, out_dir, stats)
    figures = build_figures(results_dir, out_dir, stats)
    tbd = [p.stem for p in tables if p.stem != "macros" and "\\textbf{TBD}" in p.read_text()]
    n_macros = sum(1 for line in tables[-1].read_text().splitlines() if line.startswith("\\newcommand"))
    print(f"wrote {len(tables)} tables ({len(tbd)} TBD: {', '.join(tbd) or 'none'}), {n_macros} macros")
    print(f"wrote {len(figures)}/{len(FIGURE_NAMES)} figures: {', '.join(p.stem for p in figures) or 'none'}")
