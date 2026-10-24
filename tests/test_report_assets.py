"""Tests for the report asset generator (#36): tables, macros, figures, TBD placeholders, determinism, LaTeX smoke."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from cs2wm import cli
from cs2wm.eval import upsampler_check as uc
from cs2wm.report import latex

MACRO_NAMES = {
    "TBDhours",
    "TBDtrainhours",
    "TBDtesthours",
    "TBDnumepisodes",
    "TBDlabelmin",
    "TBDlabelxsign",
    "TBDestimatorR",
    "TBDdegperpx",
    "TBDhOneRvTwo",
    "TBDhOneRvOne",
    "TBDhOneRshuf",
    "TBDhOneSignVtwo",
    "TBDhTwoSmoke",
    "TBDhTwoStatic",
    "TBDfidVtwo",
    "TBDfidVone",
    "TBDfidShift",
    "TBDupsVtwo",
    "TBDupsVone",
    "TBDupsBicubic",
    "TBDfps",
    "TBDfpsgpu",
    "TBDhTwoCondition",
}
TABLES = ("dataset", "label_accuracy", "h1", "h2", "carried", "play_fps")


# --- fixtures matching the producers' schemas ------------------------------------------------------------------------


def _split_summary(episodes: int, frames: int, fps: int = 16) -> dict[str, Any]:
    rng = np.random.default_rng(episodes)
    x_keys = [f"{d:+.3f}" for d in np.linspace(-55, 55, 23)]
    y_keys = [f"{d:+.3f}" for d in np.linspace(-11, 11, 15)]
    return {
        "episodes": episodes,
        "frames": frames,
        "hours": frames / fps / 3600,
        "length_hist": {"edges": [0, 100], "counts": [episodes]},
        "label_rates": {k: float(rng.random() / 4) for k in ("w", "a", "s", "d", "space", "lmb", "rmb")},
        "mouse_x_hist": {k: int(rng.integers(0, 500)) for k in x_keys},
        "mouse_y_hist": {k: int(rng.integers(0, 500)) for k in y_keys},
        "all_zero_frac": 0.1,
        "idle_frac": 0.2,
    }


def make_stats() -> dict[str, Any]:
    """``scripts/dataset_stats.py --json`` schema (``compute_stats``)."""
    total = _split_summary(1200, 1_152_000)
    return {
        "fps": 16,
        "total": total,
        "train": _split_summary(1000, 960_000),
        "test": _split_summary(200, 192_000),
        "by_kind": {
            "smoke": {
                "train": {"episodes": 100, "frames": 96_000, "hours": 1.6667},
                "test": {"episodes": 50, "frames": 48_000, "hours": 0.8333},
            },
            "dm": {
                "train": {"episodes": 900, "frames": 864_000, "hours": 15.0},
                "test": {"episodes": 150, "frames": 144_000, "hours": 2.5},
            },
        },
    }


def make_label_accuracy() -> dict[str, Any]:
    """``score_review_sheet`` schema."""
    accs = {"wasd": (40, 38), "jump": (12, 12), "crouch": (10, 9), "fire": (20, 19), "reload": (5, 5)}
    groups = {
        g: {"n": n, "correct": c, "accuracy": c / n, "min_accuracy": 0.9, "low_n": n < 10, "pass": c / n >= 0.9}
        for g, (n, c) in accs.items()
    }
    groups["mouse_x_sign"] = {
        "n": 200,
        "correct": 199,
        "accuracy": 0.995,
        "min_accuracy": 0.98,
        "low_n": False,
        "pass": True,
    }
    groups["switch"] = {"n": 0, "correct": 0, "accuracy": None, "min_accuracy": 0.9, "low_n": True, "pass": False}
    return {"sheets": ["a.csv"], "groups": groups, "blank_rows": 0, "unscored": ["switch"], "pass": False}


H1_SCRIPTS = (
    ("yaw_p10", "yaw", 0.55, 0.0),
    ("yaw_m10", "yaw", -0.55, 0.0),
    ("noop", "noop", 0.0, 0.0),
    ("pitch_p4", "pitch", 0.0, 0.22),
)


def _h1_model(
    yaw: tuple[float, float, float], pitch: tuple[float, float, float], drift: float, lowres_shift: float = 0.1
) -> dict[str, Any]:
    """One model of ``results``: ``yaw`` and ``pitch`` are ``(pearson_r, slope, sign_accuracy)``; lowres differs."""

    def axis(v: tuple[float, float, float], shift: float = 0.0) -> dict[str, Any]:
        r, slope, sign = v
        return {
            "n": 100,
            "pearson_r": r - shift,
            "slope": slope - shift,
            "intercept": 0.0,
            "sign_accuracy": sign - shift,
            "n_nonzero": 90,
        }

    def res(shift: float) -> dict[str, Any]:
        per_bin = []
        for name, kind, cyaw, cpitch in H1_SCRIPTS:
            sy, sp = yaw[1] - shift, pitch[1] - shift
            per_bin.append(
                {
                    "script": name,
                    "kind": kind,
                    "commanded_yaw_deg": cyaw,
                    "commanded_pitch_deg": cpitch,
                    "n_contexts": 5,
                    "yaw": {"mean": sy * cyaw, "ci_lo": sy * cyaw - 0.05, "ci_hi": sy * cyaw + 0.05},
                    "pitch": {"mean": sp * cpitch, "ci_lo": sp * cpitch - 0.05, "ci_hi": sp * cpitch + 0.05},
                }
            )
        points = [
            {
                "script": name,
                "context": c,
                "commanded_yaw_deg": cyaw,
                "commanded_pitch_deg": cpitch,
                "est_yaw_deg": (yaw[1] - shift) * cyaw + 0.01 * c,
                "est_pitch_deg": None if kind == "yaw" else (pitch[1] - shift) * cpitch,
                "n_pairs": 28,
            }
            for name, kind, cyaw, cpitch in H1_SCRIPTS
            for c in range(3)
        ]
        return {
            "estimator": "phasecorr",
            "yaw": axis(yaw, shift),
            "pitch": axis(pitch, shift),
            "per_bin": per_bin,
            "noop_drift": {"mean_abs_dyaw_deg": {"mean": drift + 10 * shift, "ci_lo": 0.0, "ci_hi": 1.0}},
            "ws_divergence": {},
            "scatter": {"unit": "per (context, script) mean over the in-stats frame pairs", "points": points},
        }

    return {"fullres": res(0.0), "lowres": res(lowres_shift)}


def make_h1() -> dict[str, Any]:
    """``h1/summary.json`` schema (module docstring of h1_controllability). Yaw and pitch, full and low res all differ."""
    fit = {"n": 90, "r": 0.9312, "slope_px_per_deg": 2.6, "slope_deg_per_px": 0.38461, "passed": True}
    pitch_fit = dict(fit, r=0.5123, slope_deg_per_px=0.77777)
    return {
        "schema_version": 1,
        "config": {"estimator": "phasecorr"},
        "models": {"v2": {}, "v1": {}, "shuffled": {}},
        "estimator_validation": {
            "threshold_r": 0.8,
            "passed": True,
            "resolutions": {
                "150x280": {"available": True, "estimators": {"phasecorr": {"yaw": fit, "pitch": pitch_fit}}},
                "30x56": {
                    "available": True,
                    "estimators": {"phasecorr": {"yaw": dict(fit, r=0.1111), "pitch": dict(pitch_fit, r=0.2222)}},
                },
            },
        },
        "results": {
            "shuffled": _h1_model((-0.0123, 0.02, 0.51), (0.04, 0.03, 0.52), 0.01),
            "v1": _h1_model((0.1, 0.0, 0.5), (0.09, 0.01, 0.49), 0.02),
            "v2": _h1_model((0.8765, 0.904, 0.9731), (0.6123, 1.1, 0.8123), 0.0421),
        },
    }


def _h2_region(lpips: float, ssim: float) -> dict[str, Any]:
    return {
        k: {"mean": v, "ci_lo": v - 0.01, "ci_hi": v + 0.01, "n_contexts": 8}
        for k, v in (("lpips", lpips), ("ssim", ssim))
    }


def make_h2() -> dict[str, Any]:
    """``h2/summary.json`` schema."""

    def cond(smoke: tuple[float, float], static: tuple[float, float]) -> dict[str, Any]:
        return {"smoke": _h2_region(*smoke), "static": _h2_region(*static), "smoke_minus_static": {}}

    return {
        "schema_version": 1,
        "results": {
            "v2": {"single": cond((0.2, 0.65), (0.15, 0.75)), "rollout": cond((0.4, 0.5), (0.3, 0.6))},
            "v1": {"single": cond((0.25, 0.7), (0.18, 0.8)), "rollout": cond((0.38, 0.55), (0.35, 0.58))},
        },
    }


def make_carried() -> dict[str, Any]:
    """``carried/summary.json`` schema."""

    def row(model: str, dataset: str, source: str, fid: float, std: float | None) -> dict[str, Any]:
        label = model if source == "gt" or model != "v1" else "v1 (zero actions)"
        return {"label": label, "model": model, "dataset": dataset, "action_source": source, "fid": fid, "fid_std": std}

    return {
        "schema_version": 1,
        "loss_curve": {
            "x_axis": "epoch",
            "v1": {
                "test": {"epochs": [95, 120, 149], "denoiser": [0.0123, 0.0119, 0.0117], "upsampler": [0.1, 0.1, 0.1]}
            },
            "v2": {"epochs": [1, 2, 3], "train_loss": [0.5, 0.4, 0.3], "test_loss": [0.04, None, 0.02]},
        },
        "fid": {
            "protocol": {"deviations_from_v1": []},
            "rows": [
                row("v1", "v1_test", "zeros", 123.456, None),
                row("v1", "v1_test", "gt", 130.0, None),
                row("v2", "v2_test", "gt", 45.678, 1.234),
                row("v1", "v2_test", "zeros", 200.04, 2.5),
            ],
        },
    }


def make_upsampler() -> dict[str, Any]:
    """``upsampler_check.json``: built with the producer's own ``summarize`` and ``judge``."""

    def scores(up: float, bi: float) -> dict[str, dict[str, np.ndarray]]:
        return {m: {k: np.array([v, v * 1.1]) for k in uc.METRICS} for m, v in (("upsampler", up), ("bicubic", bi))}

    v2, v1 = uc.summarize(scores(0.1, 0.2)), uc.summarize(scores(0.09, 0.2))
    return {"v2": v2, "v1": v1, "verdict": uc.judge(v2, v1), "n_requested": 2}


def make_fps() -> dict[str, Any]:
    """``play_fps.json`` (``loop.benchmark``) schema."""
    results: dict[str, Any] = {}
    for kind, base in (("denoiser", 30.0), ("upsampler", 12.0)):
        for steps in (1, 3):
            for prec in ("fp32", "bf16"):
                name = f"{kind}_{steps}step_{prec}"
                if kind == "upsampler" and prec == "bf16" and steps == 3:
                    results[name] = {"status": "skipped", "reason": "bf16 autocast needs CUDA"}
                else:
                    fps = base / steps * (1.5 if prec == "bf16" else 1.0)
                    results[name] = {"status": "ok", "fps": fps, "ms_per_frame": 1000 / fps}
    return {"device": "cuda", "gpu": "NVIDIA GeForce RTX 3060", "n_frames": 100, "results": results}


def write_frames_csv(path: Path, slope: float) -> None:
    rows = ["context,episode,start,script,kind,frame,cmd_yaw_deg,cmd_pitch_deg,est_yaw_deg,est_pitch_deg,in_stats"]
    for i, (cmd, in_stats) in enumerate([(0.55, 1), (-0.55, 1), (0.0, 1), (0.55, 0)]):
        rows.append(f"0,e,0,s,yaw,{i + 4},{cmd},0.0,{slope * cmd},{'' if i == 2 else 0.01},{in_stats}")
    path.write_text("\n".join(rows) + "\n")


def write_png(path: Path, w: int = 64, h: int = 24) -> None:
    img = np.random.default_rng(0).integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """A complete results directory with every input."""
    r = tmp_path / "results"
    (r / "h1").mkdir(parents=True)
    (r / "h2").mkdir()
    (r / "carried").mkdir()
    files = {
        "dataset_stats.json": make_stats(),
        "label_accuracy.json": make_label_accuracy(),
        "h1/summary.json": make_h1(),
        "h2/summary.json": make_h2(),
        "carried/summary.json": make_carried(),
        "upsampler_check.json": make_upsampler(),
        "play_fps.json": make_fps(),
    }
    for name, data in files.items():
        (r / name).write_text(json.dumps(data, indent=2))
    for model, slope in (("v2", 0.9), ("v1", 0.0), ("shuffled", 0.05)):
        write_frames_csv(r / "h1" / f"{model}_frames.csv", slope)
    write_png(r / "h2" / "v2_triptychs.png")
    write_png(r / "carried" / "rollout_strip_00.png", 256, 64)
    return r


def read(out: Path, name: str) -> str:
    return (out / "tables" / f"{name}.tex").read_text()


def macros(out: Path) -> dict[str, str]:
    text = (out / "tables" / "macros.tex").read_text()
    return dict(re.findall(r"^\\newcommand\{\\(\w+)\}\{(.*)\\xspace\}$", text, flags=re.MULTILINE))


def rows(tex: str) -> list[str]:
    """Body rows (lines ending in ``\\\\``) of a tabular."""
    return [ln for ln in tex.splitlines() if ln.endswith("\\\\")]


def is_tabular_only(tex: str) -> bool:
    code = [ln for ln in tex.splitlines() if not ln.startswith("%")]
    return (
        code[0].startswith("\\begin{tabular}")
        and code[-1] == "\\end{tabular}"
        and not re.search(r"\\(begin\{table\}|caption|label)", tex)
    )


# --- tables ----------------------------------------------------------------------------------------------------------


def test_all_tables_are_booktabs_tabulars_only(results_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "report"
    paths = latex.build_tables(results_dir, out)
    assert [p.name for p in paths] == [*(f"{t}.tex" for t in TABLES), "macros.tex"]
    for t in TABLES:
        tex = read(out, t)
        assert is_tabular_only(tex), t
        assert "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex, t
        assert "TBD" not in tex, t


def test_dataset_table_rows_and_values() -> None:
    tex = latex.dataset_table(make_stats())
    assert tex is not None
    body = rows(tex)
    assert len(body) == 1 + 4 + 3  # header, 2 kinds x 2 splits, 3 totals
    assert body[1].startswith("\\texttt{dm} & train & 900 & 864{,}000 & 15.0")
    assert body[-1] == "Total & all & 1{,}200 & 1{,}152{,}000 & 20.0 \\\\"
    assert latex.dataset_table({"fps": 16}) is None


def test_label_accuracy_table(results_dir: Path, tmp_path: Path) -> None:
    tex = latex.label_accuracy_table(make_label_accuracy())
    assert tex is not None
    body = rows(tex)
    assert len(body) == 1 + 7
    assert "WASD & 40 & 38 & 95.0 & 90" in tex
    assert "mouse-$x$ sign & 200 & 199 & 99.5 & 98" in tex
    assert "weapon switch & 0 & 0 & -- & 90" in tex  # an unscored group is shown, not dropped


def h1_cells(tex: str) -> dict[str, list[str]]:
    """Model -> its eight cells."""
    return {ln.split(" & ")[0]: ln.removesuffix(" \\\\").split(" & ") for ln in rows(tex)[2:]}


def test_h1_table_exact_cells_and_null_controls_never_bold() -> None:
    tex = latex.h1_table(make_h1())
    assert tex is not None
    body = h1_cells(tex)
    assert list(body) == ["v2", "v1", "shuffled"]  # report order, not alphabetical
    # Yaw and pitch differ in the fixture, so a swapped key shows up here.
    assert body["v2"] == [
        "v2",
        *("\\textbf{0.88}", "\\textbf{0.90}", "\\textbf{97.3}"),
        *("\\textbf{0.61}", "\\textbf{1.10}", "\\textbf{81.2}"),
        "\\textbf{0.042}",
    ]
    assert body["v1"] == ["v1", "0.10", "0.00", "50.0", "0.09", "0.01", "49.0", "0.020"]
    assert body["shuffled"] == ["shuffled", "\\ensuremath{-}0.01", "0.02", "51.0", "0.04", "0.03", "52.0", "0.010"]
    assert "\\textbf" not in body["v1"] and "\\textbf" not in "".join(body["shuffled"])  # lower drift than v2
    assert "% Bold marks the best controllable model per column; null controls (v1, shuffled) are never bold." in tex
    assert "\\cmidrule" in tex and "\\multicolumn{3}{c}{Yaw}" in tex
    assert "no-op yaw" in tex and "drift ($^\\circ$/frame)" in tex


def test_h1_table_best_is_not_always_the_first_row() -> None:
    summary = make_h1()
    summary["results"]["alt"] = _h1_model((0.95, 0.7, 0.99), (0.9, 0.7, 0.7), 0.001)
    summary["results"]["v2_final"] = _h1_model((0.5, 0.95, 0.6), (0.99, 0.95, 0.6), 0.5)
    tex = latex.h1_table(summary)
    assert tex is not None
    body = h1_cells(tex)
    assert list(body) == ["v2", "v1", "shuffled", "alt", "v2\\_final"]
    bold = {m: [i for i, c in enumerate(cells) if "\\textbf" in c] for m, cells in body.items()}
    # columns: 1 yaw r, 2 yaw slope, 3 yaw sign, 4 pitch r, 5 pitch slope, 6 pitch sign, 7 drift
    assert bold["v2"] == [6]  # only pitch sign accuracy 81.2 is not beaten... by alt (0.7) or v2_final (0.9)? see below
    assert bold["alt"] == [1, 3, 7]  # yaw r, yaw sign, drift
    assert bold["v2\\_final"] == [2, 4, 5]  # slope 0.95 beats 1.10 and 0.70; pitch r 0.99; pitch slope 0.95
    assert "\\textbf{1.10}" not in tex and "\\textbf{0.70}" not in tex


def test_h1_slope_distance_ties_in_exact_decimal() -> None:
    summary = make_h1()
    summary["results"]["alt"] = _h1_model((0.1, 1.1, 0.1), (0.1, 1.0, 0.1), 0.5)
    summary["results"]["v2"] = _h1_model((0.1, 0.9, 0.1), (0.1, 1.0, 0.1), 0.5)
    body = h1_cells(latex.h1_table(summary))
    assert body["v2"][2] == "\\textbf{0.90}" and body["alt"][2] == "\\textbf{1.10}"  # |0.9-1| == |1.1-1|


def test_best_uses_exact_decimals() -> None:
    assert latex._best([0.9, 1.1, 0.7], 2, lambda v: abs(v - 1)) == {0, 1}
    assert latex._best([0.1 + 0.2, 0.3], 2, lambda v: v) == {0, 1}  # equal once displayed
    assert latex._best([0.5, None], 2, lambda v: v) == set()
    assert latex._best([0.5, None], 2, lambda v: v, min_count=1) == {0}


def test_h1_table_missing_keys_do_not_crash() -> None:
    summary = {"results": {"v2": {"fullres": {"yaw": {"pearson_r": 0.5}}}, "extra": {}}}
    tex = latex.h1_table(summary)
    assert tex is not None
    assert "0.50" in tex and "--" in tex
    assert latex.h1_table({"results": {}}) is None


def test_h2_table_blocks_and_bold() -> None:
    tex = latex.h2_table(make_h2())
    assert tex is not None
    assert tex.count("\\multicolumn{5}{l}") == 2
    single = tex.split("Autoregressive")[0]
    assert "\\multicolumn{2}{c}{LPIPS}" in tex
    # LPIPS and SSIM are won by different models.
    assert "v2 & \\textbf{0.200} & \\textbf{0.150} & 0.650 & 0.750" in single
    assert "v1 & 0.250 & 0.180 & \\textbf{0.700} & \\textbf{0.800}" in single
    assert latex.h2_table({"results": {"v2": {"single": {"smoke": {}}}}}) is not None  # sparse input is tolerated
    assert latex.h2_table({}) is None


def test_carried_table_bolds_lowest_fid_per_test_set() -> None:
    tex = latex.carried_table(make_carried())
    assert tex is not None
    body = rows(tex)[1:]
    assert len(body) == 4
    assert body[0] == "v1 & v1 & zero & \\textbf{123.5} \\\\"  # best of the two v1-test rows
    assert body[1] == "v1 & v1 & GT & 130.0 \\\\"
    assert body[2] == "v2 & v2 & GT & \\textbf{45.7} $\\pm$ 1.2 \\\\"  # v2 beats v1 on the v2 test set
    assert "\\textbf{200.0}" not in tex


def test_carried_table_single_row_per_set_is_not_bold() -> None:
    summary = {"fid": {"rows": [{"model": "v2", "dataset": "v2_test", "action_source": "gt", "fid": 10.0}]}}
    assert "\\textbf" not in (latex.carried_table(summary) or "")


def test_play_fps_table() -> None:
    tex = latex.play_fps_table(make_fps())
    assert tex is not None
    body = rows(tex)[1:]
    assert len(body) == 4
    assert body[0].startswith("Denoiser only, 1 step & \\textbf{30.0} & \\textbf{45.0}")
    assert body[-1] == "Denoiser + upsampler, 3 steps & 4.0 & -- \\\\"  # a skipped config
    assert "RTX 3060" in tex
    assert latex.play_fps_table({"results": {"junk": {}}}) is None


# --- macros ----------------------------------------------------------------------------------------------------------


def test_macros_defined_with_formatting(results_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "report"
    latex.build_tables(results_dir, out)
    m = macros(out)
    assert set(m) == MACRO_NAMES
    expected = {
        "TBDhours": "20.0",
        "TBDtrainhours": "16.7",
        "TBDtesthours": "3.3",
        "TBDnumepisodes": "1{,}200",
        "TBDlabelmin": "90.0",
        "TBDlabelxsign": "99.5",
        "TBDestimatorR": "0.93",
        "TBDdegperpx": "0.385$^\\circ$/px",
        "TBDhOneRvTwo": "0.88",
        "TBDhOneRvOne": "0.10",
        "TBDhOneRshuf": "\\ensuremath{-}0.01",  # a negative r must also work inside $r=...$
        "TBDhOneSignVtwo": "97.3\\%",
        "TBDhTwoSmoke": "0.200",
        "TBDhTwoStatic": "0.150",
        "TBDfidVtwo": "45.7",
        "TBDfidVone": "123.5",  # the zero-action v1 row, not the GT-action one
        "TBDfidShift": "200.0",
        "TBDupsVtwo": "0.105",
        "TBDupsVone": "0.095",
        "TBDupsBicubic": "0.210",
        "TBDfps": "12.0",
        "TBDfpsgpu": "NVIDIA GeForce RTX 3060",
        "TBDhTwoCondition": "single-step",
    }
    assert m == expected
    text = (out / "tables" / "macros.tex").read_text()
    assert all(f"\\{n}: " in text for n in expected)  # the header names each macro's source
    assert "\\TBDwasdlag: " in text and "Not computable" in text


def test_missing_input_gives_tbd_table_and_no_macros(results_dir: Path, tmp_path: Path) -> None:
    (results_dir / "h1" / "summary.json").unlink()
    (results_dir / "upsampler_check.json").write_text("{not json")  # unreadable counts as missing
    out = tmp_path / "report"
    latex.build_tables(results_dir, out)
    tex = read(out, "h1")
    assert is_tabular_only(tex)
    assert "\\textbf{TBD} \\\\" in tex and tex.count("&") == 0
    assert "% TBD: results/h1/summary.json is missing or unusable (#32)" in tex
    assert "TBD" not in read(out, "h2")
    m = macros(out)
    assert (
        not {n for n in m if n.startswith(("TBDhOne", "TBDests", "TBDdegperpx", "TBDups"))} and "TBDestimatorR" not in m
    )
    assert {"TBDfidVtwo", "TBDhours"} <= set(m)


def test_empty_results_dir_never_crashes(tmp_path: Path) -> None:
    out = tmp_path / "report"
    latex.run(tmp_path / "nothing", out)
    for t in TABLES:
        tex = read(out, t)
        assert "\\textbf{TBD}" in tex and is_tabular_only(tex)
    assert macros(out) == {}
    assert not list((out / "figures").glob("*.pdf"))


def test_stats_override(results_dir: Path, tmp_path: Path) -> None:
    alt = tmp_path / "elsewhere.json"
    alt.write_text(json.dumps(make_stats() | {"total": _split_summary(10, 16 * 3600)}))
    (results_dir / "dataset_stats.json").unlink()
    out = tmp_path / "report"
    latex.build_tables(results_dir, out, stats=alt)
    assert macros(out)["TBDhours"] == "1.0"
    latex.build_tables(results_dir, out)
    assert "% TBD: results/dataset_stats.json" in read(out, "dataset")


def test_number_formatting() -> None:
    assert latex._num(-0.001, 2) == "0.00"
    assert latex._num(-0.5, 2) == "\\ensuremath{-}0.50"
    assert latex._num(float("nan"), 2) == "--"
    assert latex._num(None, 2) == "--"
    assert latex._int(1234567) == "1{,}234{,}567"
    assert latex._pct(0.9731) == "97.3"


# --- review fixes -----------------------------------------------------------------------------------------------------


def test_tex_escape_covers_every_special_character() -> None:
    assert latex._tex_escape("v2_final") == "v2\\_final"
    assert latex._tex_escape("a\\b_c#d%e&f$g{h}i~j^k") == (
        "a\\textbackslash{}b\\_c\\#d\\%e\\&f\\$g\\{h\\}i\\textasciitilde{}j\\textasciicircum{}k"
    )
    assert latex._tex_escape("two\nlines") == "two lines"  # never breaks out of a % comment line


def test_free_text_cells_are_escaped_in_every_table() -> None:
    odd = "v2_final#1"
    h1 = make_h1()
    h1["results"][odd] = h1["results"]["v2"]
    h2 = make_h2()
    h2["results"][odd] = h2["results"]["v2"]
    carried = make_carried()
    carried["fid"]["rows"].append({"model": odd, "dataset": "x_test", "action_source": "a_b", "fid": 1.0})
    stats = make_stats()
    stats["by_kind"]["we_ird"] = stats["by_kind"]["dm"]
    acc = make_label_accuracy()
    acc["groups"]["my_group"] = acc["groups"]["wasd"]
    fps = make_fps()
    fps["gpu"] = "RTX_3060 #1"
    for tex in (
        latex.h1_table(h1),
        latex.h2_table(h2),
        latex.carried_table(carried),
        latex.dataset_table(stats),
        latex.label_accuracy_table(acc),
        latex.play_fps_table(fps),
    ):
        assert tex is not None
        body = "\n".join(ln for ln in tex.splitlines() if not ln.startswith("%"))
        assert not re.search(r"(?<!\\)[_#]", body)  # no unescaped _ or # outside math
    assert "v2\\_final\\#1" in latex.h1_table(h1)
    assert "x\\_test" in latex.carried_table(carried)
    assert "my\\_group" in latex.label_accuracy_table(acc)
    assert "RTX\\_3060 \\#1" in latex.play_fps_table(fps)


def test_macro_values_are_escaped_and_negative_r_is_math_safe(results_dir: Path, tmp_path: Path) -> None:
    fps = make_fps() | {"gpu": "RTX_3060"}
    (results_dir / "play_fps.json").write_text(json.dumps(fps))
    out = tmp_path / "report"
    latex.build_tables(results_dir, out)
    m = macros(out)
    assert m["TBDfpsgpu"] == "RTX\\_3060"
    assert m["TBDhOneRshuf"] == "\\ensuremath{-}0.01"


def test_headline_macros_and_notes(results_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "report"
    latex.build_tables(results_dir, out)
    m = macros(out)
    assert m["TBDfpsgpu"] == "NVIDIA GeForce RTX 3060"
    assert m["TBDhTwoCondition"] == "single-step"
    text = (out / "tables" / "macros.tex").read_text()
    assert "FELL BACK" not in text
    assert "TBDhTwoSmoke, TBDhTwoStatic: the 'single' condition" in text
    # fp32 headline did not run: fall back to bf16 and say so.
    fps = make_fps()
    fps["results"]["upsampler_1step_fp32"] = {"status": "skipped", "reason": "x"}
    (results_dir / "play_fps.json").write_text(json.dumps(fps))
    latex.build_tables(results_dir, out)
    assert macros(out)["TBDfps"] == "18.0"
    assert "TBDfps: FELL BACK to upsampler_1step_bf16" in (out / "tables" / "macros.tex").read_text()
    # no gpu (CPU run): the macro is left undefined.
    (results_dir / "play_fps.json").write_text(json.dumps(make_fps() | {"gpu": None}))
    latex.build_tables(results_dir, out)
    assert "TBDfpsgpu" not in macros(out)


def test_labelmin_ignores_groups_without_judged_rows() -> None:
    acc = make_label_accuracy()
    acc["groups"]["switch"] = {"n": 0, "correct": 0, "accuracy": 0.0, "min_accuracy": 0.9}  # malformed but n == 0
    assert latex._label_min(acc) == 0.9  # crouch 9/10, not the unjudged 0.0
    assert latex._label_min({"groups": {}}) is None


def test_model_colour_is_stable_across_subsets() -> None:
    assert latex._model_color("v2") == latex.COLORS[0]
    assert latex._model_color("v1") == latex.COLORS[1]
    assert latex._model_color("shuffled") == latex.COLORS[2]
    assert latex._model_color("extra") == latex._model_color("extra") not in latex.COLORS[:3]


def test_scatter_prefers_summary_points_and_falls_back_to_csv(results_dir: Path, tmp_path: Path) -> None:
    h1 = json.loads((results_dir / "h1" / "summary.json").read_text())
    pts = latex._scatter_from_summary(h1, "v2")
    assert pts is not None and len(pts["cmd_yaw_deg"]) == 12
    assert np.isnan(pts["est_pitch_deg"][0]) and not np.isnan(pts["est_pitch_deg"][-1])  # null est stays NaN
    assert latex._scatter_from_summary(h1, "nope") is None
    for csv_file in (results_dir / "h1").glob("*_frames.csv"):
        csv_file.unlink()
    assert latex.fig_h1_scatter(h1, results_dir / "h1", tmp_path / "from_json.pdf")  # no CSVs needed
    for m in h1["results"].values():
        del m["fullres"]["scatter"]
    assert not latex.fig_h1_scatter(h1, results_dir / "h1", tmp_path / "none.pdf")
    write_frames_csv(results_dir / "h1" / "v2_frames.csv", 0.9)
    assert latex.fig_h1_scatter(h1, results_dir / "h1", tmp_path / "from_csv.pdf")
    pts = latex._read_frames_csv(results_dir / "h1" / "v2_frames.csv")
    assert pts is not None and len(pts["cmd_yaw_deg"]) == 3  # the in_stats == 0 row is dropped


# --- figures ---------------------------------------------------------------------------------------------------------


def test_figures_written(results_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "report"
    paths = latex.build_figures(results_dir, out)
    assert [p.name for p in paths] == [f"{n}.pdf" for n in latex.FIGURE_NAMES]
    for p in paths:
        assert p.read_bytes().startswith(b"%PDF") and p.stat().st_size > 1000, p


def test_figure_skipped_when_input_missing_and_stale_removed(results_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "report"
    latex.build_figures(results_dir, out)
    (results_dir / "carried" / "rollout_strip_00.png").unlink()
    (results_dir / "h1" / "summary.json").unlink()
    names = {p.name for p in latex.build_figures(results_dir, out)}
    assert names == {"dataset_stats.pdf", "h2_triptych.pdf", "loss_curves.pdf"}
    assert not (
        out / "figures" / "rollout_strip.pdf"
    ).exists()  # stale figure removed, the report shows its placeholder


def test_stale_figure_removal_is_announced_and_empty_results_warn(
    results_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "report"
    latex.build_figures(results_dir, out)
    capsys.readouterr()
    (results_dir / "carried" / "rollout_strip_00.png").unlink()
    latex.build_figures(results_dir, out)
    assert "removed stale figures/rollout_strip.pdf (missing carried/rollout_strip_NN.png)" in capsys.readouterr().out
    latex.build_figures(results_dir, out)
    assert "removed stale" not in capsys.readouterr().out  # nothing left to remove
    latex.run(tmp_path / "nothing", tmp_path / "report2")
    assert "no recognised result files in" in capsys.readouterr().err


def test_h2_triptych_prefers_producer_pdf(results_dir: Path, tmp_path: Path) -> None:
    (results_dir / "h2" / "v2_triptychs.pdf").write_bytes(b"%PDF-1.4 producer")
    out = tmp_path / "report"
    latex.build_figures(results_dir, out)
    assert (out / "figures" / "h2_triptych.pdf").read_bytes() == b"%PDF-1.4 producer"


def test_loss_curve_v1_alone(results_dir: Path, tmp_path: Path) -> None:
    carried = make_carried()
    carried["loss_curve"]["v2"] = None
    assert latex.fig_loss_curves(carried, tmp_path / "l.pdf")
    assert not latex.fig_loss_curves({"loss_curve": {"v1": None, "v2": None}}, tmp_path / "m.pdf")


# --- determinism -----------------------------------------------------------------------------------------------------


def test_two_runs_give_identical_tex(results_dir: Path, tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    latex.run(results_dir, a)
    latex.run(results_dir, b)
    texs = sorted((a / "tables").glob("*.tex"))
    assert [p.name for p in texs] == sorted(f"{n}.tex" for n in (*TABLES, "macros"))
    for p in texs:
        assert p.read_bytes() == (b / "tables" / p.name).read_bytes(), p.name


def test_two_runs_give_identical_pdfs(results_dir: Path, tmp_path: Path) -> None:
    """Best effort: matplotlib's PDF backend is deterministic once the dates are fixed."""
    a, b = tmp_path / "a", tmp_path / "b"
    latex.build_figures(results_dir, a)
    latex.build_figures(results_dir, b)
    same = [
        p.name for p in sorted((a / "figures").glob("*.pdf")) if p.read_bytes() == (b / "figures" / p.name).read_bytes()
    ]
    assert same == sorted(p.name for p in (a / "figures").glob("*.pdf"))


# --- CLI -------------------------------------------------------------------------------------------------------------


def test_cli_report_assets(results_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "report"
    cli.main(["report-assets", "--results", str(results_dir), "--out", str(out)])
    assert (out / "tables" / "h1.tex").exists() and (out / "figures" / "loss_curves.pdf").exists()
    assert "wrote 7 tables (0 TBD: none)" in capsys.readouterr().out
    args = cli.build_parser().parse_args(["report-assets", "--stats", "s.json"])
    assert args.stats == Path("s.json")


# --- LaTeX smoke -----------------------------------------------------------------------------------------------------


def _report_sources(repo_root: Path, dest: Path) -> Path | None:
    """Copy the report draft (branch ``feat/37-report``, else the working tree's ``report/``) into ``dest``."""
    git = subprocess.run(
        ["git", "-C", str(repo_root), "archive", "feat/37-report", "report"], capture_output=True, check=False
    )
    if git.returncode == 0:
        subprocess.run(["tar", "-x", "-C", str(dest)], input=git.stdout, check=True)
    elif (repo_root / "report" / "placeholders.tex").exists():
        shutil.copytree(repo_root / "report", dest / "report")
    else:
        return None
    return dest / "report"


@pytest.mark.skipif(shutil.which("latexmk") is None or shutil.which("pdflatex") is None, reason="needs latexmk")
@pytest.mark.parametrize("complete", [True, False], ids=["full", "empty"])
def test_report_compiles(complete: bool, results_dir: Path, tmp_path: Path, repo_root: Path) -> None:
    report = _report_sources(repo_root, tmp_path)
    if report is None:
        pytest.skip("no report draft (feat/37-report) to compile")
    if complete:  # a model name with an underscore, and the fixture's negative shuffled r (in the abstract's math)
        h1 = json.loads((results_dir / "h1" / "summary.json").read_text())
        h1["results"]["v2_final"] = h1["results"]["v2"]
        (results_dir / "h1" / "summary.json").write_text(json.dumps(h1))
        h2 = make_h2()
        h2["results"]["v2_final"] = h2["results"]["v2"]
        (results_dir / "h2" / "summary.json").write_text(json.dumps(h2))
        carried = make_carried()
        carried["fid"]["rows"].append({"model": "v2_final", "dataset": "v2_test", "action_source": "gt", "fid": 50.0})
        (results_dir / "carried" / "summary.json").write_text(json.dumps(carried))
    latex.run(results_dir if complete else tmp_path / "nothing", report)
    proc = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=report,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    log = (report / "main.log").read_text(errors="replace") if (report / "main.log").exists() else proc.stdout
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert "Undefined control sequence" not in log
    assert (report / "main.pdf").exists()
    if complete and shutil.which("pdftotext"):
        text = subprocess.run(
            ["pdftotext", str(report / "main.pdf"), "-"], capture_output=True, text=True, check=False
        ).stdout
        assert re.search(r"r\s*=\s*[\u2212-]\s*0\.01", text)  # the negative r is typeset in math mode
