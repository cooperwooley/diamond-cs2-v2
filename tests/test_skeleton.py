"""Skeleton checks: modules import, the CLI is wired, and config/cs2wm.yaml is self-consistent."""

import importlib
import pkgutil

import pytest
from omegaconf import OmegaConf

import cs2wm
from cs2wm import cli


def _module_names() -> list[str]:
    return [m.name for m in pkgutil.walk_packages(cs2wm.__path__, prefix="cs2wm.")]


@pytest.mark.parametrize("name", _module_names())
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_modules_found() -> None:
    assert "cs2wm.pipeline.pack" in _module_names()


def test_cli_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("argv", [["eval", "--help"], ["label-check", "--help"], ["play", "--help"]])
def test_subcommand_help(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 0


def test_unimplemented_command_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["pack"]) == 2
    assert "not implemented yet" in capsys.readouterr().err


@pytest.fixture(scope="module")
def cfg(repo_root):
    c = OmegaConf.load(repo_root / "config" / "cs2wm.yaml")
    OmegaConf.resolve(c)
    return c


def test_layout_matches_51_and_env_config(cfg, repo_root) -> None:
    m = cfg.actions.mouse
    total = len(cfg.actions["keys"]) + len(cfg.actions.clicks) + len(m.x_units) + len(m.y_units)
    assert total == 51
    assert total == OmegaConf.load(repo_root / "config" / "env" / "cs2.yaml").num_actions


@pytest.mark.parametrize("axis", ["x_units", "y_units"])
def test_mouse_bins_sorted_symmetric(cfg, axis: str) -> None:
    units = list(cfg.actions.mouse[axis])
    assert units == sorted(units)
    assert 0 in units
    assert units == [-u for u in reversed(units)]


def test_tickrate_multiple_of_fps(cfg) -> None:
    assert cfg.timing.tickrate % cfg.timing.fps == 0


def test_launch_options_insecure(cfg) -> None:
    assert "-insecure" in cfg.generate.launch_options
    assert "-insecure" in OmegaConf.to_container(cfg.generate.launch_options)
