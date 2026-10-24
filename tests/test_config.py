"""Tests for cs2wm.config."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ReadonlyConfigError

from cs2wm.actions import ActionSpec
from cs2wm.config import DEFAULT_CONFIG_PATH, ConfigError, config_sha256, load_config, ticks_per_frame, validate_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_default_loads_and_is_readonly(cfg) -> None:
    assert cfg.timing.fps == 16
    assert ticks_per_frame(cfg) == 4
    with pytest.raises(ReadonlyConfigError):
        cfg.timing.fps = 8


def test_overrides_apply() -> None:
    assert ticks_per_frame(load_config(overrides=["timing.fps=32"])) == 2


def test_interpolations_resolved(cfg) -> None:
    assert cfg.paths.demos == f"{cfg.paths.root}/demos"


def _bad(tmp_path: Path, overrides: list[str]) -> str:
    """Load the default config with bad overrides and return the ConfigError message."""
    with pytest.raises(ConfigError) as exc:
        load_config(DEFAULT_CONFIG_PATH, overrides)
    return str(exc.value)


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        (["timing.fps=24"], "multiple of timing.fps"),
        (["timing.fps=0"], "must be > 0"),
        (["resolution.render=[150]"], "resolution.render"),
        (["resolution.low_res=[30,-56]"], "resolution.low_res"),
        (["resolution.capture=[720.5,1280]"], "resolution.capture"),
        (["episodes.min_frames=2000"], "min_frames"),
        (["episodes.max_frames=0"], "must be > 0"),
        (["actions.keys=[w,a]"], "actions.keys must have 11"),
        (["actions.clicks=[lmb]"], "actions.clicks must have 2"),
        (["actions.mouse.x_units=[-10,0,5,10]"], "symmetric"),
        (["actions.mouse.y_units=[-10,-10,0,10]"], "strictly increasing"),
        (["actions.mouse.y_units=[-10,-5,5,10]"], "contain 0"),
        (["actions.mouse.x_units=[-10,0,10]"], "total 51"),
        (["actions.mouse.deg_per_unit=0"], "deg_per_unit"),
        (["generate.launch_options=[-novid]"], "-insecure"),
        (["qa.min_fire_recall=1.5"], "qa.min_fire_recall"),
        (["qa.max_missing_tick_frac=-0.1"], "qa.max_missing_tick_frac"),
        (["qa.max_fill_ticks=-1"], "qa.max_fill_ticks"),
    ],
)
def test_validation_rules(tmp_path: Path, overrides: list[str], needle: str) -> None:
    assert needle in _bad(tmp_path, overrides)


def test_missing_default_slot_fails(tmp_path: Path) -> None:
    raw = OmegaConf.load(DEFAULT_CONFIG_PATH)
    del raw.actions.slots["default"]
    path = tmp_path / "cfg.yaml"
    OmegaConf.save(raw, path)
    with pytest.raises(ConfigError, match=r"actions\.slots\.default"):
        load_config(path)


def test_all_problems_reported_together() -> None:
    msg = _bad(Path("."), ["timing.fps=24", "generate.launch_options=[-novid]", "episodes.min_frames=5000"])
    assert "multiple of timing.fps" in msg
    assert "-insecure" in msg
    assert "min_frames" in msg


def test_validate_config_accepts_default(cfg) -> None:
    validate_config(cfg)


def test_sha256_stable_and_sensitive(cfg) -> None:
    h = config_sha256(cfg)
    assert len(h) == 64
    assert h == config_sha256(load_config())
    assert h != config_sha256(load_config(overrides=["actions.mouse.deg_per_unit=0.056"]))


def test_sha256_ignores_paths_and_supervisor(cfg) -> None:
    other = load_config(
        overrides=["paths.root=/elsewhere", "supervisor.ntfy_topic=secret", "supervisor.max_attempts=9"]
    )
    assert config_sha256(other) == config_sha256(cfg)


def test_num_actions_matches_env_config(cfg, repo_root) -> None:
    env = OmegaConf.load(repo_root / "config" / "env" / "cs2.yaml")
    assert env.num_actions == ActionSpec.from_config(cfg).dim
