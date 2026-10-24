"""Shared pytest fixtures."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(autouse=True)
def _isolated_data_root(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``paths.root`` at a temp dir, so CLI tests that load the default config never write into the checkout."""
    monkeypatch.setenv("DCS2_ROOT", str(tmp_path_factory.mktemp("dcs2-data")))
