"""Pytest plugin (loaded with ``-p tests.isolation`` from ``pyproject.toml``): keep tests out of the real data root."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_data_root(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``paths.root`` at a temp dir, so CLI tests that load the default config never write into the checkout."""
    monkeypatch.setenv("DCS2_ROOT", str(tmp_path_factory.mktemp("dcs2-data")))
