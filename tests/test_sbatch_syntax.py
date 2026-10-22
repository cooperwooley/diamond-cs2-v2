"""bash -n on the Mill sbatch scripts."""

import shutil
import subprocess

import pytest

BASH = shutil.which("bash")


@pytest.mark.skipif(BASH is None, reason="bash not available")
@pytest.mark.parametrize("script", ["train.sbatch", "benchmark.sbatch", "env.example.sh"])
def test_bash_syntax(repo_root, script: str) -> None:
    path = repo_root / "scripts" / "mill" / script
    result = subprocess.run([BASH, "-n", str(path)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
