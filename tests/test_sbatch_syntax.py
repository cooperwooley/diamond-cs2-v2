"""bash -n on the Mill sbatch scripts."""

import os
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


@pytest.mark.parametrize("script", ["train.sbatch", "benchmark.sbatch", "env.example.sh"])
def test_scripts_are_executable_with_shebang(repo_root, script: str) -> None:
    path = repo_root / "scripts" / "mill" / script
    assert os.access(path, os.X_OK)
    assert path.read_text().startswith("#!") or script == "env.example.sh"  # env.example.sh is sourced, not run


@pytest.mark.parametrize("script", ["train.sbatch", "benchmark.sbatch"])
def test_sbatch_first_line_and_gpu_partition(repo_root, script: str) -> None:
    lines = (repo_root / "scripts" / "mill" / script).read_text().splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "#SBATCH -p gpu" in lines


def test_train_sbatch_directives(repo_root) -> None:
    text = (repo_root / "scripts" / "mill" / "train.sbatch").read_text()
    for directive in (
        "#SBATCH --time=2-00:00:00",
        "#SBATCH --signal=B:USR1@600",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --mem=128G",
        "#SBATCH -o /share/ceph/scratch/%u/dcs2/logs/%x-%j.out",
    ):
        assert directive in text.splitlines()


def test_env_example_defines_mill_variables(repo_root) -> None:
    text = (repo_root / "scripts" / "mill" / "env.example.sh").read_text()
    for needle in (
        "export S=/share/ceph/scratch/$USER/dcs2",
        "export H100_GRES=",
        "export V100_GRES=",
        "conda activate dcs2",
    ):
        assert needle in text
