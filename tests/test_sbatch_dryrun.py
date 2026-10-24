"""Tests for scripts/mill/train.sbatch and benchmark.sbatch (#7): DRY_RUN output and real runs under bash.

The Mill and SLURM are not available here, so the scripts run against fake ``sbatch``/``scancel``/``squeue``/``rsync``/
``python`` executables on PATH, with temp dirs for ``S``, ``HOME`` and the node-local scratch root.
"""

import os
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")

JOB_ID = "777"
SUCCESSOR_ID = "4242"
TIMEOUT_S = 30

FAKE_SBATCH = f"""#!/bin/bash
echo "$*" >> "$FAKE_LOG/sbatch.log"
echo {SUCCESSOR_ID}
"""
FAKE_SCANCEL = """#!/bin/bash
echo "$*" >> "$FAKE_LOG/scancel.log"
"""
FAKE_SQUEUE = """#!/bin/bash
echo "04:00:00"
"""
FAKE_RSYNC = """#!/bin/bash
echo "$*" >> "$FAKE_LOG/rsync.log"
[ -n "$FAKE_RSYNC_SLEEP" ] && { touch "$FAKE_LOG/rsync.started"; exec sleep "$FAKE_RSYNC_SLEEP"; }
mkdir -p "${@: -1}" && cp -a "${@: -2:1}". "${@: -1}"
"""
# After the trainer started, `date` (called by the script's SIGUSR1 handler) can be made slow by creating $FAKE_LOG/slow_date,
# which keeps the handler running while the trainer exits: the window of the "interrupted wait" race.
FAKE_DATE = """#!/bin/bash
[ -e "$FAKE_LOG/slow_date" ] && sleep 0.4
exec /bin/date "$@"
"""
# FAKE_PY_MODE: done | exit0 (no DONE) | preempt | crash | wait_usr1 (blocks until SIGUSR1, then behaves like a preempted
# trainer) | timed_done / timed_exit3 (exit 0 with DONE / exit 3 without PREEMPTED, 0.2 s after $FAKE_LOG/go appears).
FAKE_PYTHON = """#!/bin/bash
echo "$*" >> "$FAKE_LOG/python.log"
for arg in "$@"; do
    case "$arg" in
        hydra.run.dir=*) RUN_DIR=${arg#hydra.run.dir=} ;;
        env.path_data_low_res=*) LOW_RES=${arg#env.path_data_low_res=} ;;
    esac
done
[ -d "$LOW_RES/train" ] && echo "staged_ok" >> "$FAKE_LOG/python.log"
mkdir -p "$RUN_DIR/checkpoints"
case "$FAKE_PY_MODE" in
    done) touch "$RUN_DIR/checkpoints/DONE"; exit 0 ;;
    exit0) exit 0 ;;
    timed_done|timed_exit3)
        trap '' USR1
        touch "$FAKE_LOG/python.ready"
        while [ ! -e "$FAKE_LOG/go" ]; do sleep 0.01; done
        sleep 0.2
        if [ "$FAKE_PY_MODE" = timed_done ]; then touch "$RUN_DIR/checkpoints/DONE"; exit 0; else exit 3; fi
        ;;
    preempt) touch "$RUN_DIR/checkpoints/PREEMPTED" "$RUN_DIR/checkpoints/state.pt"; exit 3 ;;
    crash) echo "Traceback: boom" >&2; exit 1 ;;
    wait_usr1)
        trap 'echo "got_usr1" >> "$FAKE_LOG/python.log"; touch "$RUN_DIR/checkpoints/PREEMPTED"; exit 3' USR1
        touch "$FAKE_LOG/python.ready"
        while :; do sleep 0.05; done
        ;;
esac
"""
FAKE_BENCH_PYTHON = """#!/bin/bash
echo "$*" >> "$FAKE_LOG/python.log"
[ "$FAKE_PY_MODE" = crash ] && { echo "Traceback: boom" >&2; exit 1; }
echo 'BENCH {"hw": "h100", "batch_size": 32}'
"""


def write_exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class Harness:
    """Temp dirs, fake executables and an environment for running a Mill script."""

    def __init__(self, tmp_path: Path, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.log = tmp_path / "log"
        self.bin = tmp_path / "bin"
        self.home = tmp_path / "home"
        self.s = tmp_path / "scratch" / "dcs2"
        self.local = tmp_path / "local"
        for d in (self.log, self.bin, self.home, self.s / "runs", self.s / "logs"):
            d.mkdir(parents=True)
        (self.s / "datasets" / "pilot" / "low_res" / "train").mkdir(parents=True)
        (self.s / "datasets" / "pilot" / "low_res" / "train" / "0.pt").write_text("x")
        (self.s / "datasets" / "pilot" / "low_res" / "test").mkdir()
        for name, text in {
            "sbatch": FAKE_SBATCH,
            "scancel": FAKE_SCANCEL,
            "squeue": FAKE_SQUEUE,
            "rsync": FAKE_RSYNC,
            "python": FAKE_PYTHON,
            "date": FAKE_DATE,
        }.items():
            write_exe(self.bin / name, text)
        (self.home / ".dcs2_env").write_text(
            f"export S={self.s}\nexport H100_GRES=gpu:h100:1\nexport V100_GRES=gpu:v100:1\n"
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}
        for k in (
            "DRY_RUN",
            "RESUBMITS_LEFT",
            "EXTRA",
            "GRES",
            "TIME_LIMIT",
            "RUN_NAME",
            "DATASET",
            "HW",
            "S",
            "FAKE_RSYNC_SLEEP",
        ):
            env.pop(k, None)
        env.update(
            PATH=f"{self.bin}{os.pathsep}{env['PATH']}",
            HOME=str(self.home),
            LOCAL_SCRATCH_ROOT=str(self.local),
            REPO_DIR=str(repo_root),
            FAKE_LOG=str(self.log),
            FAKE_PY_MODE="done",
            RUN_NAME="run1",
            DATASET="pilot",
            HW="h100",
        )
        self.env = env

    @property
    def run_dir(self) -> Path:
        return self.s / "runs" / "run1"

    @property
    def staged(self) -> Path:
        return self.local / os.environ.get("USER", "") / JOB_ID

    def logged(self, name: str) -> list[str]:
        path = self.log / name
        return path.read_text().splitlines() if path.is_file() else []

    def start(self, script: str = "train.sbatch", slurm: bool = True, **env) -> subprocess.Popen:
        full = {**self.env, **env}
        if slurm:
            full.update(SLURM_JOB_ID=JOB_ID, SLURM_JOB_NAME="run1")
        return subprocess.Popen(
            [BASH, str(self.repo_root / "scripts" / "mill" / script)],
            env=full,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

    def run(self, script: str = "train.sbatch", slurm: bool = True, **env) -> subprocess.CompletedProcess:
        proc = self.start(script, slurm, **env)
        out, err = proc.communicate(timeout=TIMEOUT_S)
        return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)

    def wait_for(self, name: str) -> None:
        deadline = time.monotonic() + TIMEOUT_S
        while not (self.log / name).exists():
            assert time.monotonic() < deadline, f"{name} never appeared"
            time.sleep(0.02)


@pytest.fixture
def h(tmp_path, repo_root) -> Harness:
    return Harness(tmp_path, repo_root)


def test_dry_run_prints_planned_commands(h: Harness) -> None:
    res = h.run(slurm=False, DRY_RUN="1", EXTRA="training.num_final_epochs=2 evaluation.every=1")
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "sbatch --parsable --dependency=afterany:dryrun" in out
    assert "--gres=gpu:h100:1" in out
    assert "--export=ALL,RESUBMITS_LEFT=2" in out
    assert f"rsync -a {h.s}/datasets/pilot/ " in out
    assert "python src/main.py" in out
    assert f"hydra.run.dir={h.s}/runs/run1" in out
    assert "common.resume=auto" in out
    assert "hw=h100" in out
    assert f"env.path_data_low_res={h.local}/" in out
    assert "/pilot/low_res" in out
    assert "training.num_final_epochs=2" in out and "evaluation.every=1" in out
    # nothing was executed
    assert not h.logged("sbatch.log") and not h.logged("rsync.log") and not h.logged("python.log")
    assert not h.local.exists()
    assert not h.run_dir.exists()


def test_dry_run_without_successor(h: Harness) -> None:
    res = h.run(slurm=False, DRY_RUN="1", RESUBMITS_LEFT="0")
    assert res.returncode == 0, res.stderr
    assert "successor: none" in res.stdout
    assert "sbatch" not in res.stdout.replace("successor", "")


def test_extra_overrides_are_not_globbed(h: Harness) -> None:
    res = h.run(
        slurm=False, DRY_RUN="1", EXTRA="a=[1,2] *"
    )  # `*` would expand to the repo's files if EXTRA were globbed
    assert res.returncode == 0, res.stderr
    (train,) = [line for line in res.stdout.splitlines() if "src/main.py" in line]
    assert train.endswith(" a=\\[1\\,2\\] \\*")


@pytest.mark.parametrize("missing", ["RUN_NAME", "DATASET", "HW"])
def test_missing_variable_is_reported(h: Harness, missing: str) -> None:
    h.env.pop(missing)
    res = h.run()
    assert res.returncode == 2
    assert f"{missing} is not set" in res.stderr
    assert not h.logged("sbatch.log")


def test_missing_env_file_is_reported(h: Harness) -> None:
    (h.home / ".dcs2_env").unlink()
    res = h.run()
    assert res.returncode == 2
    assert ".dcs2_env not found" in res.stderr


def test_missing_slurm_job_id_is_reported(h: Harness) -> None:
    res = h.run(slurm=False)
    assert res.returncode == 2
    assert "SLURM_JOB_ID" in res.stderr


def test_unknown_hw_needs_explicit_gres(h: Harness) -> None:
    res = h.run(HW="rtx3060")
    assert res.returncode == 2 and "GRES" in res.stderr
    assert h.run(HW="rtx3060", GRES="gpu:rtx:1").returncode == 0
    assert "--gres=gpu:rtx:1" in h.logged("sbatch.log")[0]


def test_done_submits_successor_then_cancels_it(h: Harness) -> None:
    res = h.run(EXTRA="training.num_final_epochs=2 evaluation.every=1")
    assert res.returncode == 0, res.stderr
    assert f"Successor job: {SUCCESSOR_ID}" in res.stdout
    (sbatch,) = h.logged("sbatch.log")
    for expected in (
        "--parsable",
        f"--dependency=afterany:{JOB_ID}",
        "-p gpu",
        "--gres=gpu:h100:1",
        "-J run1",
        "--export=ALL,RESUBMITS_LEFT=2",
        "--time=04:00:00",  # inherited from the running job's own limit
    ):
        assert expected in sbatch
    assert sbatch.endswith("scripts/mill/train.sbatch")
    assert h.logged("scancel.log") == [SUCCESSOR_ID]
    assert (h.run_dir / "checkpoints" / "DONE").is_file()


def test_python_command_and_staging(h: Harness) -> None:
    res = h.run(EXTRA="training.num_final_epochs=2 evaluation.every=1")
    assert res.returncode == 0, res.stderr
    cmd, staged_ok = h.logged("python.log")
    assert cmd == (
        f"src/main.py hydra.run.dir={h.run_dir} common.resume=auto hw=h100 "
        f"env.path_data_low_res={h.staged}/pilot/low_res training.num_final_epochs=2 evaluation.every=1"
    )
    assert staged_ok == "staged_ok"  # the dataset was already staged (with its train/ dir) when python started
    assert h.logged("rsync.log") == [f"-a {h.s}/datasets/pilot/ {h.staged}/pilot/"]
    assert not h.staged.exists()  # removed on exit
    assert not (h.local / os.environ.get("USER", "") / JOB_ID).exists()


def test_preempted_exit_keeps_successor(h: Harness) -> None:
    res = h.run(FAKE_PY_MODE="preempt")
    assert res.returncode == 3, res.stderr
    assert f"Successor job: {SUCCESSOR_ID}" in res.stdout
    assert h.logged("scancel.log") == []
    assert (h.run_dir / "checkpoints" / "PREEMPTED").is_file()
    assert not h.staged.exists()


def test_preempted_without_successor_warns(h: Harness) -> None:
    res = h.run(FAKE_PY_MODE="preempt", RESUBMITS_LEFT="0")
    assert res.returncode == 3
    assert "No successor is queued" in res.stderr


def test_crash_cancels_successor_with_clear_error(h: Harness) -> None:
    res = h.run(FAKE_PY_MODE="crash")
    assert res.returncode == 1
    assert "trainer crashed with exit code 1" in res.stderr
    assert "Successor cancelled" in res.stderr
    assert h.logged("scancel.log") == [SUCCESSOR_ID]
    assert not h.staged.exists()


def test_resubmits_left_zero_submits_nothing(h: Harness) -> None:
    res = h.run(RESUBMITS_LEFT="0")
    assert res.returncode == 0, res.stderr
    assert "no successor job" in res.stdout
    assert h.logged("sbatch.log") == []
    assert h.logged("scancel.log") == []


def test_resubmits_left_defaults_to_three_and_counts_down(h: Harness) -> None:
    assert h.run(FAKE_PY_MODE="exit0").returncode == 0  # (no DONE, so the second run is not a no-op)
    assert "RESUBMITS_LEFT=2" in h.logged("sbatch.log")[0]
    assert h.run(FAKE_PY_MODE="exit0", RESUBMITS_LEFT="1").returncode == 0
    assert "RESUBMITS_LEFT=0" in h.logged("sbatch.log")[1]


def test_invalid_resubmits_left_is_rejected(h: Harness) -> None:
    res = h.run(RESUBMITS_LEFT="-1")
    assert res.returncode == 2 and "RESUBMITS_LEFT" in res.stderr


def test_usr1_is_forwarded_to_python(h: Harness) -> None:
    proc = h.start(FAKE_PY_MODE="wait_usr1")
    try:
        h.wait_for("python.ready")
        assert h.staged.exists()  # staged for as long as the trainer runs
        os.kill(proc.pid, signal.SIGUSR1)  # SLURM's --signal=B:USR1@600 targets the batch shell
        out, err = proc.communicate(timeout=TIMEOUT_S)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 3, err
    assert "got_usr1" in h.logged("python.log")  # python received the forwarded plain SIGUSR1
    assert "forwarding to trainer" in out
    assert "trainer preempted (exit 3" in out
    assert h.logged("scancel.log") == []  # the successor stays queued
    assert not h.staged.exists()


def test_sigterm_keeps_successor_and_cleans_up(h: Harness) -> None:
    proc = h.start(FAKE_PY_MODE="wait_usr1")
    try:
        h.wait_for("python.ready")
        os.kill(proc.pid, signal.SIGTERM)  # e.g. the wall-time kill: the chain must still resume
        proc.communicate(timeout=TIMEOUT_S)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 143
    assert h.logged("scancel.log") == []
    assert not h.staged.exists()


def test_missing_dataset_fails_and_cancels_successor(h: Harness) -> None:
    res = h.run(DATASET="nope")
    assert res.returncode == 2
    assert "nope not found" in res.stderr
    assert h.logged("scancel.log") == [SUCCESSOR_ID]
    assert h.logged("python.log") == []


@pytest.mark.parametrize(
    ("mode", "code", "cancelled"), [("timed_done", 0, True), ("timed_exit3", 3, False)], ids=["exit0_done", "exit3"]
)
@pytest.mark.parametrize("attempt", range(3))
def test_trainer_exiting_during_usr1_handler_keeps_its_real_exit_code(
    h: Harness, mode: str, code: int, cancelled: bool, attempt: int
) -> None:
    """The trainer exits while the shell's USR1 trap runs, so the interrupted `wait` must not be mistaken for 138."""
    proc = h.start(FAKE_PY_MODE=mode)
    try:
        h.wait_for("python.ready")
        (h.log / "slow_date").touch()  # the USR1 handler now takes 0.4 s ...
        os.kill(proc.pid, signal.SIGUSR1)
        (h.log / "go").touch()  # ... and the trainer exits 0.2 s from now, i.e. while the handler is running
        out, err = proc.communicate(timeout=TIMEOUT_S)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == code, (out, err)
    assert "crashed" not in err
    assert h.logged("scancel.log") == ([SUCCESSOR_ID] if cancelled else [])


def test_sigterm_during_staging_keeps_successor(h: Harness) -> None:
    proc = h.start(FAKE_RSYNC_SLEEP="30")
    try:
        h.wait_for("rsync.started")
        start = time.monotonic()
        os.kill(proc.pid, signal.SIGTERM)
        proc.communicate(timeout=TIMEOUT_S)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 143
    assert time.monotonic() - start < 10  # the trap fired promptly instead of after the rsync
    assert h.logged("scancel.log") == []
    assert h.logged("python.log") == []
    assert not h.staged.exists()


def test_failed_staging_cancels_successor(h: Harness) -> None:
    write_exe(h.bin / "rsync", "#!/bin/bash\nexit 23\n")
    res = h.run()
    assert res.returncode == 2
    assert "staging failed (rsync exit 23)" in res.stderr
    assert h.logged("scancel.log") == [SUCCESSOR_ID]


def test_resubmits_left_with_leading_zero(h: Harness) -> None:
    assert h.run(RESUBMITS_LEFT="08").returncode == 0
    assert "RESUBMITS_LEFT=7" in h.logged("sbatch.log")[0]


def test_missing_python_is_reported_before_staging(h: Harness) -> None:
    (h.bin / "python").unlink()
    if shutil.which("python", path="/usr/bin:/bin"):
        pytest.skip("a system python is on the minimal PATH")
    res = h.run(PATH=f"{h.bin}{os.pathsep}/usr/bin{os.pathsep}/bin")
    assert res.returncode == 2
    assert "python not found" in res.stderr
    assert h.logged("rsync.log") == [] and h.logged("sbatch.log") == []


def test_existing_done_exits_without_staging_or_successor(h: Harness) -> None:
    (h.run_dir / "checkpoints").mkdir(parents=True)
    (h.run_dir / "checkpoints" / "DONE").touch()
    res = h.run()
    assert res.returncode == 0, res.stderr
    assert "already exists" in res.stdout
    assert h.logged("rsync.log") == [] and h.logged("python.log") == []
    assert h.logged("sbatch.log") == []  # nothing to chain, so no successor was submitted (and none to cancel)
    assert not h.staged.exists()


def test_crash_with_stale_preempted_file_is_still_a_crash(h: Harness) -> None:
    ckpt = h.run_dir / "checkpoints"
    ckpt.mkdir(parents=True)
    stale = ckpt / "PREEMPTED"
    stale.touch()
    old = time.time() - 3600
    os.utime(stale, (old, old))  # written by a previous job: older than this job's launch marker
    res = h.run(FAKE_PY_MODE="crash")
    assert res.returncode == 1
    assert "trainer crashed with exit code 1" in res.stderr
    assert h.logged("scancel.log") == [SUCCESSOR_ID]


def test_exit_zero_without_done_is_finished_with_warning(h: Harness) -> None:
    res = h.run(FAKE_PY_MODE="exit0")
    assert res.returncode == 0
    assert "DONE is missing" in res.stderr
    assert h.logged("scancel.log") == [SUCCESSOR_ID]


def test_benchmark_rejects_non_numeric_gpus(h: Harness, tmp_path: Path) -> None:
    res = h.run("benchmark.sbatch", DATA=str(tmp_path), GPUS="two")
    assert res.returncode == 2 and "GPUS must be" in res.stderr


def test_benchmark_dry_run(h: Harness) -> None:
    res = h.run(
        "benchmark.sbatch", slurm=False, DRY_RUN="1", HW="v100", DATA="/d/low_res", BATCH_SIZES="16 32", STEPS="50"
    )
    assert res.returncode == 0, res.stderr
    assert (
        "python scripts/benchmark_denoiser.py --hw v100 --steps 50 --batch-sizes 16 32 --data /d/low_res" in res.stdout
    )
    assert not h.logged("python.log")


def test_benchmark_runs_and_keeps_bench_lines(h: Harness, tmp_path: Path) -> None:
    write_exe(h.bin / "python", FAKE_BENCH_PYTHON)
    data = tmp_path / "data"
    data.mkdir()
    res = h.run("benchmark.sbatch", DATA=str(data), BATCH_SIZES="32 64")
    assert res.returncode == 0, res.stderr
    assert res.stdout.count('BENCH {"hw": "h100"') == 1
    assert h.logged("python.log") == [
        f"scripts/benchmark_denoiser.py --hw h100 --steps 200 --batch-sizes 32 64 --data {data}"
    ]


def test_benchmark_failure_propagates(h: Harness, tmp_path: Path) -> None:
    write_exe(h.bin / "python", FAKE_BENCH_PYTHON)
    res = h.run("benchmark.sbatch", DATA=str(tmp_path), FAKE_PY_MODE="crash")
    assert res.returncode == 1
    assert "exited with code 1" in res.stderr


def test_benchmark_requires_hw_and_data(h: Harness) -> None:
    h.env.pop("HW")
    res = h.run("benchmark.sbatch")
    assert res.returncode == 2
    assert "HW is not set" in res.stderr
    res = h.run("benchmark.sbatch", HW="h100")
    assert res.returncode == 2
    assert "DATA is not set" in res.stderr
