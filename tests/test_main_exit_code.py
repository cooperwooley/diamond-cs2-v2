"""`python src/main.py` must exit with the trainer's code 3 on SIGUSR1 (#6 contract that scripts/mill/train.sbatch relies on)."""

import os
import signal
import subprocess
import sys
import time

from test_training_smoke import STEPS

TIMEOUT_S = 240


def test_main_exits_3_on_sigusr1(data, tmp_path, repo_root, tiny_model_dotlist) -> None:
    run_dir = tmp_path / "run"
    overrides = [
        f"hydra.run.dir={run_dir}",
        "common.resume=auto",
        "agent=cs2_denoiser_only",
        f"env.path_data_low_res={data / 'low_res'}",
        "env.path_data_full_res=null",
        "common.devices=cpu",
        "wandb.mode=disabled",
        "training.num_workers_data_loaders=0",
        "training.num_final_epochs=1000",
        "evaluation.every=1000",
        "denoiser.training.num_autoregressive_steps=1",
        *[f"denoiser.training.{k}={v}" for k, v in STEPS.items()],
        *[f"agent.{x}" for x in tiny_model_dotlist(("denoiser",))],
    ]
    env = {**os.environ, "PYTHONPATH": str(repo_root), "WANDB_MODE": "disabled"}
    proc = subprocess.Popen(
        [sys.executable, str(repo_root / "src" / "main.py"), *overrides],
        env=env,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # metrics.jsonl appears after the first epoch, i.e. once the Trainer (and its SIGUSR1 handler) is running.
        deadline = time.monotonic() + TIMEOUT_S
        while not (run_dir / "metrics.jsonl").is_file():
            assert proc.poll() is None and time.monotonic() < deadline, "trainer exited or stalled before epoch 1"
            time.sleep(0.2)
        proc.send_signal(signal.SIGUSR1)
        out, _ = proc.communicate(timeout=TIMEOUT_S)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 3, out[-3000:]
    assert (run_dir / "checkpoints" / "PREEMPTED").is_file()
    assert (run_dir / "checkpoints" / "state.pt").is_file()
