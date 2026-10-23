"""CPU tests for #6: hw profiles, autocast, metrics.jsonl, SIGUSR1-safe checkpointing, auto-resume, DONE, benchmark."""

import importlib.util
import json
import os
import signal
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from test_training_smoke import compose_cfg

from trainer import Trainer
from utils import autocast

STEPS_PER_EPOCH = 4
N_EPOCHS = 2

# Recorded by running the #5 code (commit bb61dfb, before any #6 change) on the same tiny denoiser-only config with
# torch.seed patched to 1234 (see `seeded_losses`): the denoiser/train/loss_denoising of each of the 4 steps of one epoch.
PRE_CHANGE_FP32_LOSSES = [0.4566754102706909, 0.5001824498176575, 0.2959883213043213, 0.7538094520568848]


@pytest.fixture(autouse=True)
def restore_sigusr1():
    """Trainer installs a SIGUSR1 handler in the test process; put the previous one back afterwards."""
    previous = signal.getsignal(signal.SIGUSR1)
    yield
    signal.signal(signal.SIGUSR1, previous)


def make_run_dir(root: Path, name: str, monkeypatch) -> Path:
    """A working dir shaped like a Hydra run dir (the Trainer moves .hydra/config.yaml on a fresh start)."""
    path = root / name
    (path / ".hydra").mkdir(parents=True, exist_ok=True)
    (path / ".hydra" / "config.yaml").write_text("{}\n")
    monkeypatch.chdir(path)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    return path


def make_cfg(repo_root, data_dir, tiny_model_dotlist, resume=False, evaluation_every=1):
    OmegaConf.register_new_resolver("eval", eval, replace=True)  # as in src/main.py
    cfg = compose_cfg(repo_root, data_dir, "denoiser_only", tiny_model_dotlist)
    cfg.denoiser.training.steps_first_epoch = STEPS_PER_EPOCH
    cfg.denoiser.training.steps_per_epoch = STEPS_PER_EPOCH
    cfg.training.num_final_epochs = N_EPOCHS
    cfg.evaluation.every = evaluation_every
    cfg.common.resume = resume
    return cfg


def preempt_after(trainer: Trainer, n_steps: int) -> None:
    """Send a real SIGUSR1 to this process during the n-th denoiser optimizer step."""
    opt = trainer.opt.denoiser
    original_step = opt.step
    calls = []

    def step(*args, **kwargs):
        calls.append(1)
        if len(calls) == n_steps:
            os.kill(os.getpid(), signal.SIGUSR1)
        return original_step(*args, **kwargs)

    opt.step = step


def optimizer_steps(trainer: Trainer) -> int:
    return int(next(iter(trainer.opt.denoiser.state.values()))["step"])


def read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("hw", ["h100", "v100", "rtx3060"])
def test_hw_profiles_keep_effective_batch_128(hw, repo_root) -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "config")):
        cfg = compose(config_name="trainer", overrides=[f"hw={hw}"])
    t = cfg.denoiser.training
    assert t.batch_size * t.grad_acc_steps == 128
    assert t.steps_per_epoch == 400
    assert cfg.training.precision == {"h100": "bf16", "v100": "fp16", "rtx3060": "fp32"}[hw]
    assert cfg.training.tf32 == (hw != "v100")
    assert "compile_wm" in cfg.training and cfg.training.num_workers_data_loaders > 0


def test_default_profile_is_rtx3060(repo_root) -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "config")):
        cfg = compose(config_name="trainer")
    assert (cfg.denoiser.training.batch_size, cfg.denoiser.training.grad_acc_steps) == (8, 16)


def test_fp32_autocast_is_a_noop_on_cpu() -> None:
    with autocast("fp32", "cpu"):
        assert not torch.is_autocast_enabled("cpu")
        assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.float32
    with pytest.raises(AssertionError):
        autocast("fp8", "cpu")


def test_preempt_and_resume_matches_uninterrupted_run(
    data, tmp_path, repo_root, tiny_model_dotlist, monkeypatch
) -> None:
    cfg = make_cfg(repo_root, data, tiny_model_dotlist)
    make_run_dir(tmp_path, "straight", monkeypatch)
    straight = Trainer(cfg, tmp_path)
    straight.run()

    run = make_run_dir(tmp_path, "preempted", monkeypatch)
    first = Trainer(cfg, tmp_path)
    preempt_after(first, STEPS_PER_EPOCH + 2)  # epoch 2, after 2 of its 4 steps
    with pytest.raises(SystemExit) as exc:
        first.run()
    assert exc.value.code == 3
    assert (run / "checkpoints" / "PREEMPTED").is_file()
    assert not (run / "checkpoints" / "DONE").exists()
    assert not list((run / "checkpoints").glob("*.tmp"))
    saved = torch.load(run / "checkpoints" / "state.pt", map_location="cpu", weights_only=False)
    assert saved["epoch"] == 2
    assert saved["epoch_progress"]["steps"] == {"denoiser": 2}
    assert len(read_metrics(run / "metrics.jsonl")) == 1  # epoch 1 only

    cfg.common.resume = "auto"
    resumed = Trainer(cfg, tmp_path)
    assert (resumed.epoch, resumed.epoch_progress["steps"]) == (2, {"denoiser": 2})
    assert not (run / "checkpoints" / "PREEMPTED").exists()
    resumed.run()

    for t in (straight, resumed):
        assert t.epoch == N_EPOCHS
        assert t.epoch_progress is None
        assert t.num_batch_train.denoiser == N_EPOCHS * STEPS_PER_EPOCH
        assert optimizer_steps(t) == N_EPOCHS * STEPS_PER_EPOCH
        assert t.lr_sched.denoiser.last_epoch == N_EPOCHS * STEPS_PER_EPOCH
    assert (run / "checkpoints" / "DONE").is_file()
    assert [m["epoch"] for m in read_metrics(run / "metrics.jsonl")] == [1, 2]


def test_preempt_on_last_step_of_the_run_resumes_to_done(
    data, tmp_path, repo_root, tiny_model_dotlist, monkeypatch
) -> None:
    """Preempted on the very last optimizer step: the resumed job only evaluates, then finishes with DONE."""
    cfg = make_cfg(repo_root, data, tiny_model_dotlist)
    run = make_run_dir(tmp_path, "last", monkeypatch)
    trainer = Trainer(cfg, tmp_path)
    preempt_after(trainer, N_EPOCHS * STEPS_PER_EPOCH)
    with pytest.raises(SystemExit) as exc:
        trainer.run()
    assert exc.value.code == 3
    trainer = Trainer(make_cfg(repo_root, data, tiny_model_dotlist, resume="auto"), tmp_path)
    trainer.run()
    assert (run / "checkpoints" / "DONE").is_file()
    assert trainer.num_batch_train.denoiser == N_EPOCHS * STEPS_PER_EPOCH
    assert [m["epoch"] for m in read_metrics(run / "metrics.jsonl")] == [1, 2]


def test_resume_auto_picks_fresh_or_resume(data, tmp_path, repo_root, tiny_model_dotlist, monkeypatch) -> None:
    cfg = make_cfg(repo_root, data, tiny_model_dotlist, resume="auto")
    run = make_run_dir(tmp_path, "auto", monkeypatch)
    (run / "checkpoints").mkdir()  # a resubmitted job's dir exists, but has no state.pt yet
    fresh = Trainer(cfg, tmp_path)
    assert fresh._resume is False
    assert (run / "checkpoints" / "state.pt").is_file()
    assert (run / "config" / "trainer.yaml").is_file()

    fresh.run()
    again = Trainer(cfg, tmp_path)  # state.pt now exists
    assert again._resume is True
    assert again.epoch == N_EPOCHS


def test_plain_fresh_run_still_refuses_an_existing_run_dir(
    data, tmp_path, repo_root, tiny_model_dotlist, monkeypatch
) -> None:
    cfg = make_cfg(repo_root, data, tiny_model_dotlist)
    run = make_run_dir(tmp_path, "strict", monkeypatch)
    Trainer(cfg, tmp_path)
    (run / ".hydra" / "config.yaml").write_text("{}\n")
    with pytest.raises(FileExistsError):
        Trainer(cfg, tmp_path)


def test_done_and_metrics_jsonl(data, tmp_path, repo_root, tiny_model_dotlist, monkeypatch) -> None:
    cfg = make_cfg(repo_root, data, tiny_model_dotlist, evaluation_every=2)
    run = make_run_dir(tmp_path, "metrics", monkeypatch)
    assert not (run / "checkpoints" / "DONE").exists()
    Trainer(cfg, tmp_path).run()
    assert (run / "checkpoints" / "DONE").is_file()

    rows = read_metrics(run / "metrics.jsonl")
    assert [r["epoch"] for r in rows] == [1, 2]
    for r in rows:
        assert set(r) == {"epoch", "train_loss", "test_loss", "lr", "duration_s", "samples_per_s", "gpu"}
        assert r["train_loss"] > 0 and r["lr"] > 0 and r["duration_s"] > 0
        assert r["gpu"] == "cpu"
        # 4 steps x batch 2 x grad_acc 1 per epoch
        assert r["samples_per_s"] == pytest.approx(STEPS_PER_EPOCH * 2 / r["duration_s"])
    assert rows[0]["test_loss"] is None and rows[1]["test_loss"] > 0


def test_main_exits_immediately_when_done_exists(
    data, tmp_path, repo_root, tiny_model_dotlist, monkeypatch, capsys
) -> None:
    OmegaConf.clear_resolver("eval")  # src/main.py registers it at import time without replace=True
    import main

    cfg = make_cfg(repo_root, data, tiny_model_dotlist, resume="auto")
    run = make_run_dir(tmp_path, "done", monkeypatch)
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "DONE").touch()

    def no_trainer(*args, **kwargs):
        raise AssertionError("Trainer must not be built when DONE exists")

    monkeypatch.setattr(main, "Trainer", no_trainer)
    main.run(cfg, tmp_path)  # returns normally, i.e. exit code 0
    assert "DONE exists" in capsys.readouterr().out


def test_fp32_losses_match_pre_change(data, tmp_path, repo_root, tiny_model_dotlist, monkeypatch) -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    monkeypatch.setattr(torch, "seed", lambda: 1234)  # Trainer seeds from torch.seed(); fix it
    make_run_dir(tmp_path, "fp32", monkeypatch)
    cfg = compose_cfg(repo_root, data, "denoiser_only", tiny_model_dotlist)
    cfg.denoiser.training.steps_first_epoch = STEPS_PER_EPOCH
    cfg.denoiser.training.steps_per_epoch = STEPS_PER_EPOCH
    assert cfg.training.precision == "fp32"
    trainer = Trainer(cfg, tmp_path)

    losses = []
    train_component = trainer.train_component

    def spy(name, steps):
        logs = train_component(name, steps)
        losses.extend(d["denoiser/train/loss_denoising"] for d in logs)
        return logs

    trainer.train_component = spy
    trainer.run()
    assert losses == pytest.approx(PRE_CHANGE_FP32_LOSSES, rel=1e-3)


@pytest.fixture(scope="module")
def benchmark_module(repo_root):
    spec = importlib.util.spec_from_file_location("benchmark_denoiser", repo_root / "scripts" / "benchmark_denoiser.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bench_args(data_dir: Path, *extra: str) -> list[str]:
    return [
        *("--hw", "rtx3060", "--steps", "3", "--warmup", "1", "--batch-sizes", "2", "4"),
        *("--data", str(data_dir / "low_res"), "--device", "cpu", "--tiny"),
        *("--override", "denoiser.training.num_autoregressive_steps=1"),
        *extra,
    ]


def bench_lines(capsys) -> list[dict]:
    return [
        json.loads(line[len("BENCH ") :]) for line in capsys.readouterr().out.splitlines() if line.startswith("BENCH ")
    ]


def test_benchmark_prints_bench_lines_on_cpu(benchmark_module, data, capsys) -> None:
    assert benchmark_module.main(bench_args(data)) == 0
    rows = bench_lines(capsys)
    assert [r["batch_size"] for r in rows] == [2, 4]
    for r in rows:
        assert r["hw"] == "rtx3060" and r["device"] == "cpu" and r["precision"] == "fp32" and r["oom"] is False
        assert r["ms_per_step"] > 0 and r["samples_per_s"] > 0 and r["peak_mem_gb"] is None
        assert r["est_hours_150_epochs"] == pytest.approx(
            150 * 400 * 128 * (r["ms_per_step"] / r["batch_size"]) / 1000 / 3600
        )


def test_benchmark_reports_oom_and_continues(benchmark_module, data, capsys, monkeypatch) -> None:
    real = benchmark_module.bench_one

    def flaky(cfg, args, device, precision, batch_size):
        if batch_size == 2:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(cfg, args, device, precision, batch_size)

    monkeypatch.setattr(benchmark_module, "bench_one", flaky)
    assert benchmark_module.main(bench_args(data)) == 0
    first, second = bench_lines(capsys)
    assert first["oom"] is True and first["ms_per_step"] is None and first["est_hours_150_epochs"] is None
    assert second["oom"] is False and second["ms_per_step"] > 0


def test_benchmark_script_is_executable(repo_root) -> None:
    path = repo_root / "scripts" / "benchmark_denoiser.py"
    assert path.read_text().startswith("#!/usr/bin/env python3")
    assert os.access(path, os.X_OK)
