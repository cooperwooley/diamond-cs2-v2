# Adapted from DIAMOND (Alonso et al., NeurIPS 2024), https://github.com/eloialonso/diamond (MIT License),
# via diamond-cs2 v1 (https://github.com/cooperwooley/diamond-cs2 @ a0b9e24). See ATTRIBUTION.md.
from functools import partial
from pathlib import Path
import random
import shutil
import signal
import sys
import threading
import time
from typing import List, Optional, Tuple

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb

from agent import Agent
from data import BatchSampler, collate_segments_to_batch, Dataset, DatasetTraverser, CS2Hdf5Dataset
from utils import (
    append_jsonl,
    autocast,
    broadcast_if_needed,
    build_ddp_wrapper,
    CommonTools,
    configure_opt,
    count_parameters,
    get_lr_sched,
    keep_agent_copies_every,
    Logs,
    move_opt_to,
    process_confusion_matrices_if_any_and_compute_classification_metrics,
    save_info_for_import_script,
    save_with_backup,
    set_seed,
    StateDictMixin,
    try_until_no_except,
    wandb_log,
)


def _ignore_sigusr1(worker_id: int) -> None:
    # DataLoader workers are forked after the trainer installed its handler; the flag it sets would be meaningless there.
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)


class Trainer(StateDictMixin):
    def __init__(self, cfg: DictConfig, root_dir: Path) -> None:
        OmegaConf.resolve(cfg)
        self._cfg = cfg
        # TF32 only when the hw profile asks for it.
        torch.backends.cuda.matmul.allow_tf32 = cfg.training.tf32
        torch.backends.cudnn.allow_tf32 = cfg.training.tf32
        self._precision = cfg.training.precision

        # SIGUSR1 (sent by SLURM shortly before the wall-time kill) only sets a flag; it is acted on at the next optimizer
        # step boundary. Handlers can only be installed from the main thread of the main process.
        self._preempt_requested = False
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGUSR1, self._on_sigusr1)
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        set_seed(torch.seed() % 10 ** 9)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu", self._rank)
        print(f"Starting on {self._device}")
        self._use_cuda = self._device.type == "cuda"
        if self._use_cuda:
            torch.cuda.set_device(self._rank)

        if self._rank == 0:
            try_until_no_except(
                partial(wandb.init, config=OmegaConf.to_container(cfg, resolve=True), reinit=True, resume=True, **cfg.wandb)
            )

        self._is_static_dataset = cfg.static_dataset.path is not None

        # Checkpointing
        self._path_ckpt_dir = Path("checkpoints")
        self._path_state_ckpt = self._path_ckpt_dir / "state.pt"
        self._path_preempted = self._path_ckpt_dir / "PREEMPTED"
        self._path_done = self._path_ckpt_dir / "DONE"
        self._keep_agent_copies = partial(
            keep_agent_copies_every,
            every=cfg.checkpointing.save_agent_every,
            path_ckpt_dir=self._path_ckpt_dir,
            num_to_keep=cfg.checkpointing.num_to_keep,
        )
        self._save_info_for_import_script = partial(
            save_info_for_import_script, run_name=cfg.wandb.name, path_ckpt_dir=self._path_ckpt_dir
        )

        # common.resume: True | False | "auto" (resume iff checkpoints/state.pt exists, start fresh otherwise).
        self._resume = self._path_state_ckpt.is_file() if cfg.common.resume == "auto" else cfg.common.resume
        # A resubmitted job's run dir may already exist (hydra.run.dir is fixed), so only a plain fresh run is strict.
        strict = cfg.common.resume is False
        if not self._resume and self._rank == 0:
            self._path_ckpt_dir.mkdir(exist_ok=not strict, parents=False)
            path_config = Path("config") / "trainer.yaml"
            path_config.parent.mkdir(exist_ok=not strict, parents=False)
            shutil.move(".hydra/config.yaml", path_config)
            wandb.save(str(path_config))

        # CS2 dataset setup
        assert cfg.env.train.id == "cs2", "This trainer is configured for CS2"
        # Full-res data is only needed to train the upsampler; denoiser-only training sets path_data_full_res to null.
        assert cfg.env.path_data_low_res is not None, "Set path_data_low_res in config/env/cs2.yaml"
        assert cfg.agent.upsampler is None or cfg.env.path_data_full_res is not None, \
            "Set path_data_full_res in config/env/cs2.yaml, or use agent=cs2_denoiser_only"
        assert self._is_static_dataset

        num_actions = cfg.env.num_actions
        # Only the upsampler trains on full-res frames; a denoiser-only agent ignores path_data_full_res.
        use_full_res = cfg.agent.upsampler is not None and cfg.env.path_data_full_res is not None
        dataset_full_res = CS2Hdf5Dataset(Path(cfg.env.path_data_full_res)) if use_full_res else None

        num_workers = cfg.training.num_workers_data_loaders
        use_manager = cfg.training.cache_in_ram and (num_workers > 0)
        p = Path(cfg.static_dataset.path) if self._is_static_dataset else Path("dataset")
        self.train_dataset = Dataset(p / "train", dataset_full_res, "train_dataset", cfg.training.cache_in_ram, use_manager)
        self.test_dataset = Dataset(p / "test", dataset_full_res, "test_dataset", cache_in_ram=True)
        self.train_dataset.load_from_default_path()
        self.test_dataset.load_from_default_path()

        # Create models
        self.agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(self._device)
        self._agent = build_ddp_wrapper(**self.agent._modules) if dist.is_initialized() else self.agent

        if cfg.training.compile_wm and self._use_cuda:
            for name in ("denoiser", "upsampler"):
                if getattr(self.agent, name) is not None:
                    getattr(self.agent, name).compile()  # in place, so state_dict keys are unchanged

        # fp16 needs loss scaling; bf16 and fp32 do not. A disabled scaler is a pass-through with an empty state.
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._precision == "fp16" and self._use_cuda)

        if cfg.initialization.path_to_ckpt is not None:
            self.agent.load(**cfg.initialization)

        # Optimizers and LR schedulers
        def build_opt(name: str) -> torch.optim.AdamW:
            return configure_opt(getattr(self.agent, name), **getattr(cfg, name).optimizer)

        def build_lr_sched(name: str) -> torch.optim.lr_scheduler.LambdaLR:
            return get_lr_sched(self.opt.get(name), getattr(cfg, name).training.lr_warmup_steps)

        model_names = ["denoiser", "upsampler"]
        self._model_names = [name for name in model_names if getattr(self.agent, name) is not None]

        self.opt = CommonTools(**{name: build_opt(name) for name in self._model_names})
        self.lr_sched = CommonTools(**{name: build_lr_sched(name) for name in self._model_names})

        # Data loaders
        make_data_loader = partial(
            DataLoader,
            dataset=self.train_dataset,
            collate_fn=collate_segments_to_batch,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=self._use_cuda,
            pin_memory_device=str(self._device) if self._use_cuda else "",
            worker_init_fn=_ignore_sigusr1,
        )

        make_batch_sampler = partial(BatchSampler, self.train_dataset, self._rank, self._world_size)

        c = cfg.denoiser.training
        seq_length = cfg.agent.denoiser.inner_model.num_steps_conditioning + 1 + c.num_autoregressive_steps
        bs = make_batch_sampler(c.batch_size, seq_length, None)
        dl_denoiser_train = make_data_loader(batch_sampler=bs)
        dl_denoiser_test = DatasetTraverser(self.test_dataset, c.batch_size, seq_length)

        if self.agent.upsampler is not None:
            c = cfg.upsampler.training
            seq_length = cfg.agent.upsampler.inner_model.num_steps_conditioning + 1 + c.num_autoregressive_steps
            bs = make_batch_sampler(c.batch_size, seq_length, None)
            dl_upsampler_train = make_data_loader(batch_sampler=bs)
            dl_upsampler_test = DatasetTraverser(self.test_dataset, c.batch_size, seq_length)
        else:
            dl_upsampler_train = dl_upsampler_test = None

        self._data_loader_train = CommonTools(dl_denoiser_train, dl_upsampler_train)
        self._data_loader_test = CommonTools(dl_denoiser_test, dl_upsampler_test)

        # Setup training
        sigma_distribution_cfg = instantiate(cfg.denoiser.sigma_distribution)
        sigma_distribution_cfg_upsampler = instantiate(cfg.upsampler.sigma_distribution) if self.agent.upsampler is not None else None
        self.agent.setup_training(sigma_distribution_cfg, sigma_distribution_cfg_upsampler)

        # Training state
        self.epoch = 0
        self.num_batch_train = CommonTools(0, 0, 0)
        self.num_batch_test = CommonTools(0, 0, 0)
        # None between epochs. While an epoch is in progress: optimizer steps done per component in this epoch, plus
        # the running sums behind its metrics.jsonl line, so a resumed epoch continues with only its remaining steps.
        self.epoch_progress = None

        if self._resume:
            self.load_state_checkpoint()
            self._path_preempted.unlink(missing_ok=True)
        else:
            self.save_checkpoint()

        if self._rank == 0:
            for name in self._model_names:
                print(f"{count_parameters(getattr(self.agent, name))} parameters in {name}")
            print(self.train_dataset)
            print(self.test_dataset)

    def run(self) -> None:
        to_log = []
        num_epochs = self._cfg.training.num_final_epochs

        # An epoch left in progress by a SIGUSR1 save is continued, not restarted.
        while self.epoch < num_epochs or self.epoch_progress is not None:
            if self.epoch_progress is None:
                self.epoch += 1
                self.epoch_progress = {"steps": {}, "loss_sum": 0.0, "loss_n": 0, "samples": 0, "elapsed": 0.0}
            self._epoch_start_time = time.time()

            if self._rank == 0:
                print(f"\nEpoch {self.epoch} / {num_epochs}\n")

            if self._cfg.training.should:
                to_log += self.train_agent()

            # Evaluation
            test_loss = None
            should_test = self._rank == 0 and self._cfg.evaluation.should and (self.epoch % self._cfg.evaluation.every == 0)
            if should_test:
                test_logs = self.test_agent()
                to_log += test_logs
                losses = [d["denoiser/test/loss_denoising"] for d in test_logs if "denoiser/test/loss_denoising" in d]
                test_loss = float(np.mean(losses)) if losses else None

            # Logging
            p = self.epoch_progress
            duration = p["elapsed"] + time.time() - self._epoch_start_time
            to_log.append({"duration": duration / 3600})
            if self._rank == 0:
                append_jsonl(
                    {
                        "epoch": self.epoch,
                        "train_loss": p["loss_sum"] / p["loss_n"] if p["loss_n"] > 0 else None,
                        "test_loss": test_loss,
                        "lr": self.lr_sched.denoiser.get_last_lr()[0],
                        "duration_s": duration,
                        "samples_per_s": p["samples"] / duration,
                        "gpu": torch.cuda.get_device_name(self._device) if self._use_cuda else "cpu",
                    },
                    Path("metrics.jsonl"),
                )
                wandb_log(to_log, self.epoch)
            to_log = []

            # Checkpointing
            self.epoch_progress = None
            self.save_checkpoint()

            if self._rank == 0 and self.epoch == num_epochs:
                self._path_done.touch()

            if dist.is_initialized():
                dist.barrier()

            if self._preempt_requested and self.epoch < num_epochs:
                self._exit_preempted()

        if self._rank == 0 and self.epoch >= num_epochs:
            self._path_done.touch()  # also when resuming a run that had already finished

    def train_agent(self) -> Logs:
        self.agent.train()
        self.agent.zero_grad()
        to_log = []
        for name in self._model_names:
            cfg = getattr(self._cfg, name).training
            if self.epoch > cfg.start_after_epochs:
                steps = cfg.steps_first_epoch if self.epoch == 1 else cfg.steps_per_epoch
                to_log += self.train_component(name, steps)
        return to_log

    @torch.no_grad()
    def test_agent(self) -> Logs:
        self.agent.eval()
        to_log = []
        for name in self._model_names:
            cfg = getattr(self._cfg, name).training
            if self.epoch > cfg.start_after_epochs:
                to_log += self.test_component(name)
        return to_log

    def train_component(self, name: str, steps: int) -> Logs:
        cfg = getattr(self._cfg, name).training
        model = getattr(self._agent, name)
        opt = self.opt.get(name)
        lr_sched = self.lr_sched.get(name)
        data_loader = self._data_loader_train.get(name)

        torch.cuda.empty_cache()
        model.to(self._device)
        move_opt_to(opt, self._device)

        model.train()
        opt.zero_grad()
        data_iterator = iter(data_loader) if data_loader is not None else None
        to_log = []

        progress = self.epoch_progress
        steps_done = progress["steps"].get(name, 0)
        num_steps = cfg.grad_acc_steps * (steps - steps_done)

        for i in trange(num_steps, desc=f"Training {name}", disable=self._rank > 0):
            batch = next(data_iterator).to(self._device) if data_iterator is not None else None
            with autocast(self._precision, self._device.type):
                loss, metrics = model(batch) if batch is not None else model()
            self.scaler.scale(loss).backward()

            if name == "denoiser":
                progress["loss_sum"] += metrics["loss_denoising"]
                progress["loss_n"] += 1
                progress["samples"] += cfg.batch_size

            num_batch = self.num_batch_train.get(name)
            metrics[f"num_batch_train_{name}"] = num_batch
            self.num_batch_train.set(name, num_batch + 1)

            if (i + 1) % cfg.grad_acc_steps == 0:
                if cfg.max_grad_norm is not None:
                    self.scaler.unscale_(opt)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm).item()
                    metrics["grad_norm_before_clip"] = grad_norm

                self.scaler.step(opt)
                self.scaler.update()
                opt.zero_grad()

                if lr_sched is not None:
                    metrics["lr"] = lr_sched.get_last_lr()[0]
                    lr_sched.step()

                progress["steps"][name] = steps_done + (i + 1) // cfg.grad_acc_steps
                if self._preempt_requested:
                    progress["elapsed"] += time.time() - self._epoch_start_time
                    self._save_state_and_exit()

            to_log.append(metrics)

        process_confusion_matrices_if_any_and_compute_classification_metrics(to_log)
        to_log = [{f"{name}/train/{k}": v for k, v in d.items()} for d in to_log]

        model.to("cpu")
        move_opt_to(opt, "cpu")

        return to_log

    @torch.no_grad()
    def test_component(self, name: str) -> Logs:
        model = getattr(self.agent, name)
        data_loader = self._data_loader_test.get(name)
        model.eval()
        model.to(self._device)
        to_log = []
        for batch in tqdm(data_loader, desc=f"Evaluating {name}"):
            batch = batch.to(self._device)
            with autocast(self._precision, self._device.type):
                _, metrics = model(batch)
            num_batch = self.num_batch_test.get(name)
            metrics[f"num_batch_test_{name}"] = num_batch
            self.num_batch_test.set(name, num_batch + 1)
            to_log.append(metrics)

        process_confusion_matrices_if_any_and_compute_classification_metrics(to_log)
        to_log = [{f"{name}/test/{k}": v for k, v in d.items()} for d in to_log]
        model.to("cpu")
        return to_log

    def _on_sigusr1(self, signum, frame) -> None:
        self._preempt_requested = True

    def _save_state_and_exit(self) -> None:
        """Full state checkpoint at an optimizer-step boundary, then exit with code 3 (preempted)."""
        if self._rank == 0:
            print("SIGUSR1 received: saving state and exiting")
            save_with_backup(self.state_dict(), self._path_state_ckpt)
        self._exit_preempted()

    def _exit_preempted(self) -> None:
        if self._rank == 0:
            self._path_preempted.touch()
        sys.exit(3)

    def state_dict(self):
        state = super().state_dict()
        state["rng_state"] = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if self._use_cuda else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        return state

    def load_state_dict(self, state_dict) -> None:
        state_dict = dict(state_dict)
        rng = state_dict.pop("rng_state")
        torch.set_rng_state(rng["torch"].cpu())
        if self._use_cuda and rng["cuda"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in rng["cuda"]])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        super().load_state_dict(state_dict)

    def load_state_checkpoint(self) -> None:
        # "cpu:0" (self._device on CPU) is not a valid map_location
        map_location = self._device if self._use_cuda else "cpu"
        self.load_state_dict(torch.load(self._path_state_ckpt, map_location=map_location, weights_only=False))

    def save_checkpoint(self) -> None:
        if self._rank == 0:
            save_with_backup(self.state_dict(), self._path_state_ckpt)
            self.train_dataset.save_to_default_path()
            self.test_dataset.save_to_default_path()
            self._keep_agent_copies(self.agent.state_dict(), self.epoch)
            self._save_info_for_import_script(self.epoch)
