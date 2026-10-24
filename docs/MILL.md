# Mill (Missouri S&T HPC)

<!-- TODO(#3): Mill env setup, GPU gres strings, scratch layout, SSH key; written from the #3 paste-back. -->

Everything here runs by hand over SSH. There is no automation from the dev machine to the Mill. Conventions:

- `S=/share/ceph/scratch/$USER/dcs2`, with `{v1,demos,hdf5,hdf5_fullres,qa,datasets,runs,logs}`. Scratch is not backed up.
- `~/.dcs2_env` (from `scripts/mill/env.example.sh`) defines `S`, `H100_GRES`, `V100_GRES` and activates the conda env
  `dcs2`. Every sbatch script sources it.
- Always submit to `-p gpu`, from the repo checkout (`cd ~/diamond-cs2-v2`). The batch scripts run in the submit directory.
- Job logs are `$S/logs/<jobname>-<jobid>.out` (the `-o` directive spells out the scratch root, because `#SBATCH` lines do
  not expand `$S`; the directory must exist, see #3).

## Training job lifecycle (`scripts/mill/train.sbatch`)

Every training job is the same command:

```bash
sbatch --gres=$H100_GRES -J <run> \
  --export=ALL,RUN_NAME=<run>,DATASET=<dataset>,HW=h100[,RESUBMITS_LEFT=3][,EXTRA="<hydra overrides>"] \
  scripts/mill/train.sbatch
```

| Variable | Meaning |
| --- | --- |
| `RUN_NAME` | Run directory `$S/runs/$RUN_NAME` (checkpoints, `metrics.jsonl`, Hydra outputs). |
| `DATASET` | `$S/datasets/$DATASET`, which holds `low_res/{train,test}` (#17). |
| `HW` | `config/hw` profile: `h100`, `v100` or `rtx3060`. |
| `RESUBMITS_LEFT` | Successors still allowed, default 3. A chain is at most `RESUBMITS_LEFT + 1` jobs. |
| `EXTRA` | Whitespace-separated Hydra overrides appended to the trainer command, e.g. `agent=cs2_denoiser_only`. |
| `GRES` | `--gres` for the successor. Default: `$H100_GRES` for `HW=h100`, `$V100_GRES` for `HW=v100`. Set it for other `HW` values or when the job used a non-default `--gres` (the successor must ask for the same hardware). |
| `TIME_LIMIT` | `--time` for the successor. Default: this job's own limit (read from `squeue`). |

`EXTRA` must not contain commas when it is given inside `--export=` (sbatch splits that list on commas). For Hydra list
overrides, export it in your shell instead and let `ALL` pass it on:
`export EXTRA='agent.denoiser.inner_model.depths=[1,1]'` then `--export=ALL,RUN_NAME=...`. The successor inherits the
whole environment (`--export=ALL`), so it sees the same `EXTRA`.

What a job does, in order:

1. **Checks** `~/.dcs2_env`, `S`, `RUN_NAME`, `DATASET`, `HW`, `RESUBMITS_LEFT` and prints clear errors.
2. **Submits its successor first**, if `RESUBMITS_LEFT > 0`: `sbatch --parsable --dependency=afterany:$SLURM_JOB_ID` with
   `RESUBMITS_LEFT-1` and the same job name and `--gres`. It prints `Successor job: <id>`. Submitting at the start means a
   hard kill (node failure, OOM-kill, wall-time overrun) still resumes.
3. **Stages** `$S/datasets/$DATASET/` to node-local NVMe, `/local/scratch/$USER/$SLURM_JOB_ID/$DATASET/`. Only the
   read-only dataset is staged. Checkpoints are written straight to `$S/runs/$RUN_NAME` on scratch.
4. **Trains** in the background: `python src/main.py hydra.run.dir=$S/runs/$RUN_NAME common.resume=auto hw=$HW
   env.path_data_low_res=<staged>/low_res $EXTRA`. `common.resume=auto` continues from `checkpoints/state.pt` if it
   exists and starts fresh otherwise.
5. **Forwards SIGUSR1.** `--signal=B:USR1@600` makes SLURM send SIGUSR1 to the batch shell 10 minutes before the 48 h limit
   (the `B:` matters: without it only job steps get the signal). The script traps it and runs `kill -USR1` on the Python
   process. The trainer saves a full state checkpoint at the next optimizer step, writes `checkpoints/PREEMPTED` and exits
   with code 3. The script waits in a loop, because `wait` returns early whenever a trapped signal arrives.
6. **Exit handling** and cleanup (the staged directory is always removed):

   | Trainer exit | Meaning | Successor | Job exit code |
   | --- | --- | --- | --- |
   | 0 (`checkpoints/DONE`) | training finished | `scancel`led | 0 |
   | 3 (`checkpoints/PREEMPTED`) | preempted, state saved | stays queued, resumes the run | 3 |
   | anything else | crash | `scancel`led, `ERROR: trainer crashed with exit code N` in the log | N |

   A job that is killed before it can react (wall-time, node failure) leaves the successor queued. A failure while staging
   or validating (including a missing `python`) also cancels the successor, so setup errors cannot loop.

A job that finds `checkpoints/DONE` already present (a successor that started after its predecessor finished) exits 0
immediately, before submitting a successor or staging anything. A job that receives SIGTERM (`scancel`, wall-time) stops
the trainer (waiting up to 30 s), removes its staged data and leaves the successor queued, also during staging.

### Local checks (no Mill needed)

```bash
bash -n scripts/mill/train.sbatch scripts/mill/benchmark.sbatch scripts/mill/env.example.sh
S=/tmp/dcs2 RUN_NAME=r DATASET=d HW=h100 H100_GRES=gpu:h100:1 DRY_RUN=1 bash scripts/mill/train.sbatch
```

`DRY_RUN=1` prints the planned successor `sbatch`, `rsync` and `python` commands and runs nothing.
`tests/test_sbatch_dryrun.py` runs the real script against fake `sbatch`, `scancel`, `rsync` and `python`.

### Monitor, cancel, resume

```bash
squeue -u $USER -o "%.10i %.12j %.8T %.10M %.20R"
tail -f $S/logs/<jobname>-<jobid>.out                      # Ctrl-C to stop following
tail -n 2 $S/runs/<run>/metrics.jsonl; ls $S/runs/<run>/checkpoints
sacct -u $USER -S today -o JobID,JobName%12,NodeList,Elapsed,State,ExitCode
```

- **Cancel a run for good:** cancel the chain's pending successor first, then the running job, otherwise the successor
  starts and resumes the run: `scancel <successor-id>; scancel <jobid>`. (`scancel -n <jobname>` cancels every job of
  the chain at once.)
- **Preempt on purpose** (save and hand over to the successor): `scancel --signal=USR1 --batch <jobid>`.
- **Resume after a crash or an exhausted chain:** fix the cause and submit the same command again. `common.resume=auto`
  continues from `checkpoints/state.pt`, so the run dir and `RUN_NAME` stay the same.
- **Switch hardware mid-run** (e.g. H100 to V100): cancel the chain and resubmit with the other `--gres` and `HW=`.
  The precision change is safe at an epoch boundary (#29).

## Benchmark (`scripts/mill/benchmark.sbatch`)

Environment: `HW`, `DATA` (a `low_res` directory), `BATCH_SIZES` (default `"32 64 128"`), `STEPS` (default 200), `GPUS`
(informational only: `benchmark_denoiser.py` is single-process). It runs `scripts/benchmark_denoiser.py` and the
`BENCH {json}` lines stay in the log; collect them with `grep -h '^BENCH ' $S/logs/bench-*.out`.

## Commands for the by-hand issues

All of these start with:

```bash
ssh mill
cd ~/diamond-cs2-v2 && git pull && git log -1 --oneline
conda activate dcs2 && source ~/.dcs2_env
```

### #8: 200-step H100 vs V100 benchmark

```bash
sbatch --gres=$H100_GRES -J bench-h100 \
  --export=ALL,HW=h100,DATA=$S/v1/data/low_res,BATCH_SIZES="32 64 128",STEPS=200 \
  scripts/mill/benchmark.sbatch
sbatch --gres=$V100_GRES -J bench-v100 \
  --export=ALL,HW=v100,DATA=$S/v1/data/low_res,BATCH_SIZES="16 32 64",STEPS=200 \
  scripts/mill/benchmark.sbatch
squeue -u $USER -o "%.10i %.12j %.8T %.10M %R"
# when both are done (each should take < 30 min):
grep -h '^BENCH ' $S/logs/bench-*.out
sacct -u $USER -S today -o JobID,JobName%12,NodeList,Submit,Start,Elapsed,State
```

Optional DDP scaling probe, only if the H100 queue is empty (note: the benchmark script itself is single-process):

```bash
sbatch --gres=gpu:<h100-type>:2 -J bench-h100x2 --export=ALL,HW=h100,GPUS=2,DATA=$S/v1/data/low_res,BATCH_SIZES="64",STEPS=200 scripts/mill/benchmark.sbatch
```

### #21: 2-epoch sanity train, resume and USR1 preemption test

```bash
# 1) build the pilot dataset (login node is fine: I/O only, ~minutes)
python scripts/hdf5_to_diamond.py --hdf5 $S/hdf5 --out $S/datasets/pilot --kinds dm --test-frac 0.1 --seed 0
python scripts/dataset_stats.py $S/datasets/pilot | tail -n 40

# 2) 2-epoch sanity run, no resubmits
sbatch --gres=$H100_GRES --time=04:00:00 -J sanity \
  --export=ALL,RUN_NAME=sanity-pilot,DATASET=pilot,HW=h100,RESUBMITS_LEFT=0,EXTRA="training.num_final_epochs=2 evaluation.every=1" \
  scripts/mill/train.sbatch
squeue -u $USER
tail -f $S/logs/sanity-<jobid>.out          # Ctrl-C to stop following
cat $S/runs/sanity-pilot/metrics.jsonl
ls -la $S/runs/sanity-pilot/checkpoints $S/runs/sanity-pilot/checkpoints/agent_versions

# 3) preemption test: 3 epochs, one allowed resubmit; send USR1 to the batch shell mid-epoch 2
sbatch --gres=$H100_GRES --time=04:00:00 -J preempt \
  --export=ALL,RUN_NAME=sanity-preempt,DATASET=pilot,HW=h100,RESUBMITS_LEFT=1,EXTRA="training.num_final_epochs=3" \
  scripts/mill/train.sbatch
# wait until metrics.jsonl has 1 line, then:
scancel --signal=USR1 --batch <jobid>
squeue -u $USER                              # successor should start after the first job exits
cat $S/runs/sanity-preempt/metrics.jsonl     # expect epochs 1,2,3 exactly once each
ls $S/runs/sanity-preempt/checkpoints        # expect DONE
sacct -u $USER -S today -o JobID,JobName%10,NodeList,Elapsed,State,ExitCode
```

### #29: main v2 training (150 epochs, H100)

```bash
sbatch --gres=$H100_GRES -J v2-main \
  --export=ALL,RUN_NAME=v2-main,DATASET=v2-full,HW=h100,RESUBMITS_LEFT=4,EXTRA="agent=cs2_denoiser_only" \
  scripts/mill/train.sbatch
squeue -u $USER -o "%.10i %.12j %.8T %.10M %.20R"
```

Daily check (~1 min):

```bash
ssh mill 'source ~/.dcs2_env; squeue -u $USER -o "%.10i %.12j %.8T %.10M %.20R"; tail -n 2 $S/runs/v2-main/metrics.jsonl; ls $S/runs/v2-main/checkpoints'
```

When `DONE` exists:

```bash
ssh mill 'source ~/.dcs2_env; mkdir -p ~/dcs2_ckpts; cp $S/runs/v2-main/checkpoints/agent_versions/agent_epoch_00150.pt ~/dcs2_ckpts/v2-main-e150.pt; sha256sum ~/dcs2_ckpts/v2-main-e150.pt'
# WSL on the PC:
mkdir -p /mnt/c/dcs2/data/checkpoints/v2
scp mill:dcs2_ckpts/v2-main-e150.pt /mnt/c/dcs2/data/checkpoints/v2/ && sha256sum /mnt/c/dcs2/data/checkpoints/v2/v2-main-e150.pt
```

Fallbacks:

- Still pending after 24 h, or the H100 is busy near finals: cancel the chain and resubmit on a V100 with
  `--gres=$V100_GRES` and `HW=v100`. It resumes from the same run dir.
- Time is short: stop at the last checkpoint at or after epoch 100 and report the epoch used.

### #30: shuffled-action control (H1 null model)

Run it concurrently with #29 on a second H100 if the queue allows, otherwise right after.

```bash
sbatch --gres=$H100_GRES -J v2-shuf \
  --export=ALL,RUN_NAME=v2-shuffled,DATASET=v2-full-shuffled,HW=h100,RESUBMITS_LEFT=4,EXTRA="agent=cs2_denoiser_only" \
  scripts/mill/train.sbatch
```

Use the same daily check and checkpoint copy as #29 with `v2-shuffled` (name the copy `v2-shuffled-e150.pt`). If compute
is tight, add `training.num_final_epochs=75` to `EXTRA` and evaluate the v2 epoch-75 checkpoint too, so both are compared at
75 epochs.
