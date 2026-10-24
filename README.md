# diamond-cs2-v2

An action-conditioned, playable [DIAMOND](https://github.com/eloialonso/diamond) diffusion world model for
Counter-Strike 2. CS 5404 final project.

<!-- TBD(#35): demo GIF, recorded play session with the input overlay. -->
![Demo: playing the v2 world model](docs/media/demo.gif)

[v1](https://github.com/cooperwooley/diamond-cs2) trained DIAMOND on about 5.7 h of YouTube footage in which every
action label was all-zero, so the model could not be controlled. v2 fixes that by generating its own paired data:

1. Offline CS2 bot deathmatches are recorded as demos.
2. HLAE renders them at a fixed 16 fps without the HUD.
3. A 51-dim action per frame is derived from player state (not from button logs).

The denoiser is retrained on this data and the v1 upsampler is reused unchanged. What is new in v2:

- the generate, segment, render, extract, pack and sync pipeline, with an unattended overnight supervisor;
- the shared 51-dim action spec used by both label extraction and live play;
- Mill (SLURM) training with auto-resume, plus a shuffled-action control model;
- evals for controllability (H1), volumetric smoke (H2) and the v1 comparisons;
- a local play loop with an FPS benchmark.

## Results at a glance

TBD(#32): H1 table, see `report/tables/h1.tex` once #32 has run (`cs2wm eval h1`, then `cs2wm report-assets`).
Nothing is reported here until the numbers exist.

## WARNING: offline only

> **HLAE and CS2 only ever run offline with `-insecure`. Never connect them to a VAC-secured server.**
> The launchers in `cs2wm` refuse to start CS2 without `-insecure`; do not work around that.

## Setup

### WSL / Linux (development, tests, eval, pack/extract)

```bash
# WSL or Linux
git clone https://github.com/cooperwooley/diamond-cs2-v2.git && cd diamond-cs2-v2
conda env create -f environment.yml        # env "dcs2", Python 3.10
conda activate dcs2
pip install -e ".[dev]"
cs2wm --help
```

`environment.yml` installs `requirements.txt`, which pulls whatever `torch` pip resolves. For a CUDA build, install
torch first from the matching index, for example
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`, then re-run the steps above.
Check with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

### Windows (generate, render, supervise, play)

The stages that drive CS2/HLAE, and `play`, run under Windows Python 3.10 from a Windows-side clone (running it on
`\\wsl$` paths is slow). Update it with `git pull` only.

```powershell
# Windows PowerShell
winget install -e --id Python.Python.3.10
mkdir C:\dcs2 -Force; cd C:\dcs2
git clone https://github.com/cooperwooley/diamond-cs2-v2.git
py -3.10 -m venv C:\dcs2\venv
C:\dcs2\venv\Scripts\python -m pip install -U pip
C:\dcs2\venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
C:\dcs2\venv\Scripts\pip install -r C:\dcs2\diamond-cs2-v2\requirements.txt
C:\dcs2\venv\Scripts\pip install -e C:\dcs2\diamond-cs2-v2
C:\dcs2\venv\Scripts\python -c "import cs2wm, demoparser2, h5py, cv2, torch; print(torch.cuda.is_available())"
```

Environment variables (`config/cs2wm.yaml` interpolates them). Open a new PowerShell afterwards:

```powershell
# Windows PowerShell
setx DCS2_ROOT "C:\dcs2\data"                 # working set (<= 10 GB) and the demo archive
setx DCS2_HLAE_DIR "C:\hlae"
setx DCS2_CS2_DIR "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive"
setx DCS2_NTFY_TOPIC "dcs2-<pick-a-long-random-string>"   # phone notifications; never commit it
```

- **HLAE:** download the latest release (2.192.8 or newer) from
  [advancedfx/advancedfx](https://github.com/advancedfx/advancedfx/releases) and extract it to `C:\hlae`.
- **CS2:** in Steam, set Properties, Updates to "Only update this game when I launch it" (a patch can break HLAE).
  Launch it once normally, then fix the video settings: windowed 1280x720, fixed quality presets, boost player
  contrast off.
- **Sensitivity:** set `play.sensitivity` in `config/cs2wm.yaml` to your CS2 sensitivity (default 2.5).
- **ntfy:** install the ntfy phone app and subscribe to your topic.

### Mill (Missouri S&T HPC, training and dataset assembly)

Everything on the Mill is run by hand over SSH. Details and the job lifecycle: [docs/MILL.md](docs/MILL.md).

```bash
# WSL: key-based SSH so `cs2wm sync` can run unattended (Host alias "mill")
ssh-keygen -t ed25519 -f ~/.ssh/mill_dcs2 -C "dcs2-sync" -N ""
# add a "Host mill" block (HostName mill.mst.edu, User <sso-id>, IdentityFile ~/.ssh/mill_dcs2) to ~/.ssh/config
ssh-copy-id -i ~/.ssh/mill_dcs2.pub mill
ssh -o BatchMode=yes mill hostname          # must print a login node with no prompt
```

```bash
# Mill login node (no compute here)
git clone https://github.com/cooperwooley/diamond-cs2-v2.git ~/diamond-cs2-v2 && cd ~/diamond-cs2-v2
conda env create -f environment.yml && conda activate dcs2 && pip install -e . && conda clean -a -y
mkdir -p /share/ceph/scratch/$USER/dcs2/{v1,demos,hdf5,hdf5_fullres,qa,datasets,runs,logs}
cp scripts/mill/env.example.sh ~/.dcs2_env  # edit H100_GRES / V100_GRES from `sinfo -p gpu -N -o "%N %G"`
echo 'source ~/.dcs2_env' >> ~/.bashrc && source ~/.dcs2_env
```

`~/.dcs2_env` defines `S=/share/ceph/scratch/$USER/dcs2`, `H100_GRES` and `V100_GRES`. Always submit with `-p gpu`.
Scratch is not backed up.

## Pipeline commands

Stages: `generate -> segment -> render -> extract -> pack -> sync`. Details and the frame/tick convention:
[docs/PIPELINE.md](docs/PIPELINE.md). Stage commands that accept `--all-pending` also take `--retry-failed`.

### One demo end to end (about 1.5 h of footage)

```powershell
# Windows PowerShell (CS2 and HLAE are launched with -insecure; leave the PC alone while generating)
cd C:\dcs2\diamond-cs2-v2; git pull
C:\dcs2\venv\Scripts\Activate.ps1
cs2wm generate --profile dm --hours 1.5 --chunk-min 10
cs2wm segment --all-pending
cs2wm render --all-pending
cs2wm extract --all-pending
cs2wm pack --all-pending
cs2wm qa-summary                  # per-kind pass rate and passing hours
cs2wm sync                        # rsync to the Mill, verify sha256, free local disk
cs2wm status                      # every demo should be "synced"
```

Useful extras: `cs2wm inspect-demo <demo.dem>`, `cs2wm render --cfg-only --plan <render_plan.json>` (writes the
cfg and prints the HLAE command without launching), `cs2wm cleanup --failed --dry-run`, and the label hand-check:

```powershell
# Windows PowerShell or WSL
cs2wm label-check render --episode <episode.hdf5> --full-res-frames <frames dir or full-res hdf5> --out clip.mp4
cs2wm label-check score clip_labels.csv --out results/label_accuracy.json   # sheets are pooled
```

Verify the data on the Mill:

```bash
# Mill
python scripts/verify_manifests.py $S/hdf5 --fullres-root $S/hdf5_fullres
```

### Unattended overnight run

The supervisor loops generate and the other stages with retries, disk and CS2-patch guards and ntfy alerts, until the
deadline or the target hours of QA-passing data are synced. Keep the PC logged in, CS2 closed and Steam running.

```powershell
# Windows PowerShell: run once by hand
cs2wm supervise --until 07:30 --target-hours 20 --profile dm
# Windows PowerShell: install the scheduled task "dcs2-supervisor" (nightly)
powershell -ExecutionPolicy Bypass -File scripts\windows\install_task.ps1 -Start 23:00 -Until 07:30 -TargetHours 20 -Profile dm
schtasks /Query /TN dcs2-supervisor /V /FO LIST | findstr /i "Status Next Last"
# every morning
cs2wm status
cs2wm qa-summary --since 24h
# stop and disable
schtasks /End /TN dcs2-supervisor
schtasks /Change /TN dcs2-supervisor /DISABLE
```

Do not run `cs2wm cleanup` while `supervise` is running.

## Dataset assembly and training on the Mill

Build the main and the shuffled-action control together, from the same HDF5 snapshot and seed (the demo split depends
on the full set of demos). Dataset layout and datasheet: [docs/DATASET.md](docs/DATASET.md).

```bash
# Mill login node (I/O only)
cd ~/diamond-cs2-v2 && git pull && conda activate dcs2 && source ~/.dcs2_env
python scripts/dataset_stats.py --hdf5 $S/hdf5 --summary          # running total of hours
python scripts/verify_manifests.py $S/hdf5 --fullres-root $S/hdf5_fullres
python scripts/hdf5_to_diamond.py --hdf5 $S/hdf5 --out $S/datasets/v2-full --kinds dm,human,smoke --test-frac 0.1 --seed 0
python scripts/hdf5_to_diamond.py --hdf5 $S/hdf5 --out $S/datasets/v2-full-shuffled --kinds dm,human,smoke --test-frac 0.1 --seed 0 --shuffle-actions
python scripts/dataset_stats.py $S/datasets/v2-full --json $S/datasets/v2-full/stats.json
```

Training is one command per job (`scripts/mill/train.sbatch`); jobs resubmit themselves, save on SIGUSR1 ten minutes
before the wall limit, and resume with `common.resume=auto`.

```bash
# Mill: main model (150 epochs, one H100, chain of up to 5 x 48 h jobs)
sbatch --gres=$H100_GRES -J v2-main \
  --export=ALL,RUN_NAME=v2-main,DATASET=v2-full,HW=h100,RESUBMITS_LEFT=4,EXTRA="agent=cs2_denoiser_only" \
  scripts/mill/train.sbatch

# Mill: shuffled-action control (H1 null model)
sbatch --gres=$H100_GRES -J v2-shuf \
  --export=ALL,RUN_NAME=v2-shuffled,DATASET=v2-full-shuffled,HW=h100,RESUBMITS_LEFT=4,EXTRA="agent=cs2_denoiser_only" \
  scripts/mill/train.sbatch

# Mill: monitor
squeue -u $USER -o "%.10i %.12j %.8T %.10M %.20R"
tail -n 2 $S/runs/v2-main/metrics.jsonl; ls $S/runs/v2-main/checkpoints     # DONE marks a finished run
```

`HW` is `h100`, `v100` or `rtx3060` (`config/hw`). Preview a job without submitting: `DRY_RUN=1 bash scripts/mill/train.sbatch`
(with `S`, `RUN_NAME`, `DATASET`, `HW` set). Cancel, resume and V100 fallback: [docs/MILL.md](docs/MILL.md).

## Checkpoints

TBD(#29): released as a GitHub release asset, with sha256 values. Checkpoints are never committed to git.
Until then, copy them from the Mill (`$S/runs/<run>/checkpoints/agent_versions/agent_epoch_00150.pt`) to the PC:

```bash
# WSL
mkdir -p /mnt/c/dcs2/data/checkpoints/v2
scp mill:dcs2_ckpts/v2-main-e150.pt /mnt/c/dcs2/data/checkpoints/v2/ && sha256sum /mnt/c/dcs2/data/checkpoints/v2/v2-main-e150.pt
```

The v1 checkpoint (`agent_epoch_00149.pt`, the source of the upsampler) goes in `$DCS2_ROOT/checkpoints/v1/`.

## Eval commands

Run in WSL on the GPU box. `<v1>` is the v1 checkpoint, `<v2>` and `<shuf>` are the v2 and shuffled checkpoints,
`<test>` is the `low_res/test` directory of the assembled dataset (fetched from the Mill), and `<split.json>` comes
with the dataset.

```bash
# WSL: upsampler compatibility check (exit 0 = PASS, 1 = FAIL, 3 = INCOMPLETE without --v1-fullres)
cs2wm eval upsampler --ckpt <v1> --fullres $DCS2_ROOT/hdf5_fullres --v1-fullres ~/v1_test_fullres --n 256 --out results --device cuda

# WSL: H1 controllability (v2 vs v1 zero-action vs shuffled control)
cs2wm eval h1 --models v2=<v2> v1=<v1> shuffled=<shuf> \
  --data <test> --split-json <split.json> --fullres-data $DCS2_ROOT/hdf5_fullres \
  --upsampler-ckpt <v1> --contexts 50 --seed 0 --device cuda --batch-size 16 --denoising-steps 1 --out results/h1

# WSL: H2 volumetric smoke (should-have)
cs2wm eval h2 --models v2=<v2> v1=<v1> --seed 0           # TBD(#33): not implemented yet

# WSL: carried-over v1 evals (loss curves, rollouts, FID at 30x56)
cs2wm eval carried --models v1=<v1> v2=<v2> --v2-metrics <run>/metrics.jsonl --v1-results results/v1 \
  --data <test> --split-json <split.json> --v1-data <v1 low_res/test> --out results/carried --seed 0 \
  --device cuda --n-fid 2048 --frames 64 --contexts 16

# WSL: regenerate report tables and figures from results/
cs2wm report-assets --results results --out report      # TBD(#36): not implemented yet
```

H1 on contexts without `--split-json` may include training demos; the run warns and records `held_out: false`.

## Play

Play runs on Windows Python (reliable relative-mouse capture; WSLg is fine for quick tests). Keys: W/A/S/D, space,
ctrl, shift, 1/2/3/4, R, LMB/RMB, mouse to turn. `Backspace` resets the context, `F1` toggles the overlay.

```powershell
# Windows PowerShell
cd C:\dcs2\diamond-cs2-v2; C:\dcs2\venv\Scripts\Activate.ps1
$v1 = "$env:DCS2_ROOT\checkpoints\v1\agent_epoch_00149.pt"
$v2 = "$env:DCS2_ROOT\checkpoints\v2\v2-main-e150.pt"
cs2wm play --ckpt $v2 --upsampler-ckpt $v1 --context <held-out test.hdf5>
cs2wm play --ckpt $v2 --upsampler-ckpt $v1 --context <held-out test.hdf5> --record session.mp4
cs2wm play --ckpt $v2 --upsampler-ckpt $v1 --benchmark 200    # FPS table -> results\play_fps.json
```

Expect about 4-6 FPS with the upsampler on an RTX 3060. If it is under 4, add in this order: `--denoising-steps 1`,
`--bf16`, `--compile`, then `--no-upsampler` (bicubic upscale instead).

## Tests

CI (`.github/workflows/tests.yml`, Python 3.10, CPU torch) runs exactly this:

```bash
# WSL / Linux
ruff check cs2wm tests scripts
pytest -q
```

The suite is headless and needs no GPU, CS2 or network.

## Repo layout

```
cs2wm/                  the package (`cs2wm` command)
  config.py             load and validate config/cs2wm.yaml
  actions.py            51-dim action spec, encode/decode, mouse bins
  timeline.py           frame/tick convention, EpisodeSpec
  demo.py               demoparser2 wrapper
  cli.py                argument parsing and dispatch
  pipeline/             generate, segment, render, extract_actions, pack, sync, supervisor, qa_summary, ntfy
  eval/                 rotation, H1, H2 and smoke masks, upsampler check, carried v1 evals, label accuracy
  play/                 keyboard/mouse input, play agent, world model and the interactive loop
  report/               LaTeX table and figure generation
src/                    DIAMOND / v1 training code (adapted, see ATTRIBUTION.md)
config/                 cs2wm.yaml (pipeline), Hydra training configs, hw profiles, CS2 console cfgs
scripts/                hdf5_to_diamond, dataset_stats, verify_manifests, benchmark_denoiser
  mill/                 train.sbatch, benchmark.sbatch, env.example.sh
  windows/              install_task.ps1, run_supervisor.ps1
docs/                   pipeline, dataset, Mill, HLAE and PC setup notes
report/                 CVPR-format report sources
results/                small committed result summaries
tests/                  pytest suite
tasks/                  todo and lessons
```

## Docs

- [docs/PIPELINE.md](docs/PIPELINE.md): stages, supervisor and the frame/tick convention
- [docs/DATASET.md](docs/DATASET.md): dataset datasheet
- [docs/MILL.md](docs/MILL.md): Mill environment, training lifecycle, monitoring
- [docs/HLAE_NOTES.md](docs/HLAE_NOTES.md): validated HLAE/CS2 launch notes
- [docs/PC_SETUP.md](docs/PC_SETUP.md): PC setup details

## Attribution, license and citation

- [DIAMOND](https://github.com/eloialonso/diamond) by Alonso et al. (MIT). `src/` is adapted from it.
- Pearce and Zhu, "Counter-Strike Deathmatch with Large-Scale Behavioural Cloning" (IEEE CoG 2022): the 51-dim
  action space and mouse bins.
- [HLAE](https://github.com/advancedfx/advancedfx) (Half-Life Advanced Effects) renders the demos.
- [demoparser2](https://github.com/LaihoE/demoparser) parses them.
- [v1](https://github.com/cooperwooley/diamond-cs2) (commit `a0b9e24`) supplies the upsampler and the carried-over evals.

Full credits and the DIAMOND license text: [ATTRIBUTION.md](ATTRIBUTION.md). This project is released under the MIT
license: [LICENSE](LICENSE).

```bibtex
@misc{wooley2026diamondcs2v2,
  author       = {Wooley, Cooper},
  title        = {Action-Conditioned, Playable {DIAMOND} World Model for {Counter-Strike 2} Trained on Self-Generated Paired Data},
  year         = {2026},
  howpublished = {CS 5404 course project report, Missouri University of Science and Technology},
  url          = {https://github.com/cooperwooley/diamond-cs2-v2}
}
```
