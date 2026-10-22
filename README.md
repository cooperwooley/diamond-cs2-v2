# diamond-cs2-v2

CS 5404 final project. [v1](https://github.com/cooperwooley/diamond-cs2) trained a DIAMOND diffusion world model for
Counter-Strike 2 on about 5.7 h of YouTube footage, where every action label was all-zero, so the model could not be
controlled. v2 generates its own paired data (offline bot deathmatches rendered with HLAE, 51-dim actions derived from
player state) and retrains the denoiser on it, giving an action-conditioned, playable world model.

## Status

Skeleton: signatures, configs and docs only; see the pinned tracking issue
[#1](https://github.com/cooperwooley/diamond-cs2-v2/issues/1).

## Repo layout

```
cs2wm/                  the package (stubs until the linked issues land)
  config.py             load and validate config/cs2wm.yaml (#9)
  actions.py            51-dim action spec, encode/decode, mouse bins (#9)
  timeline.py           frame/tick convention, EpisodeSpec (#10)
  demo.py               demoparser2 wrapper (#11)
  cli.py                the `cs2wm` command
  pipeline/             generate, segment, render, extract_actions, pack, sync, supervisor, ntfy
  eval/                 rotation, H1, H2, upsampler check, carried-over v1 evals, label accuracy
  play/                 keyboard/mouse input and the interactive loop
  report/               LaTeX table and figure generation
src/                    DIAMOND / v1 training code (adapted, see ATTRIBUTION.md)
config/                 cs2wm.yaml (pipeline), Hydra training configs, hw profiles, CS2 console cfgs
scripts/                benchmark, dataset assembly and stats, Mill sbatch, Windows tasks
docs/                   pipeline, dataset, Mill, HLAE and PC setup notes
report/                 CVPR-format report sources
results/                small committed result summaries
tests/                  pytest suite
tasks/                  todo and lessons
```

## Quickstart

<!-- TODO(#38): replace these placeholders with tested copy-paste commands. -->

- PC setup (WSL env, Windows venv, HLAE, CS2): [#2](https://github.com/cooperwooley/diamond-cs2-v2/issues/2)
- Mill setup: [#3](https://github.com/cooperwooley/diamond-cs2-v2/issues/3)
- Pilot data run: [#19](https://github.com/cooperwooley/diamond-cs2-v2/issues/19)
- Training: [#29](https://github.com/cooperwooley/diamond-cs2-v2/issues/29)
- Playing the model: [#28](https://github.com/cooperwooley/diamond-cs2-v2/issues/28)

## Docs

- [docs/PIPELINE.md](docs/PIPELINE.md): stages and the frame/tick convention
- [docs/DATASET.md](docs/DATASET.md): dataset datasheet
- [docs/MILL.md](docs/MILL.md), [docs/PC_SETUP.md](docs/PC_SETUP.md), [docs/HLAE_NOTES.md](docs/HLAE_NOTES.md)
- [ATTRIBUTION.md](ATTRIBUTION.md) and [LICENSE](LICENSE) (MIT)

## Results

<!-- TODO(#38): headline H1/H2 numbers and a demo GIF. -->
