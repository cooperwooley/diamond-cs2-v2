# results

Small summary files (JSON, a few CSVs, final figures) are committed so the report can be rebuilt from the repo with
`cs2wm report-assets`. Large outputs (checkpoints, rollouts, frame dumps, HDF5, videos) are not committed; they live
under `$DCS2_ROOT` on the PC or `$S` on the Mill.

- `v1/`: training-loss summaries from the v1 project, copied verbatim (used by #34).
- `upsampler_check.json`, `upsampler_check_grid.png` (#27): v2 upsampler (#70) vs bicubic on the held-out full-res test
  frames.
- `h1/` (#32): controllability, v2 vs the shuffled-action control. The per-frame `*_frames.csv` files are not committed
  (about 9 MB each, gitignored); they are on the PC under `$DCS2_ROOT/eval/h1_frames`. `summary.json` already holds the
  per-(script, context) scatter points the report needs.
- `carried/` (#34): loss curves (v1 vs v2), rollouts and FID at 30x56.
- Later: `label_accuracy.json` (#18), `play_fps.json` (#28), `h2/` (#33).
