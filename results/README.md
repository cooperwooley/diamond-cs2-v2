# results

Small summary files (JSON, a few CSVs, final figures) are committed so the report can be rebuilt from the repo with
`cs2wm report-assets`. Large outputs (checkpoints, rollouts, frame dumps, HDF5, videos) are not committed; they live
under `$DCS2_ROOT` on the PC or `$S` on the Mill.

- `v1/`: training-loss summaries from the v1 project, copied verbatim (used by #34).
- Later: `label_accuracy.json` (#18), `upsampler_check.json` (#27), `play_fps.json` (#28), `h1/` (#32), `h2/` (#33),
  `carried/` (#34).
