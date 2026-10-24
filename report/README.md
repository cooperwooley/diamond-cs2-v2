# Report

CVPR-format final report. Build with `latexmk -pdf main.tex` from this directory; it works at any stage, because
every missing result is typeset as a marked `TBD` placeholder.

## Template

`cvpr.sty` and `ieeenat_fullname.bst` come unmodified from the official CVPR author kit
(<https://github.com/cvpr-org/author-kit>, tag `CVPR2026-v1(latex)`). `main.tex` follows the kit's camera-ready setup
(`\usepackage{cvpr}`, review mode off).

## Layout

- `main.tex`, `preamble.tex`, `sections/*.tex`, `refs.bib`, `figures/pipeline.tex` (TikZ, hand-written).
- `placeholders.tex`: the only place that defines values that do not exist yet (`\TBD...`), each with the issue that
  fills it. `main.tex` inputs `tables/macros.tex` first and these are `\providecommand`, so generated macros win.

## Results wiring (#36)

`cs2wm report-assets` writes, and the report reads, these files. Do not edit them by hand.

- `tables/<name>.tex`: a booktabs `tabular` only (no float, caption or label), for `dataset`, `label_accuracy`, `h1`,
  `h2`, `carried`, `play_fps`.
- `tables/macros.tex`: `\newcommand` (or `\providecommand`) definitions, ending in `\xspace`, for the names in
  `placeholders.tex`.
- `figures/<name>.pdf`: `dataset_stats`, `h1_scatter`, `h1_per_bin`, `h2_triptych`, `loss_curves`, `rollout_strip`.
