# Attribution

This project builds on the work of others. Sources, and the licenses they carry, are listed here.

## DIAMOND

- Alonso et al., "Diffusion for World Modeling: Visual Details Matter in Atari", NeurIPS 2024.
- https://github.com/eloialonso/diamond (MIT License)
- `src/` is adapted from DIAMOND (trainer, agent, data pipeline, denoiser and upsampler models). Every Python file
  there carries a header crediting DIAMOND and v1.

The full DIAMOND license text:

```text
MIT License

Copyright (c) 2024 Eloi Alonso

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## diamond-cs2 (v1)

- https://github.com/cooperwooley/diamond-cs2 at commit `a0b9e24`
- The immediate source of `src/` and `config/` (trainer, agent, env and world_model_env configs), of the v1 loss
  curves under `results/v1/`, and of the evaluation scripts (`evaluate.py`, `eval_single_step.py`, `compute_fid.py`)
  that are ported into `cs2wm/eval/` in issue #34. v1's `src/cs2/` (hard-coded action bins) is not carried over;
  action encoding lives in `cs2wm.actions`.

## Counter-Strike Deathmatch with Large-Scale Behavioural Cloning

- Pearce and Zhu, https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning
- The source of the 51-dimensional action layout (11 keys, 2 clicks, 23 mouse-x bins, 15 mouse-y bins) and of the
  mouse bin values used in `config/cs2wm.yaml`.

## Tools

- HLAE (Half-Life Advanced Effects), https://github.com/advancedfx/advancedfx: renders demos at a fixed frame rate.
  It is only ever run offline with `-insecure`.
- demoparser2, https://github.com/LaihoE/demoparser: parses CS2 demos (player state and events).

## This project

Released under the MIT License; see `LICENSE`.
