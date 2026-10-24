# Pipeline

cs2wm turns offline Counter-Strike 2 bot deathmatches into paired (frame, 51-dim action) episodes. Seven components
cooperate: six stages that each advance a demo one step, and a supervisor that runs them unattended. Everything that
must agree between stages lives in `config/cs2wm.yaml` (load it with `cs2wm.config.load_config`).

```
generate -> segment -> render -> extract_actions -> pack -> sync        (run by: supervise)
```

## Stages

1. **generate** (#13, `cs2wm.pipeline.generate`). Launches offline CS2 with `-insecure` (the launcher refuses to start
   without it), plays a `de_dust2` bot deathmatch and records demos in N-minute chunks into
   `demos/<kind>/<demo_id>.dem` with a JSON sidecar. A watchdog restarts crashed or hung runs. The console commands
   come from `config/cs2/generate_*.cfg` (validated in #14).
2. **segment** (#10, `cs2wm.pipeline.segment`). Turns `player_spawn`/`player_death` events into lives, picks whom to
   spectate (one in-eye point of view at a time, switching only at a death), and splits the view segments into
   episodes of at most 1000 frames. Writes `plans/<demo_id>/render_plan.json`.
   The JSON schema is documented in the `cs2wm.pipeline.segment` module docstring.
3. **render** (#15, `cs2wm.pipeline.render`). Replays the demo in CS2 through HLAE `mirv_streams` at a fixed 16 fps,
   captured before the UI (no HUD), and downscales the frames to 280x150. Verifies per-segment frame counts.
4. **extract_actions** (#11, `cs2wm.pipeline.extract_actions`). Derives a 51-dim action per frame from player state
   (not from buttons). Needs the plan and `frames/<demo_id>/render.json` of the same plan. Writes `labels/<demo_id>/<episode_id>.npy`
   for passing episodes and rewrites `qa/<demo_id>.jsonl` (one line per episode). Failing episodes do not go on.
5. **pack** (#12, `cs2wm.pipeline.pack`). Writes one HDF5 file per episode in v1's `frame_i_x`/`frame_i_y` schema at
   30x56 (resized exactly as v1 did), a deterministic ~10% full-res JPEG subset, and a `manifest.jsonl` of sha256s
   (rewritten per run, so re-runs never duplicate lines). Each episode is read back and its actions must equal the
   labels. `frames/<demo_id>/` and `labels/<demo_id>/` are deleted once every episode is packed.
6. **sync** (#16, `cs2wm.pipeline.sync`). rsyncs HDF5, demos and QA to the Mill, verifies every checksum there, and
   only then deletes the local HDF5.
7. **supervise** (#20, `cs2wm.pipeline.supervisor`). A Windows scheduled task that loops over generate and the other
   stages with retries, a disk guard, a CS2 patch guard and ntfy notifications. State is kept in
   `state/pipeline.json` (done in #19: `pending -> segmented -> rendered -> extracted -> packed -> synced | failed`,
   per-stage wall times).

   `cs2wm supervise --until HH:MM --target-hours H [--profile dm|smoke]` repeats "generate one block
   (`supervisor.generate_block_min`), advance every pending demo through segment, render, extract and pack, sync"
   until the deadline (the next `HH:MM`, so 07:30 at 23:00 is tomorrow) or until H QA-passing hours are synced.
   Generation stops `supervisor.drain_min` before the deadline. Each stage gets `supervisor.max_attempts` attempts per
   demo (`attempts` in the state) with `backoff_s * 2^(n-1)` between them; a failed render also kills CS2/HLAE. Guards:
   generation pauses while the working set (frames, raw, labels, hdf5, hdf5_fullres; not the demo archive) exceeds
   `max_working_gb` and stops with an alert below `min_free_gb` free; when the CS2 `buildid` changes, a ~10 s render
   smoke test of the most recent demo's plan (`render_smoke_test`, in `<root>/smoke`) must pass first, else generation
   and rendering stop and a high-priority alert goes out. ntfy messages: start, hourly heartbeat, alerts (guard trips,
   repeated failures, sync down) and the morning summary. The exit code is 3 if the night ended on an alert. The
   scheduled task is `scripts/windows/install_task.ps1` (task `dcs2-supervisor`, interactive session);
   `run_supervisor.ps1` logs to `logs/supervisor-<date>.log`.

By hand (the #19 pilot): `cs2wm segment|render|extract|pack --all-pending` runs a stage over every demo that is ready
for it, in id order; a failing demo is marked `failed` (stage and reason) and the rest still run; the exit code is 1
if any failed. `--retry-failed` retries the demos that failed in that stage. `cs2wm sync` marks verified demos
`synced`, `cs2wm status` prints the state table with wall seconds per stage per demo-hour, and `cs2wm qa-summary
[--kind K] [--since 24h]` aggregates `qa/*.jsonl`. Logs: `logs/cs2wm.log` and stderr.

On the Mill, #17 assembles the synced HDF5 into DIAMOND `.pt` datasets for training (#29, #30).

## Frame/tick convention

Defined once, in `cs2wm.timeline`; no other module may re-derive it.

- A demo has `tickrate = 64`; frames are rendered at `fps = 16`, so `ticks_per_frame = 4`.
- Frame `k` of an episode is rendered at tick `T_k = T0 + 4k`, where `T0` is the episode's `start_tick`.
- Action `k` is what the player did while frame `k` was on screen and before frame `k + 1`: it covers the ticks
  `(T_k, T_{k+1}]` (open at the start, closed at the end).
- An event at tick `t` belongs to frame `ceil((t - T0) / 4) - 1`.
- Labeling the last frame `n - 1` needs the state at `T_n`, so every episode renders one extra frame (`end_tick` is
  the tick of frame `n`).
- `render.tick_offset` in the config corrects any measured render-to-demo offset (measured in #14).

Examples with `T0 = 0`: an event at tick 4 is in frame 0 (window `(0, 4]`); an event at tick 5 is in frame 1
(window `(4, 8]`); an event at tick 0 would fall before frame 0 and belongs to no frame.

## Mouse signs and bins

Labels and live input share one definition, in `actions.mouse` of `config/cs2wm.yaml`:

- Bin centres are Pearce & Zhu's values (`x_units`, `y_units`); one unit is `deg_per_unit = 0.055` degrees per frame.
  Binning happens in degrees: nearest centre, clipped at the extremes, ties to the lower index.
- `+x` is turn right (the Source yaw decreases); `+y` is look down (the Source pitch increases). So
  `x = -wrap180(d_yaw)` and `y = +d_pitch`.
- Play converts mouse counts with `counts * m_yaw * sensitivity` degrees before binning; at sensitivity 2.5 this
  reproduces Pearce & Zhu's units exactly.
