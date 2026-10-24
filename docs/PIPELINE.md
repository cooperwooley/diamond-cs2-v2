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
   (not from buttons) and writes per-episode QA to `qa/<demo_id>.jsonl`. Failing episodes do not go on.
5. **pack** (#12, `cs2wm.pipeline.pack`). Writes one HDF5 file per episode in v1's `frame_i_x`/`frame_i_y` schema at
   30x56 (resized exactly as v1 did), a deterministic ~10% full-res JPEG subset, and a `manifest.jsonl` of sha256s.
6. **sync** (#16, `cs2wm.pipeline.sync`). rsyncs HDF5, demos and QA to the Mill, verifies every checksum there, and
   only then deletes the local HDF5.
7. **supervise** (#20, `cs2wm.pipeline.supervisor`). A Windows scheduled task that loops over generate and the other
   stages with retries, a disk guard, a CS2 patch guard and ntfy notifications. State is kept in
   `state/pipeline.json`.

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
