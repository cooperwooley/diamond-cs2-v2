# HLAE / CS2 notes

What the #14 validation session (Nov 4-5, 2026) found and how the pipeline uses it. Build: CS2 **26445667** (demo
`patch_version` 14196, auto-updated Nov 4), HLAE **2.192.10**, the CS2 install on drive `D:`. Everything ran offline
with `-insecure`; never run HLAE or these commands on a VAC server. Names and steamids of the local player are left out.

Parser findings (demoparser2 cannot read these demos; demoinfocs-golang can) are in #14 and #72, not here.

## Launch lines

**Generate** (plain CS2, no HLAE), what `cs2wm.pipeline.generate` starts:

```
cs2.exe -insecure -steam -novid -console -sw -w 1280 -h 720 -netconport 2121 +exec generate_dm
```

With Steam running it reaches the menu with no prompts. Port 2121 accepts after ~0.6 s (plain).

**Render** (HLAE custom loader), what `cs2wm.pipeline.render` starts:

```
C:\hlae\HLAE.exe -customLoader -noGui -autoStart -hookDllPath "C:\hlae\x64\AfxHookSource2.dll"
  -programPath "<steam>\steamapps\common\Counter-Strike Global Offensive\game\bin\win64\cs2.exe"
  -cmdLine "-steam -insecure -novid -console -sw -w 1280 -h 720 -netconport 2121 +playdemo \"C:\dcs2\...\x.dem\" +exec render_<demo_id>"
```

(inside PowerShell `Start-Process` the inner quotes are `\"`; `powershell.exe -File` needs `-ExecutionPolicy Bypass`.)

- `+playdemo <abs path> +exec <cfg>` at launch works: the `mirv_cmd` schedule registered by the exec (which runs at
  ~6 s, before the demo loads at ~8.7 s) **survives the demo load**. No load-pause-exec step is needed.
- An absolute `playdemo` path outside `game\csgo` with backslashes works. **Paths with spaces were not tested**, so
  render keeps `paths.raw` and the demo path space-free and fails clearly otherwise (`render.check_space_free`).
- Port 2121 opens ~3-7 s after launch via HLAE. `HLAE.exe` exits with code 0 after ~3-7 s while `cs2.exe` keeps
  running, so render waits for HLAE to exit *and* `cs2.exe` to be gone (`tasklist`).
- `mirv_streams` is available over netcon (hook active). `quit` over netcon closes cleanly.
- The demo plays ~13 s after launch (HLAE defers `playdemo` by ~3 s), then at 1x until the first `record start`.
  Launch to first frame was 64 s for a start tick of 2940. `render.startup_s` is 60; the timeout adds
  `timeout_factor` x demo length.

## Netcon

- A raw TCP **log stream**: all console output is sent, CRLF-terminated, with no end-of-output marker, prompt or
  greeting. The peer does not close. The first connection receives the whole startup backlog (~400 chunks).
- Commands are sent as `cmd\n`. Idle detection works; `echo <marker>` is more robust for sequencing.
- **Lines over ~512 characters are silently truncated** (everything after the cut is lost, no error). `NetconClient`
  splits a long `;`-joined command into lines of at most 400 characters (`netcon.MAX_LINE_CHARS`) and refuses a single
  command over 400.

## Generate: cfg and netcon sequence

`config/cs2/generate_dm.cfg` (order matters):

1. **Before `map`:** `tv_enable 1; tv_transmitall 1; tv_record_immediate 1`. With `tv_enable` after `map` the GOTV
   recording stalled (status online but frozen, a 112 KB file). `tv_transmitall` defaults to 0, which hides players
   from the GOTV demo.
2. `game_type 1; game_mode 2; map de_dust2`, then the bot and weapon cvars. Cvars after `map` in the same cfg do take
   effect (read back: `bot_quota 10`, `bot_allow_rifles`, `bot_join_after_player`, ...).
3. **No `jointeam 1` in the cfg**: it is lost because the local client is not connected yet (the player ended up on T
   with a pistol). Generate sends it over netcon instead.

The gamemode cfg re-executes at map load and resets `mp_*` cvars and `bot_difficulty` (read back 1, `mp_startmoney`
1000, `tv_delay` 30). Cvars that must stick are in `generate_<profile>_post.cfg` and sent over netcon after the load;
the bot cvars set in the main cfg (`bot_quota`, `bot_allow_*`, ...) were read back fine.

Netcon sequence per session (`generate.chunking: gotv`, the default). Each step is verified, not slept:

1. Launch with the cfg; wait for the port (`netcon_connect_timeout_s`, 60).
2. Poll `status` every 2 s until it shows the map (`generate.map`) and at least one human (the local client), at most
   `map_load_timeout_s` (90); then wait `map_load_wait_s` (10; `record` first works ~6 s after `map`). On timeout the
   session fails and CS2 is relaunched.
3. `jointeam 1` while the player is still Unassigned (worked ~6-8 s after `map`). The reply must contain the server's
   `ChangeTeam() ... req team 1` line; otherwise it is retried (`jointeam_retries`, 3), then the session fails. Then
   wait `team_settle_s` (5).
4. The post-load cvars (`config/cs2/generate_<profile>_post.cfg`, e.g. `bot_difficulty 2`) are sent over netcon, in
   lines of at most 400 characters, and single-value cvars are read back; a mismatch logs a warning.
5. `mp_restartgame 1` (`generate.restart_after_join`, default on), then wait `restart_wait_s` (6). This is the tested
   recipe.
6. Per chunk: `sv_cheats 1; tv_record <name>`; after `chunk_min`: `tv_stoprecord`; then the next chunk's `tv_record`.
7. A chunk is closed once its file size is stable for `close_stable_s` (5; it was final within 2 s). Kill and
   relaunch on exit or when no demo grew for `hang_timeout_min`.

`chunking: netcon` is the same with `sv_cheats 1; record <name>` / `stop`.

**Human profile (untested).** The `human` profile sends neither `jointeam 1` nor `mp_restartgame 1`: the human plays,
so they stay on a team. Nothing of the human path was run in #14 (only a bot session with the local player
spectating), so whether a GOTV recording stalls when a human joins a team after `tv_record` is unknown there; the
profile still gets the map wait, the post-load cvars and the recording checks.

**Checks after the start command.** `sv_cheats 1; tv_record` must answer with `Recording SourceTV` (`record`:
`StartRecording`), and the demo file must hold data within `record_check_s` (30) s, otherwise the session is dropped at
once (CS2 is killed and relaunched) instead of waiting for the 5 minute hang timeout. The expected reply texts come
from the #14 log; `generate.verify_netcon_replies: false` turns the text checks off (the file check stays) if CS2
words them differently on another build. The `status` output format is not in the #14 log beyond the map name and
"1 human + 9 bots"; `generate._map_ready` accepts any reply that holds the map name and does not report 0 humans.

- `record` is **cheat-protected** (`SV: Cheat command 'record' ignored.`); `sv_cheats 1` goes on the same line (it
  seems to need re-sending after each map load). Offline and `-insecure` only ("Achievements disabled" is printed).
- `record` before the map is up: `Can't record demo, no local client!`. The `.dem` of a client recording appears only
  once the client receives deltas (7-10 s after the reply): check for the file, not for the reply.
- A name with `-` works unquoted. **Both `record` and `tv_record` write `game\csgo\<name>.dem`.** (`tv_autorecord`
  writes `game\csgo\replays\auto-<date>-<HHMM>-<map>-<server>.dem`, and "one demo per map cycle" never happened in
  5.5 min with `mp_timelimit 3`, so it is not used.)
- `stop` stops only the client demo; `tv_stoprecord` stops GOTV (`Completed SourceTV demo "...": Recording time 186.0,
  Size 13351107`). `tv_status`'s `frames N / ticks N` counter looks stuck: use file growth as the liveness signal.

### GOTV recipe and stall

SourceTV occupies a bot slot named "SourceTV" in `status` and in the demos, and the local player is a spectator in
them. Both must be ignored when enumerating bots. Segment builds lives from `player_spawn`/`player_death` events, so
neither ever becomes a target.

5 of 6 GOTV recordings **stalled** after ~12-16 s of match time. Probable cause: the human joined a team (auto-join
and game restart) *after* `tv_record` had started. The recipe that gave a full 194 s demo (11.3 MB, ~57 KB/s, no
stall): `jointeam 1` as soon as the player is in game, wait ~5 s, send the cvars and `mp_restartgame 1`, wait ~6 s,
and only then `tv_record`. The pipeline follows this order; the bot cvars come from the cfg and are not re-sent.

After a **second `map`** over netcon the player stayed Unassigned on the team-select screen and the client demo was
never written; relaunch CS2 instead of reloading the map.

`tv_delay` is 30 s by default and the gamemode cfg resets it; `tv_delay 0` had no effect when the human was not on a
team, and was not tested otherwise. Everything inside the delay is lost on a crash.

### Weapons

Not checked on a DM demo (the parser could not read them). In a casual-mode test the bots used famas, galilar, ak47
and knives because `mp_t_default_primary` / `mp_ct_default_primary` were not applied. `mp_t_default_primary
weapon_ak47`, `mp_ct_default_primary weapon_m4a1`, `mp_startmoney 16000` and `bot_difficulty 2` all read back fine
when sent over netcon. They are not enabled yet (add them to `generate_dm_post.cfg`): check the bots' weapons (`item_equip` / active
weapon in the demo dump) on the first pilot demo (#19) and enable them if scoped rifles (AUG/SG) appear.

### Partial demos

Killing `cs2.exe` 70 s into `tv_record` leaves a partial demo (3.2 MB) that demoinfocs parses up to its end
(`ErrUnexpectedEndOfDemo`, no panic): ticks 0-2588 with all 10 bots. The last ~`tv_delay` seconds are lost. Generate
keeps such a chunk as `truncated: true` when the parser can read it (`min_chunk_s` decides whether it counts as
progress for `max_failures`). It needs the parser (#72) to report it: `demo.is_parseable(path)` true, or
`demo.inspect(path)` with `truncated: True` and a positive `last_tick`.

### Client demos (`chunking: netcon`)

The client demo contains only what the spectating client sees (the PVS of a free camera): 9 bots, none for the whole
demo. This is why GOTV is the default. Client demos are 64 ticks/s, ~58 KB/s.

## Render: cfg and capture

`config/cs2/render_base.cfg`:

- `spec_show_xray 0` (before recording): the default spectator x-ray draws red silhouettes through walls.
- `mirv_streams add normal <stream>` and `mirv_streams edit <stream> capture beforeUi`: no HUD, crosshair or killfeed;
  the viewmodel stays. Checked visually.
- `mirv_streams settings edit afxDefault settings afxClassic` (TGA), `mirv_streams record fps 16` (forces
  `host_framerate`), `mirv_streams record startMovieWav 0`.
- `mirv_streams record name <raw dir>` and the `mirv_cmd addAtTick` schedule are appended per demo.

Spectating:

- `spec_mode 2` is first person with the viewmodel.
- `spec_player` takes the **controller entity id**, i.e. `user_id + 1` for bots (`status` shows user ids). Evidence:
  `spec_player 3` -> userid 2, `spec_player 10` -> userid 9. It is not the pawn id (~159+). The plan's
  `target.entity_id` must therefore be exactly the controller entity id (#72 provides it).
- A dead target is **silently ignored**: the camera stays on the previous one. Switch only while the bot is alive,
  after its spawn. The plan schedules the switch at spawn + `segment.spec_delay_ticks` (4; #14 saw a switch 3 ticks
  after a spawn work), never before the previous recording ended; recording starts at spawn + `settle_ticks`.
  `mirv_script_spec_lock` does not exist in this HLAE.
- A switch scheduled at tick T shows from the **next frame** (T0+640 showed in frame 161, not 160): a hard cut with no
  transition frames. Keep a margin of at least 1-2 frames (`segment.settle_ticks`).

Ticks: the UI tick and demoinfocs `IngameTick()` equal the `mirv_cmd addAtTick` tick (CGameRules reports +224, the
`server_start_tick` offset, which is not used). Muzzle flashes appeared ~2 frames later than
`ceil((fire_tick - T0) / 4)`; the best fit over three bursts is **+7 ticks** (16/21 shots matched at +7, 11/21 at 0).
`render.tick_offset` applies it as `mirv_tick = demo_tick + tick_offset` but stays 0 until it is measured from camera
rotation (#73): see the measurement below.

### Measured tick offset (#73)

Offline, from the kept render `render2` of `dcs2_test-gotv.dem` (481 frames, T0 2940, `spec_player 4` then `11` at
3580) and the demoinfocs dump: per-frame camera d-yaw from the frames (phase correlation on the upper 60% of
640x360 downsamples, `cs2wm.eval.rotation`, 106.26 deg hFOV) against the d-yaw of the spectated bot at ticks
`T0 + 4k + lag`; pairs within 5 frames of the switch and pairs with a jump over 15 deg (respawns, flashes) are left
out. A lag of `-L` means a frame shows the state `L` ticks **before** its nominal tick, so `tick_offset = +L`.

| lag (ticks) | -8 | -7 | -6 | -5 | -4 | -3 | -2 | 0 | +2 |
|---|---|---|---|---|---|---|---|---|---|
| r (render2, 438 pairs) | 0.73 | 0.80 | 0.89 | **0.94** | 0.94 | 0.89 | 0.82 | 0.60 | 0.29 |

Peak at lag -5 (parabolic fit -4.6), r 0.944, r >= 0.89 from -6 to -3, half-maximum width -11..+1 (the bots' yaw is
smooth, so the curve is broad; the peak is what matters). Per spectated bot: entity 4 (frames 0-150) best lag -5
(r 0.96), entity 11 (frames 170-480) best lag -4 (r 0.94). Without the outlier filter the peak is blurred (-6, r 0.66).
This is a camera-rotation estimate of **+4.6 ticks**, against **+7** from the muzzle flashes; the two are ~2.4 ticks
apart.

The second kept render, `lock30` (T0 4000), could not be used: its cfg and spectated entity are not in the #14 log and
no single bot (nor a split at the camera jumps) correlates with it (best r 0.12-0.3 at inconsistent lags), so it did
not confirm the number. `render.tick_offset` therefore stays 0 until a second render with a known schedule agrees
within 2 ticks; the candidate is +5.

### Output and frame counts

- Layout: `<record name>\take0000\<stream>\00000.tga`, 24-bit uncompressed TGA, 1280x720, 2,764,818 bytes (18-byte
  header + 2,764,800). **OpenCV cannot read these TGAs; numpy can** (`render._read_tga`).
- **n + 1 frames**: record start `T0` and end `T0 + 4n` write n + 1 frames (481 for 1920 ticks, 1921 for 120 s), because
  the frame at the end tick is captured. The plan's `rendered_frames = labeled_frames + 1` matches this.
- **0 frames is a failure.** `C:\hlae\ffmpeg\` has no `ffmpeg.exe`; with an FFmpeg preset (`afxFfmpegLosslessBest`, ...)
  HLAE logs `AFXERROR: COutFFMPEGVideoStream: CreateProcessW` and writes nothing, and CS2 still exits normally. Render
  retries once, then fails the demo with "0 frames captured". Available presets: afxClassic, afxDefault, afxFfmpeg,
  afxFfmpegHuffyuv, afxFfmpegLosslessBest, afxFfmpegLosslessFast, afxFfmpegProres, afxFfmpegProresAlpha,
  afxFfmpegRaw, afxFfmpegYuv420p, afxSampler30. FFmpeg output is untested; to try it, put `ffmpeg.exe` in
  `C:\hlae\ffmpeg` (per the HLAE readme, an `ffmpeg.ini` with `Path=...` also works).

### Throughput and disk

TGA (`afxClassic`): 1920 frames (120 s of demo) in 20.4 s = ~94 fps = **5.9x realtime**; 2.765 MB/frame, 5.31 GB for
that run, ~**159 GB per rendered hour** at 16 fps. A run of 481 frames takes ~3 s of rendering; launch to exit was 57 s
including startup and 1x playback to the start tick. `supervisor.max_working_gb` must cover the takes in flight.
Capturing at a lower resolution (e.g. 640x360) to save disk was not tried.

### Locked screen

With the workstation locked (`rundll32 user32.dll,LockWorkStation`, LogonUI running) all 481 frames were written, none
black, and CS2 exited cleanly. Rendering works on a locked screen; the user has to unlock it themselves.

## Gotchas

- `cs2_video.txt` is rewritten by CS2 (fullscreen, other resolution): keep a backup and restore it after test runs.
- CS2 auto-updates (Steam showed build 26445667 on Nov 4): `generate.expected_build` stops generation on a mismatch.
- demoparser2 0.41.3 and 0.42.1 fail on DM demos from this build (`IllegalPathOp`, LaihoE/demoparser#359); casual
  demos parse only partly. Hence demoinfocs (#72).
- Casual mode (`game_type 0; game_mode 0`) also worked as DM-like (no round end in 3+ min, bots respawn, `mp_roundtime
  60`); deathmatch mode stays the default.
