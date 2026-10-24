# Dataset

Datasheet for the v2 dataset of `diamond-cs2-v2`: paired 16 fps Counter-Strike 2 frames and 51-dimensional action labels, generated from offline bot-deathmatch demos. It is written so that a reader can reproduce the dataset, or judge its biases, without reading the code.

**Status: draft.** The real dataset has not been generated yet. Everything below that follows from the code and `config/cs2wm.yaml` is final. Everything that depends on real data is a placeholder of the form `TBD(#NN): <what goes here>`, where `#NN` is the issue that supplies the value. Find them all with `grep -n "TBD(#" docs/DATASET.md`.

Sources of truth: `config/cs2wm.yaml` (every number below is cited by its key), `cs2wm/actions.py`, `cs2wm/timeline.py`, `cs2wm/pipeline/extract_actions.py` and `cs2wm/pipeline/pack.py`. The stage-by-stage description of the pipeline is in [PIPELINE.md](PIPELINE.md) and is not repeated here.

Contents: [1 Provenance](#1-provenance) - [2 Pipeline summary](#2-pipeline-summary) - [3 Frame/tick convention](#3-frametick-convention) - [4 Label definitions](#4-label-definitions) - [5 QA](#5-qa) - [6 Label accuracy](#6-label-accuracy) - [7 Statistics](#7-statistics) - [8 File schema](#8-file-schema) - [9 Known limitations](#9-known-limitations)

## 1. Provenance

| Item | Value | Source |
|---|---|---|
| CS2 build ID(s) | `TBD(#24): list of distinct buildid values from the demo sidecars (<demo_id>.json), with demo counts` | `buildid` in `steamapps/appmanifest_730.acf`, written to each sidecar by `generate` (#13) and to the HDF5 attr `cs2_build`. `generate.expected_build` is `null` until set after #14 |
| HLAE version | `TBD(#14): HLAE release and AfxHookSource2 version` | Recorded in the #14 paste-back and `docs/HLAE_NOTES.md`; the HLAE/CS2 pair is pinned |
| demoparser2 version | `TBD(#24): version(s) found in the demoparser2_version attr` (the repo requires `>=0.42`, `requirements.txt`; the dev env has 0.42.0) | HDF5 attr `demoparser2_version` |
| cs2wm commit(s) | `TBD(#24): distinct cs2wm_git_sha attr values; TBD(#25): commit used to assemble the DIAMOND datasets` | HDF5 attr `cs2wm_git_sha`; the git commit pasted back from #25 |
| Config hash(es) | `TBD(#24): distinct config_sha256 attr values` | HDF5 attr `config_sha256` (sha256 of the sections `timing`, `resolution`, `episodes`, `actions`, `qa`, `segment`, `pack`, `render` of `config/cs2wm.yaml`; `paths` and `supervisor` are excluded) |
| Generation dates | `TBD(#24): first and last demo start time, from the sidecars` | `start time` field of the sidecar |
| Game | Counter-Strike 2, offline, launched with `-insecure` (mandatory, `generate.launch_options`), never on a VAC server | #13 |
| Map | `de_dust2` (`generate.map`) | candidate until #14 |
| Mode | Deathmatch, `game_type 1; game_mode 2`, local player on `jointeam 1` so only bots play | `config/cs2/generate_dm.cfg`; candidate until #14 |
| Bots | `bot_quota 10` (`generate.bots`), `bot_quota_mode fill`, `bot_difficulty 2`, `bot_join_after_player 0` | `generate_dm.cfg`; candidate until #14 |
| Weapons | Rifles only: `bot_allow_rifles 1`, all other `bot_allow_*` set to 0. If bots still pick scoped rifles (AUG/SG), per-team defaults `weapon_ak47` / `weapon_m4a1` are added. `TBD(#14): final cvars and the weapons the bots actually used` | `generate_dm.cfg`; candidate until #14. The `smoke` kind additionally gives every bot a smoke grenade (`generate_smoke.cfg`, finalised in #23) |
| Graphics | `TBD(#2): CS2 video settings (preset, anti-aliasing, etc.)`. Window and capture size are fixed at 1280x720 (`resolution.capture`, `-w 1280 -h 720`; final choice in #14) | The CS2 video config set up in #2 |
| FOV | CS2's fixed 90 degrees (4:3-based), which is about 106.26 degrees horizontal at 16:9 (`eval.h1.hfov_deg`, to be refit on ground truth in #32) | candidate until #14 |
| Demo type | `TBD(#14): GOTV (tv_record) or client (record) demos, whichever contains full tick and event data for every bot` | #14 decides; gates #13 |
| Episode kinds | `dm` (bots only), `human` (a human plays against bots, #22), `smoke` (grenade-heavy, #23) (`episodes.kinds`) | |
| Render | HLAE `mirv_streams`, captured before the UI, 16 fps (`timing.fps`). `TBD(#14): output format (tga or ffmpeg, render.output_format), render.tick_offset (config says 0, to be measured), segment.settle_ticks (config says 32, to be tuned)` | `config/cs2/render_base.cfg`; candidate until #14 |

## 2. Pipeline summary

A bot deathmatch is played in offline CS2 and recorded as demos in fixed-length chunks. The demo is segmented into player lives, and one first-person view is chosen at a time. HLAE replays the demo at a fixed 16 fps with no HUD, and the frames are downscaled. A 51-dimensional action is derived for every frame from the spectated player's state (position, view angles, ducking, active weapon and a handful of events), not from raw buttons. Episodes that fail QA are dropped. The rest are packed as one HDF5 file per episode of at most 1000 frames, in the same `frame_i_x`/`frame_i_y` schema as v1 at 30x56, plus a deterministic ~10% full-resolution JPEG subset. The files are synced to the Mill and assembled into DIAMOND `.pt` train/test datasets, split by demo. See [PIPELINE.md](PIPELINE.md) for the stages and `config/cs2wm.yaml` for every parameter.

## 3. Frame/tick convention

Defined once, in `cs2wm/timeline.py`.

- `timing.tickrate = 64` and `timing.fps = 16`, so there are `ticks_per_frame = 4`.
- Frame `k` of an episode is the image rendered at tick `T_k = T0 + 4k`, where `T0` is the episode's `start_tick` (HDF5 attr).
- Action `k` covers the tick window `(T_k, T_{k+1}]`, open at the start and closed at the end: it is what the player did while frame `k` was on screen and before frame `k+1`. An event at tick `t` belongs to frame `ceil((t - T0) / 4) - 1`.
- Labeling frame `n-1` needs the state at `T_n`, so an episode of `n` labeled frames needs `n + 1` rendered frames. Only the `n` frames that have an action are stored: `write_episode` requires frames and actions to have the same length, so the extra frame is rendered and then not packed.
- `render.tick_offset` (config value 0) corrects any measured render-to-demo offset. `TBD(#14): measured tick-to-frame offset`.

Worked example with `T0 = 1000` and `n = 3`. The rendered frames are at ticks 1000, 1004, 1008 and 1012; the first three are stored and 1012 is used only for labeling.

| Action | Window | Event ticks that fall in it |
|---|---|---|
| 0 | (1000, 1004] | 1001 to 1004. A `weapon_fire` at tick 1004 sets LMB on frame 0 |
| 1 | (1004, 1008] | 1005 to 1008. A fire at tick 1005 sets LMB on frame 1 |
| 2 | (1008, 1012] | 1009 to 1012. A fire at tick 1010 or 1012 sets LMB on frame 2 |

An event at tick 1000 belongs to frame -1 and one at tick 1013 to frame 3 (outside `[0, 3)`); neither is labeled. These cases are checked against `tick_to_frame`.

At 16 fps one hour is 57,600 frames. An episode holds at most `episodes.max_frames = 1000` frames (62.5 s). Remainders shorter than `episodes.min_frames = 64` frames (4 s) are dropped.

## 4. Label definitions

### 4.1 Layout

The action is `float32[51]` (`cs2wm.actions.ActionSpec`, from `actions.*`), the same layout as v1, DIAMOND CS:GO and Pearce & Zhu (P&Z): `keys[11] + clicks[2] + mouse_x one-hot[23] + mouse_y one-hot[15]`. Keys and clicks are 0/1. Each mouse block is a one-hot over bins.

Frame `k` covers ticks `(T_k, T_{k+1}]`. "Pulse" means the label is on for each frame whose window contains the event. "State rule" means the label is computed from the state sampled at `T_k` and `T_{k+1}`. Missing ticks are filled with the nearest present tick within `qa.max_fill_ticks`; a frame whose state is still missing or NaN gets zero movement and no key labels, and fails QA (see section 5).

| Index | Name | Type | Derivation |
|---|---|---|---|
| 0 | `w` | state rule | Horizontal velocity `v = (pos(T_{k+1}) - pos(T_k)) * fps` (units/s), projected on the forward vector at the window start, `f = v . (cos psi, sin psi)` with `psi = yaw(T_k)`. On if `f > +actions.movement.wasd_deadzone_ups` (30.0) |
| 1 | `a` | state rule | Right component `r = v . (sin psi, -cos psi)`. On if `r < -wasd_deadzone_ups` |
| 2 | `s` | state rule | On if `f < -wasd_deadzone_ups` |
| 3 | `d` | state rule | On if `r > +wasd_deadzone_ups` |
| 4 | `space` | event pulse, or state rule | A `player_jump` event in the window, or not airborne at `T_k` and airborne at `T_{k+1}` with `vz = (z(T_{k+1}) - z(T_k)) * fps > actions.movement.jump_vz_min_ups` (100.0) |
| 5 | `ctrl` | state rule (held) | With `d0 = duck_amount(T_k)` and `d1 = duck_amount(T_{k+1})`, `eps = actions.movement.duck_eps` (0.02): on if `d1 > d0 + eps` (rising), or `d1 >= 1 - eps` and `d1 >= d0 - eps` (fully ducked and not falling) |
| 6 | `shift` | state | `is_walking` at `T_{k+1}` if the demo has that property, else always 0 |
| 7 | `1` | event pulse (state change) | `active_weapon_name` at `T_{k+1}` differs from the one at `T_k` and the new weapon maps to slot 1 (see 4.4) |
| 8 | `2` | event pulse (state change) | Same, slot 2 |
| 9 | `3` | event pulse (state change) | Same, slot 3 ("slot 3+": knife or anything unmatched, including all utility) |
| 10 | `r` | event pulse | A `weapon_reload` event in the window |
| 11 | `lmb` | event pulse plus hold-fill | A `weapon_fire` event in the window; also every frame from one shot to the next when two consecutive shots of the same weapon are at most `actions.fire.hold_gap_s` (0.2 s, 12.8 ticks, so at most 12 ticks) apart, because the trigger was held between them; also a `grenade_thrown` event in the window |
| 12 | `rmb` | event pulse | A `weapon_zoom` event in the window |
| 13-35 | mouse x | state rule, one-hot of 23 | `x = -wrap180(yaw(T_{k+1}) - yaw(T_k))` degrees, binned to the nearest centre (4.2) |
| 36-50 | mouse y | state rule, one-hot of 15 | `y = pitch(T_{k+1}) - pitch(T_k)` degrees, binned to the nearest centre (4.2) |

Choices that are easy to miss:

- **Labels come from state, not buttons.** CS2 delta-encodes usercmds (since about Jul 2026), and demoparser2 0.42 has wrong bit positions for some buttons (`WALK`, `ZOOM`) and no `JUMP`/`DUCK` columns. Raw buttons and usercmds are used only as a non-gating QA cross-check (section 5).
- **Velocity comes from the position displacement over the frame window**, not from demoparser2's per-tick `velocity_*`, which is noisy and spikes at teleports. The displacement is exactly the movement the frame shows.
- **WASD is projected with the yaw at the window start (`T_k`)**, the camera the agent saw when it chose the key.
- **One mouse bin per axis per frame.** Both mouse blocks are always one-hot (exactly one 1 each), so a frame with no mouse movement has the zero bin set.
- Labels of one frame can be many keys at once (for example `w`, `d`, `space`, `lmb`).

### 4.2 Mouse bins

Bin centres are P&Z's values (`actions.mouse.x_units`, `actions.mouse.y_units`). One unit is `actions.mouse.deg_per_unit = 0.055` degrees per 16 Hz frame (0.022 deg/count at CS:GO sensitivity 2.5). Binning is done in degrees: nearest centre, clipped at the two extreme bins, ties to the lower index (as P&Z's `min()`). The last column is therefore the interval of the per-frame rotation that falls in the bin, open at the lower edge and closed at the upper edge (the midpoint between neighbouring centres).

The tables below were generated, not typed. Regenerate them with:

```bash
PYTHONPATH=$PWD python - <<'EOF'
from cs2wm.actions import ActionSpec
from cs2wm.config import load_config

spec = ActionSpec.from_config(load_config())

def table(name, units, centers, offset):
    print(f"| vector idx | {name} bin | centre (P&Z units) | centre (deg) | captures delta (deg) |")
    print("|---|---|---|---|---|")
    edges = [(a + b) / 2 for a, b in zip(centers[:-1], centers[1:])]
    for i, (u, c) in enumerate(zip(units, centers)):
        lo = "-inf" if i == 0 else f"{edges[i-1]:+.4f}"
        hi = "+inf" if i == len(centers) - 1 else f"{edges[i]:+.4f}"
        print(f"| {offset + i} | {i} | {u:+.0f} | {c:+.3f} | [{lo}, {hi}] |")

table("x", spec.mouse_x_units, spec.mouse_x_centers_deg, spec.mouse_x_slice.start)
print()
table("y", spec.mouse_y_units, spec.mouse_y_centers_deg, spec.mouse_y_slice.start)
EOF
```

Mouse x (23 bins, vector indices 13-35; +x is a turn to the right):

| vector idx | x bin | centre (P&Z units) | centre (deg) | captures delta (deg) |
|---|---|---|---|---|
| 13 | 0 | -1000 | -55.000 | [-inf, -41.2500] |
| 14 | 1 | -500 | -27.500 | [-41.2500, -22.0000] |
| 15 | 2 | -300 | -16.500 | [-22.0000, -13.7500] |
| 16 | 3 | -200 | -11.000 | [-13.7500, -8.2500] |
| 17 | 4 | -100 | -5.500 | [-8.2500, -4.4000] |
| 18 | 5 | -60 | -3.300 | [-4.4000, -2.4750] |
| 19 | 6 | -30 | -1.650 | [-2.4750, -1.3750] |
| 20 | 7 | -20 | -1.100 | [-1.3750, -0.8250] |
| 21 | 8 | -10 | -0.550 | [-0.8250, -0.3850] |
| 22 | 9 | -4 | -0.220 | [-0.3850, -0.1650] |
| 23 | 10 | -2 | -0.110 | [-0.1650, -0.0550] |
| 24 | 11 | +0 | +0.000 | [-0.0550, +0.0550] |
| 25 | 12 | +2 | +0.110 | [+0.0550, +0.1650] |
| 26 | 13 | +4 | +0.220 | [+0.1650, +0.3850] |
| 27 | 14 | +10 | +0.550 | [+0.3850, +0.8250] |
| 28 | 15 | +20 | +1.100 | [+0.8250, +1.3750] |
| 29 | 16 | +30 | +1.650 | [+1.3750, +2.4750] |
| 30 | 17 | +60 | +3.300 | [+2.4750, +4.4000] |
| 31 | 18 | +100 | +5.500 | [+4.4000, +8.2500] |
| 32 | 19 | +200 | +11.000 | [+8.2500, +13.7500] |
| 33 | 20 | +300 | +16.500 | [+13.7500, +22.0000] |
| 34 | 21 | +500 | +27.500 | [+22.0000, +41.2500] |
| 35 | 22 | +1000 | +55.000 | [+41.2500, +inf] |

Mouse y (15 bins, vector indices 36-50; +y is a look down):

| vector idx | y bin | centre (P&Z units) | centre (deg) | captures delta (deg) |
|---|---|---|---|---|
| 36 | 0 | -200 | -11.000 | [-inf, -8.2500] |
| 37 | 1 | -100 | -5.500 | [-8.2500, -4.1250] |
| 38 | 2 | -50 | -2.750 | [-4.1250, -1.9250] |
| 39 | 3 | -20 | -1.100 | [-1.9250, -0.8250] |
| 40 | 4 | -10 | -0.550 | [-0.8250, -0.3850] |
| 41 | 5 | -4 | -0.220 | [-0.3850, -0.1650] |
| 42 | 6 | -2 | -0.110 | [-0.1650, -0.0550] |
| 43 | 7 | +0 | +0.000 | [-0.0550, +0.0550] |
| 44 | 8 | +2 | +0.110 | [+0.0550, +0.1650] |
| 45 | 9 | +4 | +0.220 | [+0.1650, +0.3850] |
| 46 | 10 | +10 | +0.550 | [+0.3850, +0.8250] |
| 47 | 11 | +20 | +1.100 | [+0.8250, +1.9250] |
| 48 | 12 | +50 | +2.750 | [+1.9250, +4.1250] |
| 49 | 13 | +100 | +5.500 | [+4.1250, +8.2500] |
| 50 | 14 | +200 | +11.000 | [+8.2500, +inf] |

### 4.3 Sign conventions

- **Mouse x: +x is turn right.** Source yaw increases counter-clockwise (turning left), so `x = -wrap180(d_yaw)`. `wrap180` maps the yaw difference into [-180, 180), so a wrap across +-180 degrees is handled.
- **Mouse y: +y is look down.** Source pitch increases when looking down, so `y = +d_pitch`.
- Live play uses the same convention: counts are converted with `counts * play.m_yaw * play.sensitivity` degrees (`m_yaw = m_pitch = 0.022`), then binned the same way. At `play.sensitivity = 2.5` this reproduces P&Z's units exactly.
- Movement: `w` is forward along the view direction and `d` is to the right of it, both measured with the yaw at `T_k`.

### 4.4 Weapon slots, and the slot-3+ decision

The 51-dim space has keys `1`, `2`, `3` and no slot-4 key (this is the v1/P&Z layout). So the weapon-switch labels map the new `active_weapon_name` (lower-cased, `weapon_` prefix removed, substring match) through `actions.slots`:

- slot `1`: the listed rifles, SMGs, shotguns, machine guns and snipers (`actions.slots."1"`);
- slot `2`: the listed pistols (`actions.slots."2"`);
- anything else maps to `actions.slots.default = "3"`, which therefore means "slot 3 and above": the knife and every utility item, including smoke, flash, HE, decoy, molotov, the bomb and the taser. The smoke grenade is therefore indistinguishable from the knife in the action space, which matters for the `smoke` kind and H2.

`TBD(#14): the exact CS2 weapon strings seen in the demos, and any weapon that falls into the wrong slot` (the config marks the slot lists as unconfirmed until #14).

## 5. QA

Per episode, `extract_episode` computes the statistics below and `qa/<demo_id>.jsonl` stores one record per episode: `episode_id`, `passed`, `reasons` and `stats`. An episode that fails any gating check is not passed to `pack`. Thresholds are the `qa.*` keys of `config/cs2wm.yaml`.

| Check (stat name) | Threshold | Meaning | Fails the episode when |
|---|---|---|---|
| fire recall (`fire_recall`) | `qa.min_fire_recall = 0.98` | Share of the player's `weapon_fire` events whose frame has LMB = 1 | below 0.98 |
| missing ticks (`missing_tick_frac`) | `qa.max_missing_tick_frac = 0.01` | Fraction of ticks in `[T0, T_n]` absent from the demo table | above 0.01 |
| NaN frames (`nan_frames`) | 0 | Frames whose labels depend on a state sample that is missing (beyond `qa.max_fill_ticks = 2` ticks of filling) or not finite | any |
| duplicate ticks (`dup_ticks`) | 0 | More than one row per tick for the filtered player, meaning two players were mixed | any |
| mouse saturation (`mouse_saturation_frac`) | `qa.max_mouse_saturation_frac = 0.02` | Fraction of frames in an extreme bin of x or y (x beyond +-41.25 degrees, y beyond +-8.25 degrees per frame) | above 0.02 |
| yaw jump (`max_abs_dyaw_deg`) | `qa.max_abs_dyaw_deg_per_frame = 60.0` | Largest absolute yaw change in one frame: spectator glitch or teleport | above 60 degrees |
| teleport (`max_disp_per_frame`) | `qa.teleport_factor * qa.max_speed_ups / fps = 1.5 * 320 / 16 = 30` units | Largest horizontal displacement in one frame | above 30 units per frame |
| episode length (`n_frames`) | `episodes.min_frames = 64` | Labeled frames | below 64 |
| button cross-check (`xcheck_FORWARD`, `xcheck_BACK`, `xcheck_LEFT`, `xcheck_RIGHT`, `xcheck_FIRE`, `xcheck_RIGHTCLICK`, `xcheck_RELOAD`) | none | Per-frame agreement between the state-derived label (`w`, `s`, `a`, `d`, `lmb`, `rmb`, `r`) and the raw button column, when the demo has that column | never (reported only) |

`max_speed_ups = 320` and `max_abs_dyaw_deg_per_frame` are caps for glitch detection, not game-physics claims.

Every record also carries `n_fire_events` and the label rates `rate_<name>` for each of the 13 buttons plus `rate_mouse_x_nonzero` and `rate_mouse_y_nonzero`, which feed the dataset statistics.

Caveats:

- Within `extract_episode`, `fire_recall` is 1.0 by construction, because LMB is built from the same `weapon_fire` events. It becomes an alignment check only when recomputed from the packed labels against an independent signal (the pilot QA, #19), or via `xcheck_FIRE` when button columns exist.
- The teleport check uses the horizontal displacement only; a purely vertical teleport is not caught by it.

Results (to be filled from the `qa/*.jsonl` files of the full run):

| Kind | Episodes | Dropped | Drop rate | Most frequent failure reason |
|---|---|---|---|---|
| `dm` | `TBD(#24)` | `TBD(#24)` | `TBD(#24)` | `TBD(#24)` |
| `human` | `TBD(#24)` | `TBD(#24)` | `TBD(#24)` | `TBD(#24)` |
| `smoke` | `TBD(#24)` | `TBD(#24)` | `TBD(#24)` | `TBD(#24)` |

Button cross-check agreement (mean of `xcheck_*` over passing episodes):

| Column | Label | Agreement |
|---|---|---|
| `FORWARD` | `w` | `TBD(#24): agreement` |
| `BACK` | `s` | `TBD(#24): agreement` |
| `LEFT` | `a` | `TBD(#24): agreement` |
| `RIGHT` | `d` | `TBD(#24): agreement` |
| `FIRE` | `lmb` | `TBD(#24): agreement` |
| `RIGHTCLICK` | `rmb` | `TBD(#24): agreement` |
| `RELOAD` | `r` | `TBD(#24): agreement` |

`TBD(#14): whether the demo kind in use exposes these button columns at all. If not, this table is empty and says so.`

## 6. Label accuracy

Method (#18): a human reviews a rendered clip of about 2 minutes with the labels overlaid (keys, LMB/RMB, mouse bins as arrows plus numbers, frame index) at half speed, using `cs2wm label-check render`. Because episodes are capped at 1000 frames (about 62 s), the clip is two consecutive episodes of one segment and the review sheets are pooled. The sheet has one row per label transition (key down/up, fire, reload, switch, jump) plus 200 random frames for mouse direction and magnitude. The reviewer marks each row `ok` = 1 (correct), 0 (wrong), or `m` on a `mouse_x` row where the direction is right but the magnitude is off by more than one bin. A row is judged by the change from that frame to the next, because action `k` covers frames `k` to `k+1`. `cs2wm label-check score` then reports the accuracy per slot group; a group with fewer than 10 judged rows is flagged `low_n`. Any group below 90% had its thresholds tuned in `config/cs2wm.yaml` and was re-scored. The success target is every group at least 90% and mouse-x sign at least 98%.

| Slot group | Accuracy | Judged rows |
|---|---|---|
| WASD | `TBD(#18)` | `TBD(#18)` |
| jump | `TBD(#18)` | `TBD(#18)` |
| crouch | `TBD(#18)` | `TBD(#18)` |
| fire | `TBD(#18)` | `TBD(#18)` |
| reload | `TBD(#18)` | `TBD(#18)` |
| switch | `TBD(#18)` | `TBD(#18)` |
| mouse-x sign | `TBD(#18)` | `TBD(#18)` |
| mouse-x bin within +-1 | `TBD(#18)` | `TBD(#18)` |
| mouse-y | `TBD(#18)` | `TBD(#18)` |

- Clip reviewed: `TBD(#18): episode IDs, duration of the review`.
- Thresholds used (`actions.movement.*` at the commit that was scored): `TBD(#18): commit hash and final wasd_deadzone_ups, jump_vz_min_ups, duck_eps`.
- Systematic errors found: `TBD(#18): e.g. key onset or release timing, crouch hold length`.
- Raw result: `results/label_accuracy.json`.

## 7. Statistics

`scripts/dataset_stats.py` computes the statistics from the assembled `.pt` datasets (the low-res frames are not read). Regenerate everything below with:

```bash
python scripts/dataset_stats.py $S/datasets/v2-full --json $S/datasets/v2-full/stats.json
```

Use the true-action dataset `v2-full`. In `v2-full-shuffled` the train actions are other episodes' actions, tiled or cropped to length, so its label statistics differ slightly. Hours are `frames / timing.fps / 3600`. Only labeled, stored frames are counted. The JSON has the fields `fps`, `total`, `train`, `test` and `by_kind`; each of `total`, `train` and `test` holds `episodes`, `frames`, `hours`, `length_hist` (edges and counts, 10 bins over 0 to `episodes.max_frames`), `label_rates`, `mouse_x_hist`, `mouse_y_hist`, `all_zero_frac` and `idle_frac`. `by_kind[kind][split]` holds `episodes`, `frames` and `hours`.

Add `--figure stats.png` to also render a figure of the train/test label rates and the mouse-x/y histograms (matplotlib, `Agg` backend; mouse histograms are normalised per split and on a log axis). `TBD(#25): the figure from the full dataset`.

### 7.1 Size

| Kind | Split | Episodes | Frames | Hours |
|---|---|---|---|---|
| `dm` | train | `TBD(#25)` | `TBD(#25)` | `TBD(#25)` |
| `dm` | test | `TBD(#25)` | `TBD(#25)` | `TBD(#25)` |
| `human` | train | `TBD(#25)` | `TBD(#25)` | `TBD(#25)` |
| `human` | test | `TBD(#25)` | `TBD(#25)` | `TBD(#25)` |
| `smoke` | train | `TBD(#25)` | `TBD(#25)` | `TBD(#25)` |
| `smoke` | test | `TBD(#25)` | `TBD(#25)` | `TBD(#25)` |
| all | train | `TBD(#25)` | `TBD(#25)` | `TBD(#25)` (target at least 10 h, goal 20 h) |
| all | test | `TBD(#25)` | `TBD(#25)` | `TBD(#25)` |
| all | total | `TBD(#25)` | `TBD(#25)` | `TBD(#25)` |

Episode-length histogram (frames, 10 bins from 0 to 1000): `TBD(#25): counts per bin, train and test`. Number of demos per kind and split (from `split.json`): `TBD(#25)`.

### 7.2 Label rates

Fraction of frames with the label on (`label_rates`), by split, plus `all_zero_frac` (strictly zero 51-vector, which cannot happen for a valid label since each mouse block is one-hot, so it counts unlabeled rows) and `idle_frac` (no button or click and both mouse blocks on the zero bin).

| Label | Train | Test |
|---|---|---|
| `w` | `TBD(#25)` | `TBD(#25)` |
| `a` | `TBD(#25)` | `TBD(#25)` |
| `s` | `TBD(#25)` | `TBD(#25)` |
| `d` | `TBD(#25)` | `TBD(#25)` |
| `space` | `TBD(#25)` | `TBD(#25)` |
| `ctrl` | `TBD(#25)` | `TBD(#25)` |
| `shift` | `TBD(#25)` | `TBD(#25)` |
| `1` | `TBD(#25)` | `TBD(#25)` |
| `2` | `TBD(#25)` | `TBD(#25)` |
| `3` | `TBD(#25)` | `TBD(#25)` |
| `r` | `TBD(#25)` | `TBD(#25)` |
| `lmb` | `TBD(#25)` | `TBD(#25)` |
| `rmb` | `TBD(#25)` | `TBD(#25)` |
| `all_zero_frac` | `TBD(#25)` | `TBD(#25)` |
| `idle_frac` | `TBD(#25)` | `TBD(#25)` |

### 7.3 Mouse histograms

Frames per bin (`mouse_x_hist`, `mouse_y_hist`; keyed by bin centre in degrees, so they line up with the tables in 4.2). Columns are the bin indices of 4.2.

| Axis | Split | Counts per bin |
|---|---|---|
| x (23 bins) | train | `TBD(#25)` |
| x (23 bins) | test | `TBD(#25)` |
| y (15 bins) | train | `TBD(#25)` |
| y (15 bins) | test | `TBD(#25)` |

Share of frames in the zero bin and in the extreme bins: `TBD(#25)`.

## 8. File schema

### 8.1 Low-res episode HDF5 (`hdf5/<kind>/<demo_id>/<episode_id>.hdf5`)

One file per episode of `n` frames, `1 <= n <= episodes.max_frames` (1000). Written by `cs2wm.pipeline.pack.write_episode` and read by `read_episode`; `schema_version = 1`.

| Key | dtype | Shape | Content |
|---|---|---|---|
| `frame_{i}_x`, `i = 0..n-1` | `uint8` | `(30, 56, 3)` (`resolution.low_res`) | Low-res frame, **BGR**, raw (no compression, `encoding = "raw"`) |
| `frame_{i}_y`, `i = 0..n-1` | `float32` | `(51,)` | The action of frame `i` (section 4.1), always finite |

Every frame and every label is its own dataset, as in v1. The low-res image is made as v1 made it (`to_low_res`, a port of v1's `create_low_res_dataset`): BGR to RGB, scale to [-1, 1], torchvision bicubic resize (antialiased) from 150x280 to 30x56, `add(1).div(2).mul(255)`, **clamp to [0, 255]**, cast to uint8, back to BGR. The clamp is the one deliberate difference from v1 (see section 9, v1 wrap bug).

File attrs (Python scalars; `read_episode` returns them as plain Python values):

| Attr | Type | Value / source |
|---|---|---|
| `n_frames` | int | Number of frames `n` |
| `schema_version` | int | `pack.schema_version` (1) |
| `fps` | int | `timing.fps` (16) |
| `tickrate` | int | `timing.tickrate` (64) |
| `config_sha256` | str | Hash of the label-affecting config sections (section 1) |
| `encoding` | str | `"raw"` here, `"jpeg"` in the full-res file |
| `height`, `width` | int | 30, 56 here; 150, 280 in the full-res file |
| `episode_id` | str | `<kind>-<demo_id>-s<segment:03d>-c<chunk:02d>`, stable across re-runs |
| `demo_id` | str | Parent demo |
| `kind` | str | One of `episodes.kinds` (`dm`, `human`, `smoke`) |
| `player` | str | Spectated player |
| `start_tick` | int | `T0`, the tick of frame 0 |
| `cs2_build` | str or int | CS2 build ID |
| `demoparser2_version` | str | demoparser2 version used for the labels |
| `cs2wm_git_sha` | str | cs2wm commit used for the labels |

The caller may pass further scalar (str, int, float, bool or None) meta keys, which are written as attrs too (None becomes the empty string); keys that collide with the writer's own attrs are rejected. `jpeg_quality` appears in the full-res file only.

### 8.2 Full-res JPEG subset (`hdf5_fullres/<kind>/<demo_id>/<episode_id>.hdf5`)

An episode is in the subset iff `int(sha1(episode_id)[:8], 16) % pack.fullres_mod == 0` with `pack.fullres_mod = 10` (hash of the UTF-8 episode ID, first 8 hex digits as an integer), about 10% of episodes, deterministically and independent of the order of processing. The subset is for evaluating the upsampler.

The file has the same keys and attrs as the low-res file, except:

- `frame_{i}_x` is a 1-D `uint8` array holding the JPEG bytes of the 150x280 (`resolution.render`) BGR frame, encoded with OpenCV at quality `pack.jpeg_quality = 95`; decode with `cv2.imdecode(..., cv2.IMREAD_COLOR)` to get BGR back. The JPEG is lossy, so decoded pixels equal the rendered frame only up to JPEG error.
- attrs `encoding = "jpeg"`, `jpeg_quality = 95`, `height = 150`, `width = 280`.
- `frame_{i}_y` and all other attrs are identical to the low-res twin, and the file name is the same.

### 8.3 Writing and manifest

Files are written to `<name>.tmp`, fsynced and renamed, so a crash leaves no partial file; the low-res file is renamed last and is the commit point.

`hdf5/<kind>/<demo_id>/manifest.jsonl` has one JSON object per line (keys sorted), appended with `append_manifest` after each episode:

| Field | Type | Meaning |
|---|---|---|
| `file` | str | File name relative to the manifest's directory, for example `<episode_id>.hdf5`. The full-res twin is recorded as `fullres/<episode_id>.hdf5` |
| `frames` | int | Number of frames |
| `sha256` | str | Hex sha256 of the file |

`scripts/verify_manifests.py $S/hdf5 [--fullres-root $S/hdf5_fullres]` re-checks every `file` against its `sha256` (the `frames` field is not re-checked); `sync` (#16) verifies against it after transfer.

### 8.4 DIAMOND datasets (`$S/datasets/<name>`)

Produced by `scripts/hdf5_to_diamond.py --hdf5 $S/hdf5 --out $S/datasets/<name> --kinds dm,human,smoke --test-frac 0.1 --seed 0 [--shuffle-actions]`.

```
<name>/
  low_res/
    train/
      info.pt                    Dataset state (episode lengths, start indices, counters)
      000/00/0/0.pt              episode 0  (episode i is at <(i//100)%10*100:03d>/<(i//10)%10*10:02d>/<i%10>/i.pt)
      ...
    test/                        same layout
  split.json                     {demo_id: "train" | "test"}, sorted by demo id
  dataset_meta.json              seed, flags, shuffle info, source manifest sha256s, one row per episode
```

Each episode `.pt` is a dict saved with `torch.save`:

| Key | dtype | Shape | Content |
|---|---|---|---|
| `obs` | `uint8` | `(n, 3, 30, 56)` | Frames, **RGB**, channel-first (the HDF5 BGR flipped), bit-identical to the HDF5 pixels |
| `act` | `float32` | `(n, 51)` | Actions (in the shuffled variant, another train episode's actions, tiled or cropped to `n`) |
| `rew` | `float32` | `(n,)` | zeros |
| `end` | `uint8` | `(n,)` | zeros |
| `trunc` | `uint8` | `(n,)` | zeros |
| `info` | dict | | `original_file_id` (`<demo_id>/<episode_id>.hdf5`), `episode_id`, `demo_id`, `kind` |

Split rules:

- **By demo**, so no demo is in both train and test; deterministic for a given `--seed`. The split depends on the full set of demos present, so the true and shuffled datasets are built together from one HDF5 snapshot with the same seed and share an identical `split.json`.
- Each kind is split on its own: `ceil(test_frac * n_demos)` demos go to test, always keeping at least one in train.
- For `smoke`, which is the H2 evaluation set, at least 30% of the demos and at least 3 go to test. With fewer than 3 smoke demos all of them go to test and train has none.
- **Shuffled-action control** (`--shuffle-actions`, the H1 null): each train episode receives the action sequence of another train episode. The preferred method is a cross-demo shift, which guarantees that no episode receives actions from its own demo when no demo holds more than half of the train episodes; otherwise a seeded derangement is used (`dataset_meta.json` records `shuffle.method` and `same_demo_swaps`). The test split keeps its true actions. This preserves the marginal and temporal statistics of the actions while breaking the frame-action correspondence.

## 9. Known limitations

State-derived labels (what the labels are not):

- **No slot-4 key.** The 51-dim space (inherited from v1 and P&Z) has keys `1`, `2`, `3` only. Smoke and every other utility item, and the knife, share index 9. The model cannot be told to equip a smoke versus a knife.
- **Velocity-derived WASD lags by about 1 frame.** W/A/S/D are inferred from displacement, not key state, so a key release is seen only after the velocity has decayed below `wasd_deadzone_ups`, and a press only once the player is moving. Source acceleration and friction cause this. `TBD(#18): measured lag and accuracy`. Pressing into a wall or a teammate produces no displacement and so no label.
- **Bots do not press keys.** For bot episodes the "keys" are the keys that would reproduce the observed motion, and the "mouse" is the bot's view-angle change. Bot aim motion differs from human mouse motion (speed, smoothness, flick patterns), so the mouse histograms describe bot aiming. Only the `human` kind has real human input.
- **Pulse labels are approximate in time.** Weapon-switch labels fire when the active weapon name changes, which includes automatic switches that no key caused (after a grenade throw, an empty magazine, a pickup, or a spawn). Reload, jump and zoom come from game events, and a held key is not recoverable for them. `rmb` is set only by `weapon_zoom`, so a secondary attack with a knife is not labeled.
- **`shift` is 0 whenever the demo lacks `is_walking`.** `TBD(#14): whether the property exists in the demos`.
- **Hold-fill for fire** assumes the trigger was held between shots of one weapon at most 0.2 s apart; a fast tap sequence is labeled as a held trigger.
- **LMB doubles as grenade throw** (a throw sets `lmb`), so `lmb` is not only "primary fire".

Visual and distribution biases:

- **Bot behaviour bias.** All actions come from CS2's bot AI at one difficulty (`bot_difficulty 2` candidate) in a fixed deathmatch loop: predetermined routes, no human tactics, little jumping or crouching, and restricted weapons. The `dm` kind is bots only; only `human` (#22) adds human behaviour, and only for the sessions recorded.
- **One map.** Everything is `de_dust2` (`generate.map`). The model has seen no other map. The train/test split is by demo, but the demos are all one map, so the test score does not measure generalisation to new maps.
- **One point of view at a time, switching only at a death** (the `longest_remaining` segment policy for bots, `fixed_player` for human sessions). The first and last `segment.settle_ticks` of a life (camera transition; config value 32, `TBD(#14)`) are dropped.
- **No HUD.** Frames are captured before the UI: no crosshair, ammo, health, killfeed or scoreboard. The viewmodel (held weapon) should remain visible (`TBD(#14): confirmed no-HUD capture`). The model cannot read ammo or health, so reload and similar events are not predictable from the HUD. v1's upsampler was trained on frames with HUD (`Retrain the upsampler on HUD-free frames` is future work in #1), so see #27 for the compatibility check.
- **Low resolution and aspect ratio.** The model sees 30x56 (and 150x280 for the upsampler). The capture is 16:9 (1280x720, ratio 1.78) and the stored size 150x280 has ratio 1.87, so unless `render` crops, the image is stretched horizontally by about 5%. `TBD(#15): how the render stage maps 720x1280 to 150x280 (resize or crop)`. This also affects the `eval.h1.hfov_deg` (106.26) used in H1.
- **v1's uint8 wrap fixed in v2.** v1's low-res recipe ended with `.byte()`, which wraps instead of clamping. Bicubic resampling overshoots [-1, 1] at hard edges, and torch's float-to-uint8 cast wraps: for example 257.5 becomes 1 and -3.0 becomes 253, so white turned near-black and black near-white. v1 therefore trained on, and its upsampler was conditioned on, salt-and-pepper artifacts at bright/dark edges. v2 clamps to [0, 255] before the cast; the pack tests assert bit-equality with v1 wherever v1 stays in range and that wrapped values are clamped instead. On a synthetic high-contrast edge image, 3.7% of low-res pixels differ from the clamped result; on random noise none do (measured in #12). The v1 upsampler is reused, so #27 checks that it is still fine on clean inputs (expected, since only artifacts are removed). v2 low-res frames are therefore not bit-identical to what v1 saw at hard edges.
- **Episode edges.** Episodes are cut at 1000 frames; remainders under 64 frames are dropped. The last labeled frame of an episode uses the state of an extra frame that is not stored.
- **Test split is the same distribution as train** (same map, same bots), apart from the `smoke` demos held out for H2.
- **Pipeline status.** At the time of writing, `extract_actions.run`, `pack.run` and the demo parser wrapper are stubs (`NotImplementedError`) in this repository; the numbers in this document will come from the version that fills them in (`TBD(#25): commit`).
