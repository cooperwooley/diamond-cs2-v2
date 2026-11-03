# Todo

Tracking issue: https://github.com/cooperwooley/diamond-cs2-v2/issues/1

## Phase 1: Issues
- [x] Create issues #1 to #39 with specs, labels, milestones; pin #1

## Phase 2: Skeleton
- [x] Repo skeleton: stubs for `cs2wm/` and `scripts/`, configs, docs, report, tests, CI
- [x] Imported v1 code with attribution (#5)

## Phase 3: First code (branch `feat/9-11-12-extract-pack`)
- [x] #9 config and action spec (`cs2wm/config.py`, `cs2wm/actions.py`)
- [x] #10 frame/tick helpers (`cs2wm/timeline.py`; the rest of segment is still open)
- [x] #11 extract_actions core (`extract_episode`, QA); `run()` waits on #15 render outputs
- [x] #12 pack core (`to_low_res`, `write_episode`, `read_episode`, manifest); `run()` waits on #15

## Review (Phase 3)
- 210 tests pass, ruff is clean. Everything runs on synthetic data; nothing needs CS2.
- An independent review found one real bug: mouse-bin ties were decided by float rounding, so a one-count
  live mouse move could bin differently from the same move labeled offline. It is fixed, and an integer-count
  sweep against the Pearce & Zhu reference fails without the fix and passes with it.
- Hardened from the same review:
  - duplicate ticks (an unfiltered multi-player table) now fail QA;
  - pack fsyncs and renames low-res last;
  - provenance attrs and the episode id are validated;
  - internal weapon names map to the right slots.
- Found and fixed a v1 bug: `.byte()` wraps bicubic overshoot (257.5 -> 1). v2 clamps; see #12.
- Still open: `fire_recall` is 1.0 by construction inside `extract_episode`. It becomes a real alignment check
  only when recomputed from packed labels in the #19 pilot.

## Phase 4: Dev loop over the open issues (2026-10-23 to 10-24)
Each code issue got the same treatment:
1. A branch stacked on the issue it depends on.
2. Implementation.
3. An independent review, with its fixes applied.
4. A PR.
5. A comment on the issue listing any manual steps, with exact commands.

PRs #40–#61 are merged into `main` (CI green, 1152 tests).

- [x] M1: #5, #6, #7, #9, #10, #11, #12, #13, #15, #16, #17, #18 (tooling), stage wiring for #19
- [x] M2: #20, #26 (draft), #27, #28
- [x] M4: #32, #33, #34, #36
- [x] M5: #37 (draft), #38 (README, fixture demo, fresh-clone check)
- [ ] Manual or data-dependent, posted on the issues: #2, #3, #4, #8, #14 (`CONFIRM(#14)` checklist), #19, #21–#25, #29–#31, #35
- [ ] #39 slides: needs results (#35, #36)

## Review (Phase 4)
Reviews found and fixed these before merge:
- **sync (#16):** CRLF stdin on Windows (verification would never pass) and missing timeouts.
- **generate (#13):** crash loops that never alerted, Windows file locks that killed the run, and duplicate publishes.
- **supervisor (#20):** retry multiplication, Mill outages ending the night early, and a disk guard tripped by failed demos.
- **train.sbatch (#7):** a lost exit code during the USR1 trap.
- **H1 (#32):** regression dilution in the px→deg calibration. The fix regresses px on the exact label.
- **H2 (#33):** inscribed-ellipse smoke masks, which gave 0 px on near smokes. The fix is an exact ray/sphere test.
- **Pack (#12):** a crash window during cleanup, and a fire-mask wrap at episode starts. The fire-mask bug came from PR #40's extractor.
- **Report assets (#36):** negative numbers in math mode, and null controls getting bold.
