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
