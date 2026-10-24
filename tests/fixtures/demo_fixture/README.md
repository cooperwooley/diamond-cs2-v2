# demo_fixture

A 120-tick synthetic demo (two bots, `de_dust2`) for `scripts/fixture_demo.py`; about 17 KB. No `.dem` file is bundled.

- `ticks.csv`, `events/*.csv`: the tables demoparser2 would return. Bot A (user_id 1) walks forward while turning, strafes
  right from tick 60, fires, reloads, jumps and switches weapon; bot B (user_id 2) is a distractor that must not leak
  into bot A's labels.
- `render_plan.json`: the output of the real `segment` stage on those tables (`settle_ticks=10`, `max_frames=12`):
  one segment, two 12-frame episodes.
- `demo.json`: the sidecar `generate` writes next to a demo, plus the demo header.

The rendered frames are drawn at run time by the script (they stand in for HLAE's output).
