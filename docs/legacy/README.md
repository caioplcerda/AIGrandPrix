# Legacy plots — pre-rebuild `gym_env` stack

These two figures come from the original internal `gym_env` stack, before the
rebuild onto the MAVLink2/NED interface described in
[`.genie/brainstorms/round1-readiness/DESIGN.md`](../../.genie/brainstorms/round1-readiness/DESIGN.md).

They are kept as a record of where the project started. They are **not** current
results, and should not be read as such:

| | These plots | Current stack |
|---|---|---|
| Gates | 4/5 | 5/5 |
| Run time | ~40 s | 3.68 s |
| Average speed | 1.2 m/s | — |
| Interface | internal `gym_env` | MAVLink2/UDP, NED |

The attitude oscillation and saturated motor RPM visible here are exactly the
behaviour the rebuild was meant to eliminate.

## Legacy videos — `videos/`

`videos/hard_courses_3d/` and `videos/flight_visualization.*` are rendered by
`scripts/hard_course_videos.py`, which drives `simulation.gym_env` — the same
legacy stack. They top out around 4-6 m/s and are not representative of the
current code either.

For a video of the current stack, see `scripts/runs/`, rendered by
[`scripts/render_run_video.py`](../../scripts/render_run_video.py).
