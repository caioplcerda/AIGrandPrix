# AI Grand Prix — Autonomous Drone Racing

Autonomy stack for the [AI Grand Prix](https://theaigrandprix.com) virtual qualifier: fly a
quadrotor through a sequence of gates, as fast as possible, with no GPS, no map, and no human
in the loop. Perception, planning and control, plus the simulator they were developed against.

**Python · MAVLink2 · YOLOv8 · 6-DOF rigid-body sim · ~13,700 lines**

---

## Status, honestly

This stack was never run against the official DCL simulator. That binary was Windows-only and
had not been released when development started, and Track F — Windows validation and
submission — is the one track that never happened.

So every number below is measured against `mock_sim/`, a simulator I wrote to the organizers'
interface specification (VADR-TS-002): 120 Hz rigid-body physics, gate-plane crossing
detection, and a UDP vision stream with the same 24-byte chunk header, camera tilt and
pinhole intrinsics as the real thing. It is a faithful drop-in for the documented interface,
and it is still my own model of their system. Read the results as "the controller does what
it should against the specified physics", not as a competition result.

The gap is the interesting part, and it is stated rather than hidden.

---

## The problem

The interface is deliberately austere:

| | |
|---|---|
| Control | `SET_POSITION_TARGET_LOCAL_NED` over MAVLink2/UDP |
| Telemetry | `ATTITUDE`, `HIGHRES_IMU`, `TIMESYNC` — **no GPS, no global position** |
| Vision | Forward camera, 640×360 @ 30 Hz, tilted +20° up, JPEG over UDP |
| Frame | NED, origin at the arming point |
| Gates | 1500 mm inner opening, 2700 mm outer |

No absolute position fix. Position comes from integrating IMU in a NED estimator, and the
drift that integration accumulates is corrected by the one absolute reference available —
seeing a gate and knowing how big it is.

## Architecture

```
camera (UDP 5600) ─► gate_detector ──► state_estimator ──► path_planner ──► drone_controller
                     YOLOv8n              NED, IMU +          cubic spline      50 Hz
                     + HSV fallback       vision correction   5 m lookahead     MAVLink out
```

| Module | Job |
|---|---|
| `comms/mavlink_client.py` | MAVLink2 link, 2 Hz heartbeat, telemetry decode |
| `comms/vision_stream.py` | Reassembles chunked JPEG frames off UDP |
| `perception/gate_detector.py` | YOLOv8n gate detection, HSV colour fallback |
| `perception/state_estimator.py` | NED state from IMU, corrected by gate bearings |
| `planning/path_planner_ned.py` | Gate ordering and approach geometry |
| `planning/trajectory_opt.py` | Cubic-spline trajectory under acceleration limits |
| `control/drone_controller.py` | 50 Hz position/velocity loop |
| `control/mpc_controller.py` | Model-predictive alternative controller |
| `mock_sim/` | The simulator: 120 Hz physics, gate detection, vision stream |

Distance to a gate is recovered from its apparent size: pixel bearings through a pinhole model
(`fx=fy=320, cx=320, cy=180`), tilt-compensated for the +20° camera, then blended softly into
the state estimate — gain 0.3, correction capped at 2 m, replanning forced on correction. A
hard snap to a single noisy detection would put a step into the trajectory at 40 m/s.

## Results

Against `mock_sim`, five gates at 12 m spacing:

| | |
|---|---|
| Gates passed | 5/5 |
| End-to-end | 3.68 s (from 9.8 s — see below) |
| Control loop | 50 Hz |
| Unit tests | 243 passing |

Progressive stress batteries, each course generated to be harder and faster than the last:

| Battery | Speed | Result | Courses |
|---|---|---|---|
| Hard | 25 m/s | 6/6 | 3D spiral, compound diagonal, 10-gate circuit |
| Extreme | 25 m/s | 10/10 | figure-8, omega, corkscrew, 6 m gates |
| Ultra | 30 m/s | 8/8 | 25-gate speed wall, 30-gate grand prix |
| Legend | 30 m/s | 6/6 | 40-gate marathon, 30-gate hypersonic |
| Godtier | 30 m/s | 8/8 | 50-gate marathon, chaos 30, triple helix |
| Omega | 33 m/s | 8/8 | 50-gate diagonal, hypersonic 35 |
| Titan | 38 m/s | 8/8 | genuine 38 m/s flight |
| Gauntlet | 38 & 48 m/s | 60/60 | all patterns chained, either segment order |

## Three findings that mattered

**The drone was hard-capped at 20 m/s and every earlier benchmark was wrong.**
Terminal velocity in the physics model is `sqrt(MAX_ACCEL / DRAG)` — `sqrt(40 / 0.10)` = 20 m/s.
The `max_speed` parameter shaped the planned trajectory but the vehicle could never fly it, so
months of "speed" tuning had been measuring a number the simulator would not produce. Dropping
`DRAG` to 0.03 and raising `MAX_ACCEL` to 60 lifted terminal velocity to 44.7 m/s and cut the
end-to-end run 9.8 s → 3.72 s. It also broke the controller: drag had been supplying implicit
velocity damping, and in the low-drag regime `kd_pos` had to go 2.0 → 8.0 or the drone
overshot every gate. A parameter that looks like a physics detail was the binding constraint on
the whole system.

**Speed-adaptive lookahead was implemented, measured, and reverted.**
Scaling lookahead with velocity is the obvious optimization and it is wrong here: on lateral
weaves it targets a point past the offset, so the drone flies straight through where the gate
isn't. The straight-line end-to-end test never exposed it — 3.68 s versus 3.72 s, noise. The
60-gate gauntlet did: **1/60 gates with adaptive lookahead, 60/60 with fixed 5 m.** Completion
is the primary objective, so fixed wins, and the speed gain it appeared to deliver was
actually the drag fix landing in the same window.

**Deeper planning is worse.** Planning against `rem[:1]` — the next gate only — beat both
alternatives outright: `rem[:2]` was 5× slower (199 s vs 35.6 s on the helix) and `rem[:3]`
cut corners badly enough to fail 4 of 20. Anticipatory braking, tested the same way, fights
the approach-blend and stalls the run: 11/60 with, 60/60 without.

All three are negative results, kept in the repo so they are not re-attempted.

## Perception

YOLOv8n trained on 5,000 synthetic images generated by `data_generator.py`, with domain
randomization across eight background types (dark arena, stadium lights, grass, overexposed,
gradient, stripe, noise, solid) and motion blur, JPEG artifacts, Gaussian blur, brightness and
sensor noise.

| | Baseline | Stressed |
|---|---|---|
| Precision | 1.000 | 1.000 |
| Recall | 0.966 | 1.000 through motion blur to 31 px, JPEG quality 8, brightness 0.2–2.8× |
| Occlusion | — | 1.000 to 50%, 0.912 at 75% |
| Distance error | 1.06 m | 1.73 m |
| Inference | 11.8 ms | 30 Hz capable |

Robust *within the synthetic domain*. Real sim-to-real transfer is untested, for the reason in
the status section.

## Course design rules

Derived from courses the vehicle physically could not fly:

- Minimum gate spacing ≥ 8 m at 30 m/s; 5 m is impossible. Scale with speed.
- Altitude delta ≤ 5 m between consecutive gates.
- Minimum turn radius is `v²/MAX_ACCEL` — 24 m at 38 m/s. A course cannot demand tighter.
- Helices need inward-pointing normals and radius ≥ 12 m; split spirals past 8 gates.

## Running it

```bash
pip install -r requirements_vq1.txt

# Validate the whole stack against the mock simulator
python scripts/test_e2e_mock.py

# Full validation suite
./scripts/validate_all.sh

# Against a real DCL sim endpoint
python run_vq1.py --host <ip> --mavlink_port 14550 --vision_port 5600 \
  --cnn_model datasets/gate_yolo_mps/runs/gate_detector2/weights/best.pt

# Retrain the detector
python scripts/train_yolo_gate.py --n_train 5000 --epochs 50 --device mps
```

Trained weights are committed at `datasets/gate_yolo_mps/runs/gate_detector2/weights/best.pt`.
Rendered flight videos for the stress courses are in `scripts/hard_courses_3d/`.

## Repository layout

| Path | |
|---|---|
| `src/aigrandprix/` | The stack — perception, planning, control, comms, mock sim |
| `run_vq1.py` | Competition entry point |
| `tests/` | 243 unit tests across 19 modules |
| `scripts/test_*_courses.py` | Stress batteries, one per difficulty tier |
| `scripts/hard_courses_3d/` | Rendered flight videos |
| `docs/` | Roadmap, task breakdown, study notes |
| `CLAUDE.md` | Detailed engineering log — phases, findings, tuning history |
| `.claude/`, `.genie/` | Agent tooling and design notes — see below |

## On the tooling

`CLAUDE.md`, `.claude/commands/` and `.genie/` are the agent harness this was built with, kept
in the repo deliberately. The engineering judgement — deciding the 20 m/s cap was suspicious,
building a gauntlet specifically to catch what the straight-line test missed, throwing away
three optimizations that measured worse — is the work. The harness is how the work was
executed, and hiding it would misrepresent the process.

## Licence

MIT — see [LICENSE](LICENSE). Competition rules retain full IP ownership with the entrant.

The organizers' specification document (VADR-TS-002) is not redistributed here; it is theirs.

---

Built by [Caio Lacerda](https://github.com/caioplcerda) · Aerospace Engineering, CU Boulder
