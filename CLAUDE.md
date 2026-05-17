# AI Grand Prix - Autonomous Drone Racing Competition

## Competitor
**Caio de Paula Lacerda** | Teams of up to 8 allowed

## Competition Overview
The **AI Grand Prix** is a global autonomous drone racing competition by **Anduril** in partnership with **Drone Champions League (DCL)**, **Neros Technologies**, and **JobsOhio**.

- **Prize Pool**: $500,000 + job opportunity at Anduril
- **Website**: https://theaigrandprix.com
- **Contact**: contact_aigp@theaigrandprix.com
- **Entry Fee**: None (teams cover their own travel/accommodation expenses)

## Timeline
| Phase | Date | Details |
|-------|------|---------|
| 1st Virtual Qualifier (Round 1) | May 2026 | Windows app, downloadable DCL platform, time-based scoring |
| 2nd Virtual Qualifier | June 2026 | Second chance / improved submissions |
| Physical Qualifier | September 2026 | In-person, Southern California (2 weeks) |
| AI Grand Prix Final | November 2026 | Columbus, Ohio - live head-to-head racing |

**Spec oficial**: VADR-TS-002 Issue 00.02 (2026-05-08) — MAVLink2/UDP interface released. See `260508_Technical_Spec_0002.pdf`.

## Technical Requirements

### What We Must Build
An autonomous system that navigates a drone through a sequence of gates in a virtual environment, as fast as possible.

### Round 1 Format (1st Virtual Qualifier)
- **Platform**: Windows 11 app (DCL sim). Linux **not supported**.
- **Interface**: MAVLink2 over UDP (MAVSDK-compatible). No REST/gRPC/custom API.
- **Control**: `SET_POSITION_TARGET_LOCAL_NED` or `SET_ATTITUDE_TARGET` — client's choice
- **Telemetry**: `ATTITUDE` (quat) + `HIGHRES_IMU` (accel, gyro, vel) + `TIMESYNC`. **No GPS, no global position.**
- **Vision**: Forward camera, 640×360 @ 30Hz, tilt +20° up, UDP port 5600, JPEG chunks with 24-byte header. **Pinhole: fx=fy=320, cx=320, cy=180, VFoV=90°**
- **Coordinates**: NED (X north, Y east, Z down). Origin = arming point (0,0,0).
- **Physics**: Rigid body 120Hz (thrust, drag, gravity, collision)
- **Gates**: Outer 2700×2700×260mm, inner opening 1500×1500mm
- **Drone**: 280×280×160mm chassis (Neros Technologies)
- **Course VQ1**: <10 gates, max 8 min, primary goal = completion
- **Course VQ2**: <20 gates, faster time wins
- **Hardware min**: i5-10400F, RTX 2060 Super (8GB VRAM), 16GB RAM, 60GB storage
- **Internet**: active connection required (anti-cheat)
- **IP**: Full ownership retained

### Three Core Pillars
1. **Gate Recognition (Perception)**: Detect and locate gates using sensor data + visual feed. Gates are mostly standardized.
2. **Drone Control**: Command flight dynamics (speed, orientation, thrust) with precision. Balance speed vs accuracy.
3. **Path Planning & Navigation**: Plot efficient route through all gates in correct order under realistic physics.

### Competition Rules
- **NO manual control** - fully autonomous
- **NO hardware modifications** - identical Neros Technology drones with DCL AI vector module
- **Python-based AI algorithms** submitted to DCL-built platform
- **Fastest time wins** - but passing all gates is the primary goal
- Software quality is the ONLY differentiator
- No entry fees

### Hardware (Provided)
- Identical drones built by **Neros Technologies**, incorporates DCL AI vector module
- Chassis: 280×280×160mm
- Gate outer: 2700×2700×260mm; inner opening: 1500×1500mm

## Development Strategy

### Approach: MAVLink-Native Stack (VADR-TS-002 compliant)

**CRITICAL**: Internal gym_env stack is incompatible with real DCL interface. Full rebuild underway.

Real interface: MAVLink2/UDP, NED coordinates, no GPS, mandatory vision, Windows 11 only.

Strategy: mock-first (build faithful VADR-TS-002 mock + autonomy stack; swap IP when DCL binary releases).

Pillars:
- **MAVLink client** (pymavlink): heartbeat, ATTITUDE+IMU recv, POSITION_TARGET_LOCAL_NED send, vision stream UDP 5600
- **State estimation**: quaternion attitude + linear velocity integration (no GPS)
- **NED path planner**: gate-to-gate waypoints in NED, lookahead velocity
- **Vision CNN**: YOLOv8n trained on synthetic 640×360 gate images (tilt +20°), fallback HSV detector
- **Mock DCL sim**: faithful UDP server for development/validation before official binary

### Architecture

**Phase 6 target (MAVLink-native, NED):**
```
src/aigrandprix/
├── comms/                    # NEW — MAVLink2/UDP interface
│   ├── mavlink_client.py     # pymavlink client, heartbeat, recv ATTITUDE+IMU, send POSITION_TARGET
│   └── vision_stream.py      # UDP 5600 JPEG chunk reassembler (header 24B, LE)
├── mock_sim/                 # NEW — Mock DCL sim for dev/validation
│   ├── dcl_mock_server.py    # UDP MAVLink2 server + vision publisher
│   ├── physics_6dof.py       # 6DOF rigid body 120Hz integrator
│   └── gate_renderer.py      # OpenCV gate projection (pinhole, tilt +20°)
├── state/                    # NEW — State estimation (no GPS)
│   └── state_estimator.py    # quat(ATTITUDE) + vel_ned integration from HIGHRES_IMU
├── perception/               # UPDATED
│   ├── gate_detector.py      # CNN (YOLOv8n) + HSV fallback, output: bearing NED + dist estimate
│   ├── cnn_model.py          # YOLOv8n weights (trained on synthetic 640×360 gates, tilt +20°)
│   └── data_generator.py     # Synthetic gate dataset: 2.7m outer, NED, tilt +20°, varied bg
├── planning/                 # REFACTORED to NED
│   └── path_planner.py       # NED waypoints → POSITION_TARGET_LOCAL_NED
├── control/                  # LEGACY (gym_env mode, preserved for regression)
│   ├── drone_controller.py   # PID (ENU, simplified physics) — do not break
│   └── mpc_controller.py     # MPC — preserved for Round 2
├── simulation/               # LEGACY (gym_env) — preserved, 243 tests
│   ├── gym_env.py
│   └── run_sim.py
├── submission/               # STUB — awaiting DCL API
│   └── entry_point.py
├── run_vq1.py                # NEW — single entry point: python run_vq1.py --host X --mavlink_port Y
└── utils/
    └── math_utils.py
```

**Key MAVLink messages (VADR-TS-002):**

| Message | Direction | Purpose |
|---------|-----------|---------|
| HEARTBEAT | both | Connection keepalive (≥2Hz from client) |
| ATTITUDE | Sim→Client | Vehicle attitude (quat, rates) |
| HIGHRES_IMU | Sim→Client | Accel, gyro, linear velocity |
| TIMESYNC | Sim→Client | Clock sync |
| SET_POSITION_TARGET_LOCAL_NED | Client→Sim | Primary control command |
| SET_ATTITUDE_TARGET | Client→Sim | Alternative (reserved Round 2) |

**Coordinate convention:** NED. X=north, Y=east, Z=down. Altitude = -Z.
**Camera:** forward, tilt +20° up from body. Frame 640×360. fx=fy=320, cx=320, cy=180, VFoV=90°.

### Key Reference Repositories

**Tier 1 - Most Critical (study these first):**
- **uzh-rpg/agile_flight** ("Swift") - STATE OF THE ART: beat human world champions via deep RL. Full pipeline: perception, planning, control. Published in Nature 2023.
- **utiasDSL/gym-pybullet-drones** (~1200 stars) - Best lightweight RL training environment, Gymnasium-compatible
- **CodexLabsLLC/Colosseum** (~1500 stars) - Actively maintained AirSim fork, full-featured drone sim with vision
- **microsoft/AirSim-NeurIPS2019-Drone-Racing** (~350 stars) - Purpose-built racing with minimum jerk trajectories

**Tier 2 - Essential Infrastructure:**
- **utiasDSL/lsy_drone_racing** - Autonomous drone racing with crazyflow simulator, Python 3.10+, progressive difficulty
- **ethz-asl/mav_trajectory_generation** (~800 stars) - Polynomial trajectory through waypoints/gates (minimum snap)
- **uzh-rpg/flightmare** (~1000 stars) - High-speed RL training with Unity rendering + vision
- **uzh-rpg/agile_autonomy** (~500 stars) - Learning high-speed flight in the wild, sim-to-real

**Tier 3 - Specialized Components:**
- **open-airlab/GateNet** - Neural network for gate perception in drone racing
- **uzh-rpg/deep_drone_racing** (~200 stars) - End-to-end vision-based racing, sim-to-real transfer
- **ZJU-FAST-Lab/ego-planner** (~1000 stars) - Fast gradient-based local replanning
- **mit-acl/faster** (~300 stars) - MIT fast and safe trajectory planner
- **arplaboratory/learning-to-fly** (~200 stars) - NYU ultra-fast drone RL training (massively parallel GPU)
- **uzh-rpg/rpg_mpc** (~300 stars) - Model Predictive Control for quadrotors
- **phuongboi/drone-racing-using-reinforcement-learning** - PPO-based drone gate racing
- **aqeelanwar/PEDRA** - Programmable Engine for Drone RL Applications
- **google-deepmind/mujoco** (~8000 stars) - High-fidelity physics engine, drone models available

### Key Technologies
- **Python 3.10+** (competition requirement)
- **PyTorch** for neural networks (perception + RL)
- **OpenCV** for image processing
- **NumPy/SciPy** for numerical computation
- **Gymnasium** for RL environment interface
- **PyBullet** for physics simulation
- **stable-baselines3** for RL algorithms (PPO, SAC)

## Development Guidelines

### Code Style
- Python 3.10+ with type hints
- Follow PEP 8
- Use dataclasses for configs
- Docstrings for public APIs only
- Tests in `tests/` directory

### Performance Priority
1. **Gate completion** is the primary goal — pass all gates in correct order
2. **Lap time** is the tiebreaker — fastest valid run wins among completions
3. Optimize for reliability first, then speed through gates
4. Profile and benchmark everything
5. Sim-to-real transfer must be considered

### Testing
- Unit tests for each module
- Integration tests for full pipeline
- Benchmark tests for lap timing
- Sim validation against reference trajectories

## Phase 2 Benchmarks — Trajectory Planning

Comparison of trajectory methods on a 5-gate course with turns and altitude changes (max_speed=15, max_accel=10).

### Trajectory Quality (plan_through_gates)

| Method | Duration | Avg Spd | Max Acc | Path Len | Max Jerk | Avg Jerk | Solve Time |
|--------|----------|---------|---------|----------|----------|----------|------------|
| cubic_spline | 3.56s | 10.62 | 323 | 38.0m | 4,801 | 362 | 3ms |
| min_jerk | 4.32s | 9.03 | 319 | 37.7m | 10,918 | 841 | 9ms |
| min_snap | 4.32s | 10.00 | 287 | 46.5m | 5,789 | 594 | 15ms |
| min_snap + racing_line | 4.20s | 9.93 | 285 | 44.8m | 5,791 | 592 | 15ms |
| min_snap + RL + time_alloc | 6.92s | 5.46 | **10.2** | **37.0m** | **55** | **25** | 97s |

### Sim Flight (5 gates, 3 seeds, max_speed=12)

| Method | Avg Gates | Avg Spd | Completed | Crashed |
|--------|-----------|---------|-----------|---------|
| cubic_spline (baseline) | 4.0 | 1.12 | 0/3 | 0/3 |
| min_snap | 4.0 | 1.60 | 0/3 | 1/3 |
| min_snap + racing_line | 2.0 | 1.90 | 0/3 | 1/3 |

### Key Findings

1. **Time allocation is the smoothness game-changer**: max jerk drops 100x (5,789 -> 55), max accel drops 28x (287 -> 10). Solve time (~97s) needs optimization for real-time use.
2. **Min-snap reduces peak acceleration** vs cubic_spline (287 vs 323) — the intended physical benefit.
3. **Racing line shortens path**: 44.8m vs 46.5m for plain min_snap.
4. **Sim performance is bottlenecked by PID controller**, not trajectory quality. Aggressive polynomial accelerations push the drone off-track — controller tuning is the next priority.
5. **Replanning always uses cubic_spline** for real-time speed; trajectory method only affects initial plan before first replan at 0.5s.

## Phase 3 Results — Control System

### Bug Fix: `_accel_to_command` Thrust Mapping

The Phase 2 bottleneck was a **critical thrust mapping bug**. The simplified physics uses `accel_z = (thrust - 0.5) * 20 - 9.81`, requiring thrust=0.99 for hover. The old controller computed `thrust = norm(accel + gravity) / 20`, which gave thrust=0.49 for hover — the drone effectively free-fell.

**Fix**: Added `physics_mode` dispatch in `DroneController`:
- `_accel_to_command_simplified` — exact inverse of gym_env: `thrust = (accel_z + 9.81) / 20 + 0.5`, `pitch_rate = accel_x * 2`, `roll_rate = -accel_y * 2`
- `_accel_to_command_full` — corrected attitude-based mapping: `thrust = norm(total_accel) / (2 * 9.81)` (hover = 0.50)

### Simplified Physics Constraints

The simplified physics has highly asymmetric z-axis control:
- **Max climb**: thrust=1.0 → accel_z = +0.19 m/s² (very slow)
- **Max descent**: thrust=0.0 → accel_z = -19.81 m/s² (free-fall)
- **Lateral**: pitch/roll rate ±8 → accel_x/y = ±4.0 m/s² (decoupled from z)

Trajectory feedforward z-acceleration (cubic_spline peaks at ±57 m/s²) is far beyond what the physics can deliver. The controller zeros z-feedforward in simplified mode and lets PD handle altitude, preventing unrecoverable dives from aggressive trajectories.

### New Features

| Feature | Details |
|---------|---------|
| **Physics mode dispatch** | `simplified` (default) or `full` — selected via `control.physics_mode` config |
| **Acceleration limiting** | Component-wise clamp to `max_accel` (default 15.0 m/s²) |
| **Gain scheduling** | Speed-dependent: kp×0.6, kd×1.6, ki×0.2 at max speed (configurable) |
| **MPC controller** | Linear QP on simplified model, horizon=10, solve <5ms, warm-start |

### MPC Controller Architecture

`SimplifiedLinearModel` — discrete LTI matching gym_env:
- State: `x = [pos(3), vel(3)]` (6D), Control: `u = [thrust, roll_rate, pitch_rate]` (3D)
- `x_{k+1} = A*x_k + B*u_k + d` with drag=0.99, dt=0.02, gravity offset in `d`
- Batch prediction: `X = S*x0 + T*U + W` for horizon N

`MPCController` — unconstrained QP with post-clipping:
- Precomputed `H = 2*(T^T*Q*T + R)` and `H_inv` at init
- Per-step: `U = -H_inv * f` where `f = 2*(T^T*Q*e - R*U_ref)`
- Bounds: thrust [0,1], rates [-8,8], applied via clipping
- Fallback: feedforward command from first trajectory point on solve failure
- Warm-start: shifted previous solution

### Test Results

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_controller.py` | 10 (3 original + 7 new) | All pass |
| `test_mpc_controller.py` | 8 (new) | All pass |
| Full suite | 164 | All pass |

### Key Findings

1. **Thrust mapping was the root cause**: old controller gave thrust=0.49 for hover in simplified physics (needs 0.99). Drone was in constant free-fall, masked by always-saturated thrust=1.0 from large PD outputs.
2. **Simplified physics z-axis is nearly unusable for aggressive trajectories**: max climb is +0.19 m/s². Any trajectory requiring >0.19 m/s² upward acceleration is physically impossible. Z-feedforward must be zeroed; PD handles altitude.
3. **MPC solves in <1ms** for horizon=10 with precomputed matrices. Ready for real-time use.
4. **Gain scheduling reduces overshoot at speed**: kp drops 40%, kd increases 60%, ki drops 80% at max speed.

## Phase 4 Results — DCL Submission & Robustness

### Full Dynamics Controller

Fixed `_accel_to_command_full()` with Mellinger-style cascaded position→attitude controller:
- PD on attitude error with configurable gains (kp=8, kd=2) loaded from config
- `use_dynamics_model` flows from config through `run_sim.py` to `DroneRacingEnv`
- Full dynamics PID gain preset added to `default.yaml` (commented out)
- `scripts/tune_full_dynamics.py` for parametric gain sweeps

### DCL Submission Interface

New `src/aigrandprix/submission/` module for competition integration:

| Component | Purpose |
|-----------|---------|
| `entry_point.py` | `AutonomousRacer` facade — single entry point for DCL platform |
| `dcl_adapter.py` | DCL platform adapter stub (to be filled when API is released) |
| `type_converters.py` | Converts between internal and DCL observation/command/gate formats |
| `safety.py` | `SafeActionWrapper` — latency tracking, exception catching, hover fallback |

- `current_gate_index` param added to `RacingAgent.compute_action()` for DCL gate sync
- `configs/dcl_submission.yaml` created for submission-specific settings

### Robustness & Stress Testing

- **Stress tests**: parametric gates × seeds matrix, memory stability, latency P95
- **Robustness tests**: gate miss recovery, close gates, sharp turns, altitude changes
- **Perception tests**: size/brightness sweep, no false positives, fallback chain verification
- **Gate detector hardened**: fallback chain with monitoring properties
- `scripts/benchmark_gate_completion.py` for failure mode classification

### Test Results

| Test File | Tests | Status |
|-----------|-------|--------|
| Stress tests | 12 (new) | All pass |
| Robustness tests | 15 (new) | All pass |
| Perception robustness | 10 (new) | All pass |
| Submission module | 19 (new) | All pass |
| Full suite | 220 | All pass |

## Phase 5 Results — 100% Gate Completion

### Optimizations

| Area | Change |
|------|--------|
| **Gate closing mode** | Activation range 2.5m→5.0m, distance-scaled approach speed (dist×1.5) |
| **Altitude safety** | Thrust floor 0.75, descent intervention, z-vel error clamping, floor protection at z=0.8 |
| **Path planner** | Min altitude floor (z ≥ 0.8), reduced replan approach/exit distances |
| **Trajectory following** | Advance past passed waypoints, force replan on gate pass, altitude boost |

### Benchmark Results: 200/200 Episodes, 100% Gate Completion

| Scenario | Episodes | Gates/Episode | Gates Passed | Crashes | Timeouts |
|----------|----------|---------------|--------------|---------|----------|
| Standard | 50 | 5 | 250/250 | 0 | 0 |
| Dense | 50 | 8 | 400/400 | 0 | 0 |
| Long course | 100 | 10 | 1000/1000 | 0 | 0 |
| **Total** | **200** | — | **1650/1650** | **0** | **0** |

### Test Results

| Full suite | 243 tests | All pass |
|-----------|-------|--------|

## Current Status & Remaining Work

### Phase 5 Legacy (gym_env internal sim)
- 100% gate completion in simplified physics (200/200 episodes, 0 crashes) — **does not transfer to DCL**
- PID + MPC controllers work in gym_env mode
- 243 tests passing (preserved as regression suite)

### Phase 6 — VQ1 Readiness (deadline 2026-05-23)

**CRITICAL GAP**: Internal stack is structurally incompatible with real DCL interface (MAVLink2/UDP, NED, no GPS, mandatory vision). Full rebuild required.

See `NEXT_STEPS.md` for full 6-track plan. Summary:

| Track | Status | Deadline |
|-------|--------|----------|
| A — MAVLink Client | Not started | 2026-05-20 |
| B — Mock DCL Sim | Not started | 2026-05-21 |
| C — NED Refactor + State Estimation | Not started | 2026-05-20 |
| D — Vision CNN (YOLOv8n) | Not started | 2026-05-22 |
| E — Planner/Controller Adapter (NED + POSITION_TARGET) | Not started | 2026-05-22 |
| F — Integration + Windows CI | Not started | 2026-05-23 |

**Sim DCL binary**: not yet released (as of 2026-05-17). Strategy: build against mock, swap IP when binary releases.

### What Remains (ordered by priority)
1. **MAVLink client** (pymavlink): heartbeat, recv ATTITUDE+HIGHRES_IMU, recv vision UDP 5600, send POSITION_TARGET_LOCAL_NED
2. **Mock DCL sim**: faithful VADR-TS-002 server — only way to validate before official binary
3. **NED refactor**: all coordinates ENU→NED across planning + control + perception
4. **State estimator**: pos from vel integration, yaw from quat, no GPS
5. **Vision CNN**: train YOLOv8n on synthetic 640×360 gates with +20° tilt. Fallback: HSV blue detector
6. **run_vq1.py**: single entry point `python run_vq1.py --host X --port Y`
7. **Windows 11 / Python 3.14.2** validated environment
