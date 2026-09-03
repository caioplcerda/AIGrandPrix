# AI Grand Prix — Development Roadmap

> **Goal**: Win the AI Grand Prix autonomous drone racing competition (May 2026 qualifier)
> **Competitor**: Caio de Paula Lacerda
> **Last updated**: 2026-02-14

## Current State

**Working**: PID controller, cubic spline planner, color-based gate detector, Gymnasium env, racing agent loop, simulation runner with visualization, 43 passing tests.

**Missing**: CNN perception, RL training pipeline, trajectory optimization, MPC, drone dynamics model, state estimator, depth estimation, sim-to-real strategy.

---

## Phase 0 — Foundation (Weeks 1–2)

Critical fixes to make the existing stack RL-ready and extensible.

- [ ] **Normalize Gymnasium observation/action spaces for SB3 compatibility**
  - File: `src/aigrandprix/simulation/gym_env.py`
  - Currently: observation is Box(19) with mixed ranges, action space partially normalized
  - Do: Normalize observations to [-1, 1] or [0, 1], ensure action space is [-1, 1]⁴ with internal rescaling
  - Acceptance: `stable_baselines3.common.env_checker.check_env()` passes cleanly, PPO trains without NaN

- [ ] **Create abstract simulator interface**
  - New file: `src/aigrandprix/simulation/sim_interface.py`
  - Define `SimInterface` ABC with methods: `reset()`, `step()`, `get_image()`, `get_state()`, `get_gate_poses()`
  - Implement `InternalSimAdapter` wrapping `DroneRacingEnv`
  - Purpose: Swap between internal sim, PyBullet, Colosseum, and DCL platform without touching agent code
  - Acceptance: `RacingAgent` works through the interface, not directly with gym env

- [ ] **Build quadrotor dynamics model**
  - New file: `src/aigrandprix/control/drone_dynamics.py`
  - Implement Newton-Euler rigid body dynamics: mass, inertia tensor, motor thrust curves, drag coefficients
  - Use `DroneParams` dataclass: mass, arm_length, Ixx/Iyy/Izz, k_thrust, k_drag, motor_time_constant
  - Acceptance: Given motor RPMs → compute forces/torques → integrate to next state, matches PyBullet within 5%

- [ ] **Wire YAML config loading into all modules**
  - Files: `src/aigrandprix/main.py`, all module constructors
  - Currently: `configs/default.yaml` exists but modules use hardcoded defaults
  - Do: Load config at startup, pass relevant sections to each module constructor
  - Acceptance: Changing `configs/default.yaml` values changes runtime behavior without code edits

- [ ] **Replace simplified physics in gym_env with dynamics model**
  - File: `src/aigrandprix/simulation/gym_env.py`
  - Currently: Euler-angle integration with magic damping constants (0.98, 0.99)
  - Do: Use `drone_dynamics.py` for state propagation, keep simplified mode as fallback
  - Acceptance: Drone responds realistically to control inputs; gravity, drag, motor lag all modeled

- [ ] **Add proper quaternion math utilities**
  - New file: `src/aigrandprix/utils/math_utils.py`
  - Functions: `quat_multiply`, `quat_rotate`, `quat_to_euler`, `euler_to_quat`, `quat_to_rotation_matrix`
  - Currently: scattered quaternion operations in controller and env
  - Acceptance: All quaternion ops in codebase use shared utilities, no gimbal lock issues

---

## Phase 1 — Perception (Weeks 3–5)

Gate detection is the eyes of the system. Must work at 50+ FPS for racing speed.

- [ ] **Implement CNN gate detector (YOLOv8-nano or GateNet)**
  - File: `src/aigrandprix/perception/gate_detector.py` (replace `_detect_cnn` stub)
  - Architecture options: (a) YOLOv8-nano fine-tuned on gate images, (b) custom GateNet from open-airlab repo
  - Must output: bounding box, center pixel, confidence, gate class
  - Target: >95% detection rate at 60+ FPS on single GPU
  - Reference: `open-airlab/GateNet`, `uzh-rpg/deep_drone_racing`
  - Acceptance: Detects gates in synthetic images with >90% mAP, runs <15ms per frame

- [ ] **Build synthetic training data pipeline**
  - New file: `src/aigrandprix/perception/data_generator.py`
  - Generate gate images with varied: lighting, angle, distance, background, motion blur
  - Output: COCO-format annotations (bounding boxes + keypoints for corners)
  - Use Colosseum or Unity (Flightmare) for photorealistic renders
  - Acceptance: Generate 10k+ training images with labels, train detector to >85% mAP

- [ ] **Implement PnP-based depth/pose estimation**
  - New file: `src/aigrandprix/perception/depth_estimation.py`
  - Use detected gate corners + known gate dimensions → `cv2.solvePnP` → 3D gate pose
  - Replace current heuristic: `distance = focal_length * gate_real_size / apparent_size`
  - Need camera intrinsics (fx, fy, cx, cy) — parameterize for DCL drone camera
  - Acceptance: Distance error <10% at 1-10m range, orientation error <5° with clean corners

- [ ] **Implement Extended Kalman Filter state estimator**
  - New file: `src/aigrandprix/perception/state_estimator.py`
  - State vector: [position(3), velocity(3), orientation(4), angular_velocity(3), accel_bias(3)]
  - Prediction: IMU-driven (accel + gyro)
  - Update: Gate detections (PnP pose), optionally visual odometry
  - Reference: Standard MEKF (multiplicative EKF for quaternions)
  - Acceptance: Position estimate within 0.1m RMS in simulation, smooth even with noisy inputs

- [ ] **Add temporal gate tracking**
  - File: `src/aigrandprix/perception/gate_detector.py`
  - Track gates across frames with simple Kalman filter or Hungarian matching
  - Predict gate position when momentarily occluded or out of frame
  - Acceptance: Maintain gate tracks for 0.5s without detection, no ID switches during approach

---

## Phase 2 — Trajectory Planning (Weeks 5–7)

Replace cubic spline with competition-grade trajectory generation.

- [ ] **Implement minimum-jerk trajectory (5th-order polynomial)**
  - New file: `src/aigrandprix/planning/trajectory_opt.py`
  - Solve boundary-value problem: match position, velocity, acceleration at each gate
  - Closed-form solution for point-to-point segments
  - Reference: `microsoft/AirSim-NeurIPS2019-Drone-Racing` (contains working implementation)
  - Acceptance: Trajectory through 5 gates is C² continuous, jerk bounded, 2x faster than cubic spline

- [ ] **Implement minimum-snap trajectory (7th-order polynomial)**
  - File: `src/aigrandprix/planning/trajectory_opt.py`
  - QP formulation: minimize ∫snap²dt subject to waypoint constraints
  - Use corridor constraints to keep trajectory within safe bounds
  - Reference: `ethz-asl/mav_trajectory_generation` (Richter et al. 2016)
  - Acceptance: Trajectory is C³ continuous, smoother than min-jerk at high speeds, solve time <50ms

- [ ] **Implement time-optimal trajectory allocation**
  - File: `src/aigrandprix/planning/trajectory_opt.py`
  - Iteratively adjust segment times to minimize total lap time
  - Respect dynamic constraints: max velocity, max acceleration, max angular rate
  - Method: Either gradient-based optimization or bisection on segment times
  - Acceptance: Lap time improves >15% vs equal-time allocation

- [ ] **Implement racing line optimization**
  - New file: `src/aigrandprix/planning/race_strategy.py`
  - Optimize gate crossing points (not just gate center — exploit full gate aperture)
  - Use gate normal vectors to find optimal entry/exit angles
  - Consider: cutting corners, optimal speed at each gate, banking angles
  - Acceptance: Racing line is measurably faster than center-of-gate trajectory

- [ ] **Replace cubic spline planner with polynomial trajectories**
  - File: `src/aigrandprix/planning/path_planner.py`
  - Swap `_generate_smooth_trajectory()` from scipy CubicSpline to min-jerk/min-snap
  - Keep cubic spline as fallback for quick replanning
  - Acceptance: Existing tests pass, trajectories are smoother, agent navigates faster

---

## Phase 3 — Reinforcement Learning (Weeks 7–11)

Train an RL policy that outperforms the classical stack.

- [ ] **Implement reward shaping module**
  - New file: `src/aigrandprix/simulation/reward_shaping.py`
  - Implement configurable reward components:
    - Gate passage: +100 (scaled by speed through gate)
    - Progress: distance reduction to next gate (potential-based, Ng 1999)
    - Time penalty: -0.01/step (encourage speed)
    - Smoothness: penalize jerk (∝ ||da/dt||²)
    - Alignment: reward facing next gate
    - Crash: -200
  - All weights configurable via YAML
  - Acceptance: Reward function is potential-based (guarantees optimal policy invariance), agent learns gate navigation

- [ ] **Create PPO training script**
  - New file: `scripts/train_ppo.py`
  - Use `stable_baselines3.PPO` with tuned hyperparameters
  - Features: WandB logging, checkpoint saving, evaluation callback, learning rate schedule
  - Network: MLP [256, 256] or [512, 256] with layer norm
  - Hyperparameters baseline: lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95
  - Acceptance: Agent learns to pass ≥3 gates consistently after 1M steps

- [ ] **Create SAC training script**
  - New file: `scripts/train_sac.py`
  - Use `stable_baselines3.SAC` — better for continuous control
  - Compare with PPO; SAC often better for drone control due to entropy regularization
  - Acceptance: Agent matches or exceeds PPO gate passage rate

- [ ] **Implement curriculum learning**
  - File: `src/aigrandprix/simulation/gym_env.py` or new wrapper
  - Stages: (1) hover → (2) fly to single gate → (3) 3 gates → (4) full course → (5) randomized courses
  - Auto-advance when success rate >80% over 100 episodes
  - Acceptance: Agent trained with curriculum reaches 5-gate completion 2x faster than without

- [ ] **Implement hybrid RL+classical policy**
  - New file: `src/aigrandprix/planning/hybrid_policy.py`
  - RL provides high-level waypoints or velocity commands
  - Classical controller (PID/MPC) provides low-level motor commands
  - This is the architecture that won in the Swift paper (Nature 2023)
  - Acceptance: Hybrid policy completes courses faster than pure RL or pure classical

- [ ] **Multi-environment parallel training**
  - File: `scripts/train_ppo.py`
  - Use `stable_baselines3.common.vec_env.SubprocVecEnv` with 8-16 parallel envs
  - Significantly speeds up training wallclock time
  - Acceptance: Training throughput >10k steps/sec on available hardware

- [ ] **Domain randomization for sim-to-real**
  - File: `src/aigrandprix/simulation/gym_env.py`
  - Randomize: gate colors/textures, lighting, drone mass (±10%), motor response, wind, sensor noise
  - Acceptance: Policy trained with randomization transfers to Colosseum/PyBullet without fine-tuning

---

## Phase 4 — Advanced Control (Weeks 9–12)

Upgrade from PID to competition-grade control.

- [ ] **Implement Model Predictive Controller (MPC)**
  - New file: `src/aigrandprix/control/mpc_controller.py`
  - Formulation: Minimize tracking error + control effort over N-step horizon
  - Use `drone_dynamics.py` as prediction model
  - Solver: CasADi + IPOPT or ACADOS for real-time performance
  - Horizon: 20 steps at 100Hz = 0.2s lookahead
  - Reference: `uzh-rpg/rpg_mpc`
  - Acceptance: MPC tracks aggressive trajectories with <0.1m error at 10m/s, runs at 100Hz

- [ ] **PID auto-tuning via CMA-ES**
  - New file: `scripts/tune_pid.py`
  - Optimize PID gains (Kp, Ki, Kd for each axis) to minimize lap time in simulation
  - Use CMA-ES (covariance matrix adaptation): population-based, gradient-free, good for 18-dim space
  - Objective: weighted sum of tracking error + lap time + smoothness
  - Acceptance: Tuned gains improve lap time >10% over hand-tuned defaults in `configs/default.yaml`

- [ ] **Implement INDI (Incremental Nonlinear Dynamic Inversion) controller**
  - New file: `src/aigrandprix/control/indi_controller.py`
  - More robust than PID for aggressive maneuvers, used in many racing drones
  - Needs angular acceleration measurement (from gyro derivative)
  - Reference: Used in `uzh-rpg/agile_flight` (Swift)
  - Acceptance: Tracks step responses faster than PID with less overshoot

- [ ] **Motor mixing and allocation**
  - File: `src/aigrandprix/control/drone_dynamics.py`
  - Implement proper motor mixing matrix: [thrust, roll, pitch, yaw] → [motor1, motor2, motor3, motor4]
  - Handle motor saturation with prioritized allocation (altitude > attitude)
  - Acceptance: Motor commands respect physical limits, no clipping artifacts in aggressive flight

---

## Phase 5 — Integration & Competition Prep (Weeks 12–14)

Polish, benchmark, and prepare for the DCL platform submission.

- [ ] **Build benchmark suite**
  - New file: `scripts/benchmark.py`
  - Metrics: lap time, gate success rate, avg speed, max speed, tracking error, control smoothness
  - Run: classical stack vs RL vs hybrid on 10 random courses, 100 episodes each
  - Output: comparison table, statistical significance tests
  - Acceptance: Clear winner identified with confidence intervals

- [ ] **PyBullet integration**
  - New file: `src/aigrandprix/simulation/pybullet_env.py`
  - Wrap `gym-pybullet-drones` as a `SimInterface` implementation
  - More realistic physics than internal sim (proper rotor dynamics, ground effect, etc.)
  - Reference: `utiasDSL/gym-pybullet-drones`
  - Acceptance: Agent trained in internal sim achieves >70% gate completion in PyBullet without retraining

- [ ] **Colosseum (AirSim) integration**
  - New file: `src/aigrandprix/simulation/colosseum_env.py`
  - Photorealistic rendering for perception testing
  - Reference: `CodexLabsLLC/Colosseum`
  - Acceptance: Gate detector works on Colosseum-rendered images, agent completes courses

- [ ] **DCL platform adapter**
  - New file: `src/aigrandprix/simulation/dcl_adapter.py`
  - Implement `SimInterface` for the DCL competition platform (API TBD)
  - Handle: their observation format, their action format, their timing constraints
  - Acceptance: `RacingAgent` runs on DCL platform, completes at least one gate

- [ ] **Sim-to-real transfer validation**
  - Run same policy across: internal sim, PyBullet, Colosseum
  - Measure performance degradation at each transfer
  - Apply domain randomization if degradation >20%
  - Acceptance: <15% lap time degradation across simulators

- [ ] **End-to-end latency optimization**
  - Profile full pipeline: perception → planning → control
  - Target: <10ms total latency at 100Hz control rate
  - Optimize bottlenecks: batch inference, precompute trajectories, cache gate detections
  - Acceptance: Control loop runs at ≥100Hz with perception at ≥30Hz

- [ ] **Competition submission package**
  - Verify all code runs in Python 3.10+ with allowed dependencies
  - Create `submission/` directory with clean entry point
  - Document: what it does, how to run, any known limitations
  - Test on clean Python environment (Docker)
  - Acceptance: `python -m aigrandprix.main` runs without errors on fresh install

---

## Phase 6 — Stretch Goals (If Time Permits)

- [ ] Visual odometry for GPS-denied environments
- [ ] Opponent avoidance for head-to-head racing (Finals)
- [ ] Dynamic gate sequence optimization (if gates can be taken in any order)
- [ ] Neural network distillation for faster inference
- [ ] Learned residual dynamics model (compensate for sim-to-real gap)
- [ ] Multi-agent racing strategy for finals
- [ ] Online system identification (adapt to actual drone parameters)

---

## Dependency Graph

```
Phase 0 (Foundation)
  ├─→ Phase 1 (Perception)    ─→ Phase 5 (Integration)
  ├─→ Phase 2 (Planning)      ─→ Phase 5
  ├─→ Phase 3 (RL Training)   ─→ Phase 5
  └─→ Phase 4 (Control)       ─→ Phase 5
```

Phase 0 is a hard prerequisite. Phases 1–4 can be parallelized across team members.
Phase 5 requires all prior phases to be substantially complete.

---

## Quick Reference: File Map

| File | Status | Phase |
|------|--------|-------|
| `src/aigrandprix/main.py` | ✅ Working | — |
| `src/aigrandprix/perception/gate_detector.py` | ⚠️ CNN stubbed | Phase 1 |
| `src/aigrandprix/perception/depth_estimation.py` | ❌ Not created | Phase 1 |
| `src/aigrandprix/perception/state_estimator.py` | ❌ Not created | Phase 1 |
| `src/aigrandprix/perception/data_generator.py` | ❌ Not created | Phase 1 |
| `src/aigrandprix/control/drone_controller.py` | ✅ Working (PID) | — |
| `src/aigrandprix/control/drone_dynamics.py` | ❌ Not created | Phase 0 |
| `src/aigrandprix/control/mpc_controller.py` | ❌ Not created | Phase 4 |
| `src/aigrandprix/control/indi_controller.py` | ❌ Not created | Phase 4 |
| `src/aigrandprix/planning/path_planner.py` | ✅ Working (cubic spline) | — |
| `src/aigrandprix/planning/trajectory_opt.py` | ❌ Not created | Phase 2 |
| `src/aigrandprix/planning/race_strategy.py` | ❌ Not created | Phase 2 |
| `src/aigrandprix/planning/hybrid_policy.py` | ❌ Not created | Phase 3 |
| `src/aigrandprix/simulation/gym_env.py` | ✅ Working | Phase 0 (normalize) |
| `src/aigrandprix/simulation/sim_interface.py` | ❌ Not created | Phase 0 |
| `src/aigrandprix/simulation/reward_shaping.py` | ❌ Not created | Phase 3 |
| `src/aigrandprix/simulation/pybullet_env.py` | ❌ Not created | Phase 5 |
| `src/aigrandprix/simulation/colosseum_env.py` | ❌ Not created | Phase 5 |
| `src/aigrandprix/simulation/dcl_adapter.py` | ❌ Not created | Phase 5 |
| `src/aigrandprix/utils/math_utils.py` | ❌ Not created | Phase 0 |
| `scripts/train_ppo.py` | ❌ Not created | Phase 3 |
| `scripts/train_sac.py` | ❌ Not created | Phase 3 |
| `scripts/tune_pid.py` | ❌ Not created | Phase 4 |
| `scripts/benchmark.py` | ❌ Not created | Phase 5 |
| `configs/default.yaml` | ✅ Exists | Phase 0 (wire up) |
