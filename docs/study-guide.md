# AI Grand Prix — Study Guide

> Comprehensive reference for building a competition-winning autonomous drone racing system.
> Organized by topic, with theory, papers, code references, and implementation notes.

---

## Table of Contents

1. [Drone Physics & Dynamics](#1-drone-physics--dynamics)
2. [Control Theory for Quadrotors](#2-control-theory-for-quadrotors)
3. [Computer Vision for Gate Racing](#3-computer-vision-for-gate-racing)
4. [Path Planning & Trajectory Optimization](#4-path-planning--trajectory-optimization)
5. [Reinforcement Learning for Drone Racing](#5-reinforcement-learning-for-drone-racing)
6. [Key Papers (Priority Reading List)](#6-key-papers-priority-reading-list)
7. [Key Repositories (What to Study & Extract)](#7-key-repositories-what-to-study--extract)
8. [Competition Strategy](#8-competition-strategy)

---

## 1. Drone Physics & Dynamics

### Core Concepts

**Newton-Euler Equations for a Quadrotor**

A quadrotor is a rigid body with 6 degrees of freedom (3 position + 3 orientation) actuated by 4 rotors.

```
Translational dynamics (world frame):
  m * a = R * [0, 0, F_total]^T + [0, 0, -m*g]^T + F_drag

Rotational dynamics (body frame):
  I * α + ω × (I * ω) = τ

Where:
  m     = mass (typically 0.5–2.0 kg for racing drones)
  a     = linear acceleration (world frame)
  R     = rotation matrix (body → world)
  F_total = sum of all motor thrusts
  g     = 9.81 m/s²
  F_drag = -k_drag * v (aerodynamic drag)
  I     = inertia tensor (diagonal for symmetric quadrotor)
  α     = angular acceleration
  ω     = angular velocity (body frame)
  τ     = torque vector from motor differential thrust
```

**Motor Model**

Each motor produces thrust proportional to the square of its angular velocity:

```
F_i = k_f * ω_i²     (thrust)
τ_i = k_m * ω_i²     (reaction torque)

Where:
  k_f = thrust coefficient (N/(rad/s)²)
  k_m = moment coefficient (Nm/(rad/s)²)
  ω_i = motor angular velocity
```

**Motor Mixing Matrix**

Maps desired [thrust, roll_torque, pitch_torque, yaw_torque] to individual motor speeds:

```
For "X" configuration (standard racing quad):

  ω₁² = (F + τ_roll - τ_pitch - τ_yaw) / (4 * k_f)    (front-right)
  ω₂² = (F - τ_roll - τ_pitch + τ_yaw) / (4 * k_f)    (front-left)
  ω₃² = (F - τ_roll + τ_pitch - τ_yaw) / (4 * k_f)    (rear-left)
  ω₄² = (F + τ_roll + τ_pitch + τ_yaw) / (4 * k_f)    (rear-right)
```

**Motor Dynamics**

Motors are not instant — they have a first-order lag:

```
dω/dt = (ω_desired - ω_current) / τ_motor

τ_motor ≈ 0.01–0.05s for racing motors (very fast)
```

**Key Parameters to Know**

| Parameter | Typical Range | Impact |
|-----------|--------------|--------|
| Mass | 0.5–2.0 kg | Thrust-to-weight ratio |
| Arm length | 0.1–0.3 m | Torque authority |
| I_xx, I_yy | 0.001–0.01 kg·m² | Roll/pitch responsiveness |
| I_zz | 0.002–0.02 kg·m² | Yaw responsiveness |
| k_f | 1e-6 – 1e-5 | Thrust per motor speed² |
| k_drag | 0.01–0.1 | Air resistance |
| Max RPM | 20,000–40,000 | Maximum thrust |

**Aerodynamic Effects at High Speed**

- **Blade flapping**: Thrust varies with forward velocity
- **Ground effect**: Increased thrust near ground (< 1 body length)
- **Rotor wash interaction**: Rotors affect each other at high angles
- **Parasitic drag**: Increases with v² — dominates at racing speeds

### What to Implement

- `DroneParams` dataclass with all physical parameters → `src/aigrandprix/control/drone_dynamics.py`
- Forward dynamics: (state, motor_commands) → next_state
- Use as prediction model in MPC and as simulation physics in `gym_env.py`

### Recommended Reading

- Mellinger & Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors" (ICRA 2011) — Sections 2-3 cover dynamics
- Beard & McLain, "Small Unmanned Aircraft" — Chapter 4 (Forces and Moments)
- `utiasDSL/gym-pybullet-drones` source code — Clean Python implementation of quadrotor dynamics

---

## 2. Control Theory for Quadrotors

### PID Cascade Control (Current Implementation)

Our current controller in `src/aigrandprix/control/drone_controller.py` uses a cascade PID:

```
Position Loop (outer):
  error_pos = target_pos - current_pos
  desired_vel = Kp_pos * error_pos + Ki_pos * ∫error_pos + Kd_pos * d(error_pos)/dt

Velocity Loop (middle):
  error_vel = desired_vel - current_vel
  desired_accel = Kp_vel * error_vel + ...

Attitude Loop (inner):
  desired_angles = accel_to_angles(desired_accel)  ← flight physics conversion
  error_angle = desired_angle - current_angle
  rate_command = Kp_att * error_angle + ...
```

**Current PID Gains** (from `configs/default.yaml`):
- Kp = [6.0, 6.0, 8.0]
- Ki = [0.1, 0.1, 0.2]
- Kd = [3.5, 3.5, 4.5]

**Common PID Issues and Fixes**:

| Issue | Symptom | Fix |
|-------|---------|-----|
| Derivative kick | Spikes on setpoint change | Differentiate process variable, not error |
| Integral windup | Overshoot after saturation | Clamp integral term, or back-calculation |
| Noise amplification | Jittery control | Low-pass filter on derivative term |
| Setpoint weighting | Aggressive on step changes | Weight setpoint in P term (β * setpoint - measurement) |

### Model Predictive Control (MPC)

MPC is the competition-grade controller. It optimizes a sequence of future actions over a prediction horizon.

**Formulation**:

```
minimize  Σ_{k=0}^{N-1} [ (x_k - x_ref_k)^T Q (x_k - x_ref_k) + u_k^T R u_k ]
          + (x_N - x_ref_N)^T Q_f (x_N - x_ref_N)

subject to:
  x_{k+1} = f(x_k, u_k)     ← drone dynamics
  u_min ≤ u_k ≤ u_max        ← motor limits
  x_k ∈ X                    ← state constraints (no crash)

Where:
  N     = prediction horizon (10-30 steps)
  x_k   = state at step k [pos, vel, orientation, angular_vel]
  u_k   = control input at step k [motor commands]
  Q     = state cost matrix (tracking error weight)
  R     = control cost matrix (control effort weight)
  Q_f   = terminal cost matrix
  f()   = drone dynamics model
```

**Implementation Approach**:

1. **Linearize** dynamics around current state → get A, B matrices
2. **Solve QP** at each timestep (fast for linear MPC)
3. **Apply first control**, shift horizon, repeat

**Solver Options**:
- **CasADi + IPOPT**: General nonlinear, flexible, Python-friendly
- **ACADOS**: Purpose-built for MPC, very fast (C-generated code)
- **OSQP**: Fast QP solver for linear MPC

**Tuning MPC**:
- Horizon N: 20 steps at 100Hz = 0.2s lookahead (good balance)
- Q: High weight on position tracking (diag [100, 100, 100, 10, 10, 10, ...])
- R: Small weight on control effort (diag [0.01, 0.01, 0.01, 0.01])
- Q_f: Solve discrete-time algebraic Riccati equation (DARE) for terminal cost

### LQR (Linear Quadratic Regulator)

Simpler than MPC, good baseline. Infinite-horizon optimal control for linear systems.

```
u = -K * (x - x_ref)

K = solution of continuous-time algebraic Riccati equation
```

Useful as: (a) baseline to compare against MPC, (b) terminal controller in MPC, (c) fast fallback controller.

### INDI (Incremental Nonlinear Dynamic Inversion)

Used in the Swift drone that beat human champions:

```
Δu = G⁻¹ * (α_desired - α_measured)

Where:
  G = control effectiveness matrix (how motors affect angular acceleration)
  α = angular acceleration (measured from gyro derivative)
```

**Why INDI for racing**:
- Doesn't need accurate dynamics model (only incremental changes)
- Very robust to parameter uncertainty
- Handles aggressive maneuvers better than PID
- Fast computation (no optimization loop)

### What to Implement

1. MPC controller → `src/aigrandprix/control/mpc_controller.py`
2. PID auto-tuning script → `scripts/tune_pid.py`
3. Optionally INDI controller → `src/aigrandprix/control/indi_controller.py`

### Recommended Reading

- Kamel et al., "Model Predictive Control for Trajectory Tracking of Unmanned Aerial Vehicles Using Robot Operating System" — Clean MPC formulation
- Song et al., "Autonomous Drone Racing with Deep Reinforcement Learning" (RSS 2023, Swift paper) — INDI + RL architecture
- `uzh-rpg/rpg_mpc` — Clean C++ MPC implementation for quadrotors (study the formulation)

---

## 3. Computer Vision for Gate Racing

### Gate Detection Approaches

**Option A: YOLOv8-nano (Recommended First Approach)**

- Pre-trained on COCO, fine-tune on gate images
- Outputs bounding boxes + confidence per frame
- YOLOv8-nano: ~3M parameters, ~2ms inference on GPU
- Use Ultralytics library: `from ultralytics import YOLO`

```python
# Training
model = YOLO('yolov8n.pt')
model.train(data='gates.yaml', epochs=100, imgsz=640)

# Inference
results = model(frame)
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    confidence = box.conf[0]
```

**Option B: GateNet (Specialized Architecture)**

- From `open-airlab/GateNet` — purpose-built for drone racing gates
- Predicts gate corners (4 keypoints) directly, not just bounding box
- Better for PnP pose estimation (corners are more precise)
- Smaller dataset needed due to domain-specific architecture

**Option C: End-to-End (Advanced)**

- From `uzh-rpg/deep_drone_racing` — image directly to control commands
- Skips explicit detection, learns implicit perception
- Harder to debug, but potentially faster
- Good as advanced strategy, not first approach

### Pose Estimation via PnP

Once you detect gate corners, recover 3D pose:

```python
import cv2
import numpy as np

# Known 3D gate corner positions (gate frame, centered)
gate_width, gate_height = 1.0, 1.0  # meters
object_points = np.array([
    [-gate_width/2, -gate_height/2, 0],
    [ gate_width/2, -gate_height/2, 0],
    [ gate_width/2,  gate_height/2, 0],
    [-gate_width/2,  gate_height/2, 0],
], dtype=np.float32)

# Detected 2D image corners (from detector)
image_points = np.array([...], dtype=np.float32)  # 4x2

# Camera intrinsics
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1],
], dtype=np.float32)

# Solve PnP
success, rvec, tvec = cv2.solvePnP(
    object_points, image_points, camera_matrix, dist_coeffs=None,
    flags=cv2.SOLVEPNP_IPPE  # Use IPPE for coplanar points (gates are planar)
)

# tvec = translation (distance to gate)
# rvec = rotation (gate orientation relative to camera)
distance = np.linalg.norm(tvec)
```

**Key: Use IPPE (Infinitesimal Plane-based Pose Estimation)** for planar targets like gates. More stable than P3P for coplanar points.

### Camera Model

```
Pinhole model:
  u = fx * X/Z + cx
  v = fy * Y/Z + cy

Where:
  (u, v) = pixel coordinates
  (X, Y, Z) = 3D point in camera frame
  fx, fy = focal length in pixels
  cx, cy = principal point (usually image center)
```

For the DCL drone camera, we'll need to calibrate these parameters (or they'll be provided). Current placeholder: `fx = fy = 500px`, `cx = cy = 320px`.

### Temporal Filtering

Single-frame detection is noisy. Use temporal filtering:

1. **Kalman Filter on gate position**: Smooth detections, predict during occlusion
2. **Hungarian algorithm**: Match detections across frames (handle multiple gates)
3. **Confidence decay**: If gate not seen for N frames, decay confidence

```python
# Simple exponential moving average for gate position
alpha = 0.3  # smoothing factor
gate_pos_filtered = alpha * gate_pos_detected + (1 - alpha) * gate_pos_prev
```

### Synthetic Data Generation

**Why**: Real gate images are scarce. Synthetic data with domain randomization bridges the gap.

**What to randomize**:
- Gate color/texture (primary: red, but test others)
- Background (sky, indoor, outdoor, cluttered)
- Lighting (direction, intensity, time of day)
- Camera angle (approach from any direction)
- Distance (0.5m to 20m)
- Motion blur (proportional to speed)
- Lens distortion
- Gate occlusion (partial views)

**Tools**: Colosseum/AirSim for photorealistic renders, or programmatic OpenCV for basic augmentation.

### What to Implement

1. CNN detector in `src/aigrandprix/perception/gate_detector.py` (replace `_detect_cnn` stub)
2. PnP depth estimation → `src/aigrandprix/perception/depth_estimation.py`
3. Synthetic data pipeline → `src/aigrandprix/perception/data_generator.py`
4. Temporal tracking in `gate_detector.py`

### Recommended Reading

- Li et al., "GateNet: A Gate Detection Network for Autonomous Drone Racing" (open-airlab)
- Redmon et al., "YOLOv3: An Incremental Improvement" (understand YOLO fundamentals)
- Lepetit et al., "EPnP: Efficient Perspective-n-Point Camera Pose Estimation"
- Collins & Bartoli, "IPPE: Infinitesimal Plane-based Pose Estimation" (best for gate PnP)

---

## 4. Path Planning & Trajectory Optimization

### Current Implementation

`src/aigrandprix/planning/path_planner.py` uses SciPy `CubicSpline` through waypoints (approach point → gate center → exit point). This is functional but suboptimal:

- Cubic splines are only C² continuous (acceleration is continuous, but jerk has discontinuities)
- No time optimization — equal-time segments regardless of distance
- No dynamic feasibility constraints

### Minimum-Jerk Trajectory (5th-order polynomial)

**The simplest upgrade from cubic splines**. Each segment between waypoints is a 5th-order polynomial:

```
p(t) = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵

Boundary conditions per segment:
  Position, velocity, acceleration at start and end = 6 conditions → 6 coefficients ✓
```

**Closed-form solution** (no optimizer needed):

```
Given segment from state_0 = (p₀, v₀, a₀) to state_f = (p_f, v_f, a_f) in time T:

Solve 6x6 linear system:
  [1  0  0   0    0     0  ] [a₀]   [p₀]
  [0  1  0   0    0     0  ] [a₁]   [v₀]
  [0  0  2   0    0     0  ] [a₂] = [a₀]
  [1  T  T²  T³   T⁴    T⁵] [a₃]   [p_f]
  [0  1  2T  3T²  4T³   5T⁴] [a₄]   [v_f]
  [0  0  2   6T   12T²  20T³] [a₅]   [a_f]
```

This gives C² continuous trajectories that minimize jerk (rate of change of acceleration), producing smooth flight.

### Minimum-Snap Trajectory (7th-order polynomial)

**The gold standard for quadrotor trajectory planning**. Minimizing snap (4th derivative of position) minimizes motor effort because:

```
Motor commands ∝ angular acceleration ∝ jerk of position
→ Minimizing snap minimizes rate of change of motor commands
→ Smoother, more efficient flight
```

**QP Formulation** (Richter et al. 2016):

```
minimize   Σ_i ∫₀^{T_i} ||d⁴p_i/dt⁴||² dt

subject to:
  Continuity at waypoints: p, v, a, j match between segments
  Waypoint constraints: p(t_k) = waypoint_k
  Boundary conditions: v, a at start and end

This is a Quadratic Program (QP):
  minimize  c^T H c
  subject to  A c = b

Where:
  c = polynomial coefficient vector
  H = Hessian (from snap cost integral)
  A, b = continuity and waypoint constraints
```

**Solve with**: `scipy.optimize.minimize` (small problems) or `osqp` (large problems).

### Time Allocation

The segment times T_i dramatically affect trajectory quality. Three approaches:

1. **Trapezoidal velocity profile**: Accelerate → cruise → decelerate per segment. Simple, decent baseline.

2. **Iterative refinement**: Start with distance-proportional times, then iteratively adjust:
   ```
   For each segment:
     If max_velocity exceeded → increase T_i
     If max_acceleration exceeded → increase T_i
     If constraints satisfied with margin → decrease T_i (go faster)
   ```

3. **Full time-optimal**: Jointly optimize all T_i to minimize total time subject to dynamic constraints. Non-convex — use gradient-based methods or backtracking search.

### Racing Line Optimization

**Don't fly through gate centers.** Optimal racing lines cut corners:

```
For each gate:
  - Gate has position P_g and normal vector n_g
  - Gate aperture is W × H (e.g., 1.0m × 1.0m)
  - Crossing point can be anywhere within the gate aperture
  - Optimal crossing point minimizes curvature of the trajectory

Strategy:
  1. Compute straight-line path between consecutive gates
  2. Offset crossing point toward the inside of turns
  3. Adjust approach/exit angles to minimize speed loss
  4. Re-optimize trajectory through adjusted crossing points
```

This is analogous to the racing line in motorsport — the apex of each turn.

### What to Implement

1. Minimum-jerk trajectory → `src/aigrandprix/planning/trajectory_opt.py`
2. Minimum-snap trajectory → same file
3. Time allocation optimization → same file
4. Racing line → `src/aigrandprix/planning/race_strategy.py`
5. Replace cubic spline in `src/aigrandprix/planning/path_planner.py`

### Recommended Reading

- **Mellinger & Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors" (ICRA 2011)** — The foundational paper. Read this first.
- **Richter et al., "Polynomial Trajectory Planning for Aggressive Quadrotor Flight in Dense Environments" (ISRR 2016)** — Unconstrained QP formulation, much faster
- Mueller et al., "A Computationally Efficient Motion Primitive for Quadrocopter Trajectory Generation" (TRO 2015) — Closed-form time-optimal primitives
- `ethz-asl/mav_trajectory_generation` — C++ implementation of the above papers (study the math)
- `microsoft/AirSim-NeurIPS2019-Drone-Racing` — Working Python minimum-jerk implementation

---

## 5. Reinforcement Learning for Drone Racing

### Why RL for Racing

Classical control + planning gives a solid baseline, but RL can:
1. **Learn to exploit physics** that are hard to model (ground effect, rotor interactions)
2. **Optimize globally** rather than stage-by-stage (race-level strategy)
3. **Adapt to uncertainty** through domain randomization
4. **Go beyond human intuition** — the Swift drone found trajectories humans never considered

### PPO (Proximal Policy Optimization)

**The default algorithm for drone racing RL.** On-policy, stable, parallelizable.

```
Core idea: Update policy to maximize expected reward, but constrain the update
to stay close to the old policy (prevents catastrophic forgetting).

Loss = min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)

Where:
  r(θ) = π_new(a|s) / π_old(a|s)  (probability ratio)
  A = advantage estimate (from GAE)
  ε = clip parameter (typically 0.2)
```

**Hyperparameters for drone racing** (good starting point):

```yaml
learning_rate: 3e-4
n_steps: 2048          # steps per env before update
batch_size: 64         # minibatch size
n_epochs: 10           # epochs per update
gamma: 0.99            # discount factor
gae_lambda: 0.95       # GAE parameter
clip_range: 0.2        # PPO clip
ent_coef: 0.01         # entropy bonus (exploration)
vf_coef: 0.5           # value function coefficient
max_grad_norm: 0.5     # gradient clipping
```

### SAC (Soft Actor-Critic)

**Better for continuous control** (like drone racing). Off-policy, sample-efficient, entropy-regularized.

```
Core idea: Maximize expected reward + entropy bonus.
The entropy term encourages exploration and produces robust policies.

Objective: maximize E[Σ γ^t (r_t + α * H(π(·|s_t)))]

Where:
  H(π) = entropy of the policy
  α = temperature (auto-tuned to target entropy)
```

**When to use SAC over PPO**:
- Continuous action spaces (drone control is continuous)
- When sample efficiency matters (SAC reuses old data)
- When you want robust, multi-modal policies

### Reward Shaping

**Critical for learning gate racing.** Sparse rewards (only on gate passage) are nearly impossible to learn from.

**Potential-Based Reward Shaping (Ng et al. 1999)**:

```
F(s, s') = γ * Φ(s') - Φ(s)

Where Φ(s) = -distance_to_next_gate

Key property: potential-based shaping preserves the optimal policy.
(Non-potential-based shaping can create local optima!)
```

**Recommended reward structure**:

```python
def compute_reward(state, next_state, action, info):
    reward = 0.0

    # Gate passage (sparse, large)
    if info['gate_passed']:
        reward += 100.0 * (speed_through_gate / max_speed)  # faster = more reward

    # Progress (potential-based)
    dist_now = distance_to_next_gate(next_state)
    dist_prev = distance_to_next_gate(state)
    reward += gamma * (-dist_now) - (-dist_prev)  # = dist_prev - gamma * dist_now

    # Time penalty (encourage speed)
    reward -= 0.01

    # Smoothness penalty (optional)
    jerk = np.linalg.norm(action - prev_action)
    reward -= 0.001 * jerk

    # Alignment bonus (face next gate)
    alignment = np.dot(forward_vector, gate_direction)
    reward += 0.01 * alignment

    # Crash penalty
    if info['crashed']:
        reward -= 200.0

    return reward
```

### Network Architecture

**For drone racing, keep it simple**:

```
Observation → [256] → ReLU → [256] → ReLU → Action (continuous)
                                              Value (scalar)

Use separate value and policy networks for SAC.
Use shared backbone with separate heads for PPO.
```

**Layer normalization** helps with varying observation scales. Don't use batch norm (breaks with small batches in RL).

**Observation space design** (what the network sees):

```
Core (always available):
  - Relative position to next gate (3)
  - Relative velocity (3)
  - Drone orientation as quaternion (4) or rotation matrix (9)
  - Angular velocity (3)
  - Gate normal vector (3)

Optional (if available):
  - Distance to next gate (1)
  - Relative positions to gates N+1, N+2 (6) — look ahead
  - Previous action (4) — helps with smoothness
```

### Curriculum Learning

**Essential for racing.** Don't start with a full course — build up difficulty:

```
Stage 1: Hover at a point (learn basic stability)
  Success: hold position within 0.5m for 3 seconds

Stage 2: Fly to a single gate (learn gate approach)
  Success: pass through gate within 30 seconds

Stage 3: Three gates in a line (learn sequential navigation)
  Success: pass all 3 gates within 60 seconds

Stage 4: Five gates with turns (learn racing)
  Success: pass all 5 gates within 45 seconds

Stage 5: Randomized courses (learn generalization)
  Success: pass all gates in 90% of random courses

Stage 6: Speed optimization (learn to race)
  Success: minimize lap time while maintaining >90% completion
```

Auto-advance when success rate > 80% over 100 episodes.

### Sim-to-Real Transfer

**Domain Randomization** — randomize simulation parameters during training so the policy becomes robust:

```python
randomization_ranges = {
    'mass': (0.9, 1.1),          # ±10% of nominal
    'inertia': (0.9, 1.1),       # ±10%
    'motor_constant': (0.9, 1.1), # ±10%
    'drag_coeff': (0.5, 2.0),    # wide range
    'motor_lag': (0.01, 0.05),   # time constant
    'obs_noise_pos': (0, 0.02),  # meters
    'obs_noise_vel': (0, 0.05),  # m/s
    'obs_noise_ori': (0, 0.02),  # radians
    'action_delay': (0, 2),      # timesteps
    'wind': (-1.0, 1.0),         # m/s per axis
}
```

**System Identification** — measure real drone parameters and update simulation:
- Fly step responses → measure time constants
- Weigh the drone → set mass
- Measure motor curves → set k_f, k_m

### What to Implement

1. Reward shaping module → `src/aigrandprix/simulation/reward_shaping.py`
2. PPO training script → `scripts/train_ppo.py`
3. SAC training script → `scripts/train_sac.py`
4. Curriculum wrapper in `gym_env.py`
5. Hybrid RL+classical policy → `src/aigrandprix/planning/hybrid_policy.py`

### Recommended Reading

- **Schulman et al., "Proximal Policy Optimization Algorithms" (2017)** — PPO paper
- **Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (ICML 2018)** — SAC paper
- **Ng et al., "Policy Invariance Under Reward Transformations" (ICML 1999)** — Reward shaping theory (read this before designing rewards)
- **Song et al., "Reaching the Limit in Autonomous Racing" (Nature 2023)** — Swift: RL + INDI, beat human champions
- **Tobin et al., "Domain Randomization for Sim-to-Real Transfer" (IROS 2017)** — Domain randomization theory

---

## 6. Key Papers (Priority Reading List)

### Tier 1 — Read These First

| # | Paper | Year | What to Extract |
|---|-------|------|----------------|
| 1 | **Song et al., "Reaching the Limit in Autonomous Racing: Autonomous Drone Racing with Deep RL"** (Nature 2023) | 2023 | THE winning architecture: deep RL + INDI controller, curriculum learning, domain randomization. Study the full pipeline. This is what beat human world champions. |
| 2 | **Mellinger & Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors"** (ICRA 2011) | 2011 | Foundation of quadrotor trajectory planning. Understand the QP formulation, differential flatness, and how snap minimization produces motor-efficient trajectories. Implement this. |
| 3 | **Schulman et al., "Proximal Policy Optimization Algorithms"** | 2017 | Understand PPO clipping, GAE, and why it's the default RL algorithm. Focus on hyperparameter sensitivity (Section 5). |
| 4 | **Ng et al., "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping"** (ICML 1999) | 1999 | Potential-based reward shaping. Must read before designing reward functions. Guarantees your shaping doesn't change the optimal policy. |
| 5 | **Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor"** (ICML 2018) | 2018 | SAC algorithm. Understand entropy regularization, automatic temperature tuning, why it works better than PPO for continuous control. |

### Tier 2 — Read for Specific Components

| # | Paper | Year | What to Extract |
|---|-------|------|----------------|
| 6 | **Richter et al., "Polynomial Trajectory Planning for Aggressive Quadrotor Flight in Dense Environments"** (ISRR 2016) | 2016 | Unconstrained QP formulation for minimum-snap. Much faster than Mellinger's constrained version. Implement this for real-time replanning. |
| 7 | **Loquercio et al., "Learning High-Speed Flight in the Wild"** (Science Robotics 2021) | 2021 | Sim-to-real for agile flight using privileged learning. Teacher-student framework. Study the simulation pipeline. |
| 8 | **Kaufmann et al., "Deep Drone Racing: Learning Agile Flight in Dynamic Environments"** (CoRL 2018) | 2018 | End-to-end vision-to-control for gate racing. Useful for understanding the perception → control interface. |
| 9 | **Mueller et al., "A Computationally Efficient Motion Primitive for Quadrocopter Trajectory Generation"** (TRO 2015) | 2015 | Closed-form time-optimal trajectories. Very fast to compute. Good for online replanning. |
| 10 | **Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"** (IROS 2017) | 2017 | Foundations of domain randomization. What to randomize and why. Critical for sim-to-real. |

### Tier 3 — Reference as Needed

| # | Paper | Year | What to Extract |
|---|-------|------|----------------|
| 11 | **Li et al., "GateNet: A Gate Detection Network for Autonomous Drone Racing"** | 2019 | Gate-specific detector architecture. Corner keypoint detection. Dataset generation. |
| 12 | **Foehn et al., "Time-Optimal Planning for Quadrotor Waypoint Flight"** (Science Robotics 2021) | 2021 | Complementary Time-Optimal Planning (CPC). True time-optimal, respects all dynamics. Advanced. |
| 13 | **Yunlong et al., "Learning to Fly in Seconds"** (arXiv 2023) | 2023 | Massively parallel GPU training for drones. Thousands of environments simultaneously. Good for scaling up. |
| 14 | **Bauersfeld et al., "NeuroBEM: Hybrid Aerodynamic Quadrotor Model"** (RSS 2021) | 2021 | Neural network + blade element momentum theory for accurate drag at high speed. Important for sim fidelity. |
| 15 | **Hanover et al., "Performance, Pair-Wise Comparison and Adversarial Drone Racing"** (CoRL 2023) | 2023 | Head-to-head racing strategy. Relevant for the AI Grand Prix finals (November 2026). |

---

## 7. Key Repositories (What to Study & Extract)

### Tier 1 — Study These Thoroughly

#### `uzh-rpg/agile_flight` (Swift)
- **What it is**: The system that beat human drone racing champions (Nature 2023)
- **What to study**:
  - RL training pipeline (PPO + curriculum)
  - INDI controller implementation
  - Reward function design
  - Domain randomization ranges
  - Sim-to-real transfer methodology
- **What to extract**: Training hyperparameters, reward coefficients, controller architecture
- **Language**: Python + C++
- **Key files**: `learning/`, `controller/`

#### `utiasDSL/gym-pybullet-drones`
- **What it is**: Lightweight Gymnasium-compatible drone simulation
- **What to study**:
  - Clean quadrotor dynamics in Python
  - Gymnasium environment structure (observation/action spaces)
  - Multi-drone support
  - RL integration with SB3
- **What to extract**: `BaseAviary` dynamics implementation, motor model, drag model
- **How to use**: Drop-in replacement for our `gym_env.py` for more realistic physics
- **Key files**: `gym_pybullet_drones/envs/BaseAviary.py`

#### `CodexLabsLLC/Colosseum`
- **What it is**: Actively maintained AirSim fork with full drone simulation + vision
- **What to study**:
  - Photorealistic rendering pipeline
  - Camera/sensor simulation
  - API for drone control
- **What to extract**: Perception testing environment, gate rendering setup
- **How to use**: Wrap as `SimInterface` implementation for perception validation

#### `microsoft/AirSim-NeurIPS2019-Drone-Racing`
- **What it is**: Purpose-built for drone racing competitions
- **What to study**:
  - Minimum-jerk trajectory implementation (Python!)
  - Gate detection pipeline
  - Competition submission structure
  - Reward shaping for racing
- **What to extract**: `trajectory_planning/` — minimum jerk Python code, `perception/` — gate detection
- **Key insight**: This was built for a competition just like ours. Study the architecture decisions.

### Tier 2 — Study for Specific Components

#### `utiasDSL/lsy_drone_racing`
- **What to study**: Progressive difficulty levels, crazyflow sim integration, Python 3.10+ compatible
- **What to extract**: Task structure, evaluation methodology, how they handle different difficulty levels

#### `ethz-asl/mav_trajectory_generation`
- **What to study**: C++ implementation of minimum-snap polynomial trajectories
- **What to extract**: Mathematical formulation, constraint handling, time allocation
- **Note**: Code is C++ but the math translates directly to Python

#### `uzh-rpg/flightmare`
- **What to study**: Unity-based rendering for vision RL, high-throughput training
- **What to extract**: Rendering pipeline, how they bridge sim (C++) with RL (Python)

#### `uzh-rpg/agile_autonomy`
- **What to study**: Learning-based agile flight in unknown environments
- **What to extract**: Privileged learning (teacher-student), trajectory sampling strategies

### Tier 3 — Reference for Specialized Needs

| Repository | When to Use | What to Extract |
|-----------|-------------|----------------|
| `open-airlab/GateNet` | Implementing gate detector | Network architecture, training procedure, dataset format |
| `uzh-rpg/deep_drone_racing` | End-to-end approach | Vision-to-control mapping, sim-to-real transfer |
| `ZJU-FAST-Lab/ego-planner` | Online replanning | Gradient-based trajectory optimization in real-time |
| `mit-acl/faster` | Fast replanning | Safe corridor generation, trajectory library approach |
| `arplaboratory/learning-to-fly` | Massively parallel training | GPU-accelerated env, thousands of parallel drones |
| `uzh-rpg/rpg_mpc` | MPC implementation | Quadrotor MPC formulation, solver setup, cost function tuning |
| `phuongboi/drone-racing-using-reinforcement-learning` | Simple RL baseline | Clean PPO for gate racing, good first reference |
| `google-deepmind/mujoco` | Physics engine | Alternative to PyBullet, very fast, JAX-compatible |
| `PEDRA` (aqeelanwar) | RL with Unreal Engine | Environment setup patterns, reward structures |

---

## 8. Competition Strategy

### What Winners Do

Based on studying the Swift paper and previous drone racing competitions:

1. **Perception must be fast and reliable** — even 1 missed gate detection = lost race
2. **Trajectory planning makes the biggest time difference** — optimal racing lines shave seconds
3. **Control robustness matters more than control precision** — aggressive controllers that crash lose
4. **RL beats classical at the limit** — but classical provides a reliable fallback
5. **Sim-to-real is the hardest part** — invest heavily in domain randomization

### Recommended Architecture (Hybrid)

```
┌─────────────────────────────────────────────┐
│                 HIGH LEVEL                   │
│                                              │
│  RL Policy (PPO/SAC)                        │
│  - Trained in simulation                     │
│  - Outputs: velocity commands / waypoints    │
│  - Handles: racing strategy, gate approach   │
│                                              │
├─────────────────────────────────────────────┤
│                 MID LEVEL                    │
│                                              │
│  Trajectory Generator (Min-Snap)             │
│  - Converts RL waypoints to smooth path      │
│  - Ensures dynamic feasibility               │
│  - Handles: replanning, time optimization    │
│                                              │
├─────────────────────────────────────────────┤
│                 LOW LEVEL                    │
│                                              │
│  INDI / MPC Controller                       │
│  - Tracks trajectory at 100Hz+               │
│  - Robust to parameter uncertainty           │
│  - Handles: attitude control, motor mixing   │
│                                              │
└─────────────────────────────────────────────┘
```

**Why this architecture**:
- RL learns *what* to do (strategy), classical control ensures *how* (execution)
- If RL produces unsafe commands, the trajectory generator constrains them
- If perception fails momentarily, the controller continues tracking the last trajectory
- Each layer can be developed and tested independently

### Pacing Strategy (Competition Timeline)

**Now → March 2026 (Weeks 1–4): Foundation**
- Get RL training working end-to-end (even if policy is bad)
- Implement drone dynamics model
- Wire up config system
- Goal: `python scripts/train_ppo.py` runs and produces a policy that improves

**April 2026 (Weeks 5–8): Core Components**
- CNN gate detection working on synthetic data
- Minimum-snap trajectory generation
- MPC or INDI controller replacing PID
- Goal: Agent completes 5-gate courses reliably using classical stack

**Late April 2026 (Weeks 9–11): RL + Integration**
- Train RL policies that beat classical baseline
- Implement hybrid RL+classical architecture
- Curriculum learning and domain randomization
- Goal: RL agent is consistently faster than classical

**Early May 2026 (Weeks 12–14): Competition Prep**
- Benchmark extensively on varied courses
- Tune for maximum speed with minimum crashes
- Test on PyBullet/Colosseum for sim-to-real confidence
- Package for DCL platform submission
- Goal: Submission ready, tested, optimized

### Performance Targets

| Metric | Classical Baseline | Competition Target |
|--------|-------------------|-------------------|
| 5-gate course time | ~30s | <12s |
| Gate success rate | 80% | >98% |
| Max speed | 8 m/s | 15+ m/s |
| Control frequency | 50 Hz | 100+ Hz |
| Perception latency | 50ms | <15ms |
| Replanning time | 100ms | <20ms |

### Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| DCL platform API differs from simulator | High | Abstract sim interface early, adapt quickly |
| CNN fails on real gate appearance | Medium | Domain randomization + fallback to color detection |
| RL policy doesn't transfer to competition sim | Medium | Classical stack as reliable fallback |
| Insufficient compute for training | Medium | Use cloud GPUs (Lambda, RunPod), efficient algorithms |
| Late specification changes from organizers | Medium | Modular architecture, rapid iteration capability |
| Unknown drone parameters | High | System ID on real drone, wide domain randomization |

### Key Decisions to Make

1. **Gate detector**: YOLOv8-nano (easier, proven) vs GateNet (specialized, potentially better) → Start with YOLOv8-nano, try GateNet if time permits

2. **RL algorithm**: PPO (stable, parallelizable) vs SAC (sample-efficient, better for continuous) → Try both, benchmark. Start with PPO.

3. **Low-level controller**: Tuned PID (simple) vs MPC (optimal but complex) vs INDI (robust) → Implement MPC as primary, keep PID as fallback

4. **Trajectory**: Min-jerk (simple, fast) vs Min-snap (smoother, more efficient) → Implement min-jerk first, upgrade to min-snap

5. **Full RL** (end-to-end) vs **Hybrid** (RL + classical) → Hybrid. It's what won before, and classical fallback is insurance.

---

## Appendix: Quick Setup Commands

```bash
# Install project in development mode
pip install -e ".[dev,sim]"

# Run existing tests
pytest tests/ -v

# Run simulation
python -m aigrandprix.simulation.run_sim --episodes 3 --plot --telemetry

# Future: Train RL agent
python scripts/train_ppo.py --timesteps 1000000 --n-envs 8

# Future: Benchmark
python scripts/benchmark.py --episodes 100 --courses 10
```
