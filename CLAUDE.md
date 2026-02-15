# AI Grand Prix - Autonomous Drone Racing Competition

## Competitor
**Caio de Paula Lacerda** | Teams of up to 8 allowed

## Competition Overview
The **AI Grand Prix** is a global autonomous drone racing competition by **Anduril** in partnership with **Drone Champions League (DCL)**, **Neros Technologies**, and **JobsOhio**.

- **Prize Pool**: $500,000 + job opportunity at Anduril
- **Website**: https://theaigrandprix.com
- **Contact**: contact_aipg@theaigrandprix.com

## Timeline
| Phase | Date | Details |
|-------|------|---------|
| 1st Virtual Qualifier | May 2026 | Submit Python AI algorithms to DCL platform |
| 2nd Virtual Qualifier | June 2026 | Second chance / improved submissions |
| Physical Qualifier | September 2026 | In-person, Southern California (2 weeks) |
| AI Grand Prix Final | November 2026 | Columbus, Ohio - live head-to-head racing |

## Technical Requirements

### What We Must Build
An autonomous system that navigates a drone through a sequence of gates in a virtual environment, as fast as possible.

### Three Core Pillars
1. **Gate Recognition (Perception)**: Detect and locate gates using sensor data + visual feed. Gates are mostly standardized.
2. **Drone Control**: Command flight dynamics (speed, orientation, thrust) with precision. Balance speed vs accuracy.
3. **Path Planning & Navigation**: Plot efficient route through all gates in correct order under realistic physics.

### Competition Rules
- **NO manual control** - fully autonomous
- **NO hardware modifications** - identical Neros Technology drones with DCL AI vector module
- **Python-based AI algorithms** submitted to DCL-built platform
- **Fastest time wins** - complete all gates in shortest time
- Software quality is the ONLY differentiator

### Hardware (Provided)
- Identical drones built by **Neros Technologies**
- Incorporates **DCL's AI vector module**
- Detailed specs TBD (will be released closer to qualifiers)

## Development Strategy

### Approach: Hybrid Stack (Classical + ML)
We pursue a hybrid approach combining:
- **Classical control** for precise drone dynamics (PID/MPC controllers)
- **Computer vision** for gate detection (CNN-based, potentially YOLO/custom)
- **Reinforcement learning** for optimal path planning and racing strategy
- **Minimum jerk trajectory planning** for smooth high-speed flight

### Architecture
```
src/aigrandprix/
├── perception/       # Gate detection, visual processing
│   ├── gate_detector.py      # CNN/YOLO gate detection
│   ├── depth_estimation.py   # Distance to gate
│   └── state_estimator.py    # Drone state from sensors
├── control/          # Flight dynamics control
│   ├── pid_controller.py     # PID flight controller
│   ├── mpc_controller.py     # Model Predictive Control
│   └── drone_dynamics.py     # Physics model
├── planning/         # Path planning & navigation
│   ├── path_planner.py       # Gate-to-gate trajectory
│   ├── trajectory_opt.py     # Minimum-jerk / time-optimal
│   └── race_strategy.py      # Overall race optimization
├── simulation/       # Sim interface & training
│   ├── sim_interface.py      # Abstract simulator interface
│   ├── gym_env.py            # Gymnasium environment wrapper
│   └── reward_shaping.py     # RL reward functions
└── main.py           # Entry point / competition submission
```

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
1. **Lap time** is the ONLY metric that matters
2. Optimize for speed through gates, not code elegance
3. Profile and benchmark everything
4. Sim-to-real transfer must be considered

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
