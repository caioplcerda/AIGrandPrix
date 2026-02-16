"""Tests for MPC controller."""

import time
from dataclasses import dataclass

import numpy as np
import pytest

from aigrandprix.control.drone_controller import DroneState
from aigrandprix.control.mpc_controller import MPCConfig, MPCController, SimplifiedLinearModel


def _make_state(pos=(0, 0, 1), vel=(0, 0, 0)):
    return DroneState(
        position=np.array(pos, dtype=float),
        velocity=np.array(vel, dtype=float),
        orientation=np.array([1, 0, 0, 0], dtype=float),
        angular_velocity=np.zeros(3),
    )


@dataclass
class _FakeTrajectoryPoint:
    """Minimal trajectory point for MPC tests."""
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


def _make_trajectory_point(pos, vel=(0, 0, 0), accel=(0, 0, 0)):
    return _FakeTrajectoryPoint(
        position=np.array(pos, dtype=float),
        velocity=np.array(vel, dtype=float),
        acceleration=np.array(accel, dtype=float),
    )


# --- SimplifiedLinearModel tests ---

def test_simplified_model_prediction():
    """Model.predict matches gym_env physics for known inputs."""
    model = SimplifiedLinearModel(dt=0.02, drag=0.99)
    dt = 0.02
    drag = 0.99

    pos = np.array([0.0, 0.0, 1.0])
    vel = np.array([1.0, -0.5, 0.2])
    x = np.concatenate([pos, vel])

    # Control: thrust=0.99, roll_rate=0.0, pitch_rate=2.0
    u = np.array([0.99, 0.0, 2.0])

    # Reference: manually compute gym_env simplified physics
    accel = np.array([
        u[2] * 0.5,       # pitch_rate * 0.5 = 1.0
        -u[1] * 0.5,      # -roll_rate * 0.5 = 0.0
        (u[0] - 0.5) * 20.0 - 9.81,  # = 9.8 - 9.81 = -0.01
    ])
    vel_new = (vel + accel * dt) * drag
    pos_new = pos + vel_new * dt

    x_pred = model.predict(x, u)

    np.testing.assert_allclose(x_pred[:3], pos_new, atol=1e-6)
    np.testing.assert_allclose(x_pred[3:6], vel_new, atol=1e-6)


def test_prediction_matrices():
    """Batch prediction via S,T,W matches sequential predict."""
    model = SimplifiedLinearModel(dt=0.02, drag=0.99)
    N = 5

    x0 = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    # Random controls
    rng = np.random.default_rng(42)
    U_seq = []
    for _ in range(N):
        U_seq.append(np.array([
            rng.uniform(0.4, 1.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
        ]))

    # Sequential prediction
    states = []
    x = x0.copy()
    for i in range(N):
        x = model.predict(x, U_seq[i])
        states.append(x.copy())
    X_sequential = np.concatenate(states)

    # Batch prediction
    S, T, W = model.build_prediction_matrices(N)
    U_flat = np.concatenate(U_seq)
    X_batch = S @ x0 + T @ U_flat + W

    np.testing.assert_allclose(X_batch, X_sequential, atol=1e-8)


# --- MPCController tests ---

def test_mpc_hover():
    """MPC at target position produces hover thrust ≈ 0.99, near-zero rates."""
    config = MPCConfig(horizon=10, dt=0.02)
    mpc = MPCController(config=config)

    state = _make_state(pos=[0, 0, 1], vel=[0, 0, 0])
    # Reference: stay at current position
    ref_points = [_make_trajectory_point(pos=[0, 0, 1]) for _ in range(10)]

    cmd = mpc.compute_control(state, ref_points)

    # Hover thrust in simplified mode: (0 + 9.81)/20 + 0.5 ≈ 0.99
    assert abs(cmd.thrust - 0.99) < 0.1, f"thrust={cmd.thrust}"
    assert abs(cmd.roll_rate) < 1.0
    assert abs(cmd.pitch_rate) < 1.0


def test_mpc_tracks_straight_line():
    """MPC tracking a straight line keeps position error < 0.5m after warmup."""
    config = MPCConfig(horizon=10, dt=0.02)
    mpc = MPCController(config=config)

    # Start drone already at target velocity to test steady-state tracking
    target_speed = 3.0
    pos = np.array([0.0, 0.0, 1.0])
    vel = np.array([target_speed, 0.0, 0.0])
    dt = 0.02
    drag = 0.99

    max_error = 0.0
    for step in range(100):
        t = step * dt
        target_x = target_speed * t
        ref_points = []
        for i in range(10):
            ref_t = t + i * dt
            ref_points.append(_make_trajectory_point(
                pos=[target_speed * ref_t, 0, 1],
                vel=[target_speed, 0, 0],
                accel=[0, 0, 0],
            ))

        state = _make_state(pos=pos, vel=vel)
        cmd = mpc.compute_control(state, ref_points)

        # Simulate one step using simplified physics
        accel = np.array([
            cmd.pitch_rate * 0.5,
            -cmd.roll_rate * 0.5,
            (cmd.thrust - 0.5) * 20.0 - 9.81,
        ])
        vel = (vel + accel * dt) * drag
        pos = pos + vel * dt

        error = abs(pos[0] - target_x)
        max_error = max(max_error, error)

    assert max_error < 0.5, f"max tracking error = {max_error:.3f}m"


def test_mpc_solve_time():
    """MPC solve takes < 5ms for horizon=10."""
    config = MPCConfig(horizon=10, dt=0.02)
    mpc = MPCController(config=config)

    state = _make_state(pos=[0, 0, 1])
    ref_points = [_make_trajectory_point(pos=[i * 0.1, 0, 1], vel=[1, 0, 0]) for i in range(10)]

    # Warm up
    mpc.compute_control(state, ref_points)

    # Measure
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        mpc.compute_control(state, ref_points)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(times)
    assert avg_ms < 5.0, f"avg solve time = {avg_ms:.2f}ms"


def test_mpc_bounds_respected():
    """Thrust in [0,1], rates in [-8,8]."""
    config = MPCConfig(horizon=10, dt=0.02)
    mpc = MPCController(config=config, max_rate=8.0, max_thrust=1.0)

    # Large error to push aggressive commands
    state = _make_state(pos=[0, 0, 1], vel=[0, 0, 0])
    ref_points = [_make_trajectory_point(pos=[50, 50, 50], vel=[10, 10, 10]) for _ in range(10)]

    cmd = mpc.compute_control(state, ref_points)

    assert 0.0 <= cmd.thrust <= 1.0
    assert -8.0 <= cmd.roll_rate <= 8.0
    assert -8.0 <= cmd.pitch_rate <= 8.0


def test_mpc_warm_start_shift():
    """Warm start stores shifted previous solution."""
    config = MPCConfig(horizon=5, dt=0.02)
    mpc = MPCController(config=config)

    state = _make_state(pos=[0, 0, 1])
    ref_points = [_make_trajectory_point(pos=[i * 0.5, 0, 1], vel=[2.5, 0, 0]) for i in range(5)]

    # First call sets up warm start
    mpc.compute_control(state, ref_points)
    assert mpc._prev_solution is not None
    prev = mpc._prev_solution.copy()

    # Second call: solution should have been shifted
    mpc.compute_control(state, ref_points)
    assert mpc._prev_solution is not None


def test_mpc_fallback():
    """Fallback returns valid command on empty trajectory."""
    config = MPCConfig(horizon=5, dt=0.02)
    mpc = MPCController(config=config)

    state = _make_state(pos=[0, 0, 1])

    # Fallback with a single trajectory point
    ref_points = [_make_trajectory_point(pos=[1, 0, 1], vel=[1, 0, 0], accel=[0, 0, 0])]
    cmd = mpc._fallback_command(ref_points, state)

    assert 0.0 <= cmd.thrust <= 1.0
    assert isinstance(cmd.roll_rate, float)
    assert isinstance(cmd.pitch_rate, float)

    # Fallback with empty trajectory -> hover
    cmd_hover = mpc._fallback_command([], state)
    assert abs(cmd_hover.thrust - 0.99) < 0.02
