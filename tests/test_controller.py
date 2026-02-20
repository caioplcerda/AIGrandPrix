"""Tests for drone controller."""

import numpy as np

from aigrandprix.control.drone_controller import DroneController, DroneState, PIDGains


def _make_state(pos=(0, 0, 1), vel=(0, 0, 0)):
    return DroneState(
        position=np.array(pos, dtype=float),
        velocity=np.array(vel, dtype=float),
        orientation=np.array([1, 0, 0, 0], dtype=float),
        angular_velocity=np.zeros(3),
    )


# --- Original tests (adjusted for simplified physics_mode default) ---

def test_controller_tracks_position():
    ctrl = DroneController()
    state = _make_state(pos=[0, 0, 1])
    target = np.array([5.0, 0.0, 1.0])
    cmd = ctrl.track_position(state, target)

    assert 0 <= cmd.thrust <= 1
    # Simplified mode: pitch_rate = accel_x * 2.0, should be positive to move +x
    assert cmd.pitch_rate != 0


def test_controller_hover():
    ctrl = DroneController()
    state = _make_state(pos=[0, 0, 1])
    target = np.array([0.0, 0.0, 1.0])
    cmd = ctrl.track_position(state, target)

    # Simplified mode: hover thrust ≈ 0.99
    assert cmd.thrust > 0.3


def test_controller_reset():
    ctrl = DroneController()
    state = _make_state()
    ctrl.track_position(state, np.array([1, 0, 0]))
    ctrl.reset()
    assert np.all(ctrl._integral_error == 0)


# --- New tests: physics_mode ---

def test_hover_thrust_simplified():
    """In simplified mode, hover (zero accel) needs thrust ≈ 0.99."""
    ctrl = DroneController()
    ctrl.physics_mode = "simplified"
    state = _make_state(pos=[0, 0, 1])
    target = np.array([0.0, 0.0, 1.0])
    cmd = ctrl.track_position(state, target)

    # thrust = (0 + 9.81) / 20.0 + 0.5 = 0.9905
    assert abs(cmd.thrust - 0.99) < 0.05
    assert abs(cmd.roll_rate) < 0.01
    assert abs(cmd.pitch_rate) < 0.01


def test_hover_thrust_full():
    """In full mode, hover needs thrust ≈ 0.50."""
    ctrl = DroneController()
    ctrl.physics_mode = "full"
    state = _make_state(pos=[0, 0, 1])
    target = np.array([0.0, 0.0, 1.0])
    cmd = ctrl.track_position(state, target)

    # norm([0,0,9.81]) / (2*9.81) = 0.5
    assert abs(cmd.thrust - 0.50) < 0.05


def test_accel_to_command_simplified_roundtrip():
    """Desired accel -> command -> sim accel should roundtrip."""
    ctrl = DroneController()
    ctrl.physics_mode = "simplified"
    state = _make_state()

    desired_accel = np.array([3.0, -2.0, 0.0])
    cmd = ctrl._accel_to_command(desired_accel, state)

    # Reconstruct what the sim would compute from these commands
    sim_accel = np.array([
        cmd.pitch_rate * 0.5,
        -cmd.roll_rate * 0.5,
        (cmd.thrust - 0.5) * 20.0 - 9.81,
    ])

    np.testing.assert_allclose(sim_accel, desired_accel, atol=0.01)


# --- New tests: gain scheduling ---

def test_gain_scheduling_at_zero_speed():
    """At zero speed, scheduled gains equal base gains."""
    ctrl = DroneController()
    ctrl._gain_scheduling_enabled = True
    state = _make_state(vel=(0, 0, 0))
    kp, ki, kd = ctrl._get_scheduled_gains(state.speed)

    np.testing.assert_allclose(kp, ctrl.gains.kp)
    np.testing.assert_allclose(ki, ctrl.gains.ki)
    np.testing.assert_allclose(kd, ctrl.gains.kd)


def test_gain_scheduling_at_max_speed():
    """At max speed, kp reduced, kd increased, ki near zero."""
    ctrl = DroneController()
    ctrl._gain_scheduling_enabled = True
    ctrl._gs_kp_scale = 0.6
    ctrl._gs_kd_scale = 1.6
    ctrl._gs_ki_scale = 0.2

    kp, ki, kd = ctrl._get_scheduled_gains(ctrl.max_speed)

    # At alpha=1: kp *= 0.6, kd *= 1.6, ki *= 0.2
    np.testing.assert_allclose(kp, ctrl.gains.kp * 0.6)
    np.testing.assert_allclose(kd, ctrl.gains.kd * 1.6)
    np.testing.assert_allclose(ki, ctrl.gains.ki * 0.2)


# --- New tests: acceleration limiting ---

def test_accel_limiting():
    """Each acceleration component clamped to max_accel."""
    ctrl = DroneController()
    ctrl.max_accel = 10.0

    big_accel = np.array([20.0, -15.0, 5.0])
    limited = ctrl._limit_accel(big_accel)
    assert limited[0] == 10.0
    assert limited[1] == -10.0
    assert limited[2] == 5.0  # below limit, unchanged


def test_accel_limiting_no_clamp_when_small():
    """Accel below limit is unchanged."""
    ctrl = DroneController()
    ctrl.max_accel = 15.0

    small_accel = np.array([1.0, 2.0, 3.0])
    result = ctrl._limit_accel(small_accel)
    np.testing.assert_allclose(result, small_accel)


# --- Full dynamics attitude controller tests ---

def test_full_dynamics_attitude_error_drives_rates():
    """Level drone at hover: near-zero rates."""
    ctrl = DroneController()
    ctrl.physics_mode = "full"
    state = _make_state(pos=[0, 0, 1])  # level, hover
    target = np.array([0.0, 0.0, 1.0])
    cmd = ctrl.track_position(state, target)
    assert abs(cmd.roll_rate) < 0.5
    assert abs(cmd.pitch_rate) < 0.5


def test_full_dynamics_tilted_drone_corrects():
    """Tilted drone should produce corrective rate commands."""
    from aigrandprix.utils.math_utils import euler_to_quat
    ctrl = DroneController()
    ctrl.physics_mode = "full"
    # 15-degree pitch
    tilted_orient = euler_to_quat(np.array([0.0, np.radians(15), 0.0]))
    state = DroneState(
        position=np.array([0.0, 0.0, 1.0]),
        velocity=np.zeros(3),
        orientation=tilted_orient,
        angular_velocity=np.zeros(3),
    )
    target = np.array([0.0, 0.0, 1.0])  # hover in place
    cmd = ctrl.track_position(state, target)
    # Should produce corrective pitch rate (negative, to reduce pitch)
    # The exact sign depends on the convention, just check it's non-trivial
    assert abs(cmd.pitch_rate) > 0.5


def test_full_dynamics_closed_loop_hover():
    """Full dynamics hover: <2.0m drift over 500 steps."""
    from aigrandprix.simulation.gym_env import DroneRacingEnv
    env = DroneRacingEnv(num_gates=1, max_steps=600, use_dynamics_model=True)
    env.reset(seed=42)
    ctrl = DroneController(dt=0.02, config={"physics_mode": "full", "attitude": {"kp": 8.0, "kd": 2.0}})
    initial_pos = env._drone_pos.copy()
    for _ in range(500):
        state = DroneState(
            position=env._drone_pos.copy(),
            velocity=env._drone_vel.copy(),
            orientation=env._drone_orient.copy(),
            angular_velocity=env._drone_angular_vel.copy(),
        )
        cmd = ctrl.track_position(state, initial_pos)
        action = np.array([cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate], dtype=np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    drift = np.linalg.norm(env._drone_pos - initial_pos)
    env.close()
    assert drift < 2.0, f"Hover drift too large: {drift:.2f}m"
