"""Tests for EKF state estimator."""

from __future__ import annotations

import numpy as np
import pytest

from aigrandprix.control.drone_controller import DroneState
from aigrandprix.perception.state_estimator import StateEstimator


def _hover_imu():
    """IMU readings for stationary hover (compensating gravity)."""
    # In body frame, accelerometer reads +g upward to counteract gravity
    accel = np.array([0.0, 0.0, 9.81])
    gyro = np.zeros(3)
    return accel, gyro


class TestStateEstimator:
    def test_predict_stationary(self):
        """Zero-motion IMU (with gravity compensation) → state unchanged."""
        ekf = StateEstimator()
        state0 = DroneState(
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        )
        ekf.reset(state0)
        accel, gyro = _hover_imu()
        for _ in range(100):
            ekf.predict(accel, gyro, dt=0.01)
        result = ekf.get_state()
        np.testing.assert_allclose(result.position, state0.position, atol=0.1)
        np.testing.assert_allclose(result.velocity, np.zeros(3), atol=0.1)

    def test_predict_gravity_fall(self):
        """Zero accel input (no thrust) → drone accelerates down at ~g."""
        ekf = StateEstimator()
        ekf.reset(DroneState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        ))
        accel = np.zeros(3)  # no thrust
        gyro = np.zeros(3)
        dt = 0.01
        for _ in range(100):
            ekf.predict(accel, gyro, dt)
        result = ekf.get_state()
        # After 1s of free fall: v_z ≈ -9.81, pos_z ≈ -0.5*g*1^2 ≈ -4.9
        assert result.velocity[2] < -8.0
        assert result.position[2] < -3.0

    def test_update_corrects_position(self):
        """Position error is reduced by gate measurement update."""
        ekf = StateEstimator()
        # Start with wrong position
        ekf.reset(DroneState(
            position=np.array([1.0, 0.0, 0.0]),  # 1m off in X
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        ))

        # True position is origin. Gate is at [5,0,0] in world.
        gate_world = np.array([5.0, 0.0, 0.0])
        # If we're at [0,0,0] truly, gate in camera = R^T @ (gate - pos) = [5,0,0]
        # But EKF thinks we're at [1,0,0], so expected = [4,0,0]
        # Measured (true) = [5,0,0]
        gate_camera = np.array([5.0, 0.0, 0.0])

        for _ in range(5):
            ekf.update_gate_detection(gate_world, gate_camera)

        result = ekf.get_state()
        # Position should move closer to true [0,0,0]
        assert abs(result.position[0]) < 0.5

    def test_quaternion_stays_normalized(self):
        """After many predict/update cycles, quaternion stays unit length."""
        ekf = StateEstimator()
        ekf.reset(DroneState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        ))
        rng = np.random.default_rng(42)
        for _ in range(200):
            accel = rng.normal(0, 2, 3) + np.array([0, 0, 9.81])
            gyro = rng.normal(0, 0.5, 3)
            ekf.predict(accel, gyro, dt=0.01)
        q = ekf.get_state().orientation
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6

    def test_covariance_grows_on_predict(self):
        """Prediction without measurement → covariance trace increases."""
        ekf = StateEstimator()
        ekf.reset(DroneState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        ))
        trace_before = np.trace(ekf._P)
        accel, gyro = _hover_imu()
        for _ in range(10):
            ekf.predict(accel, gyro, dt=0.01)
        trace_after = np.trace(ekf._P)
        assert trace_after > trace_before

    def test_covariance_shrinks_on_update(self):
        """Measurement update → covariance trace decreases."""
        ekf = StateEstimator()
        ekf.reset(DroneState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        ))
        # Grow covariance first
        accel, gyro = _hover_imu()
        for _ in range(50):
            ekf.predict(accel, gyro, dt=0.01)
        trace_before = np.trace(ekf._P)

        # Perform measurement update
        gate_world = np.array([5.0, 0.0, 0.0])
        gate_camera = np.array([5.0, 0.0, 0.0])
        ekf.update_gate_detection(gate_world, gate_camera)
        trace_after = np.trace(ekf._P)
        assert trace_after < trace_before

    def test_reset_sets_state(self):
        """Reset should match input DroneState."""
        ekf = StateEstimator()
        target = DroneState(
            position=np.array([10.0, 20.0, 30.0]),
            velocity=np.array([1.0, 2.0, 3.0]),
            orientation=np.array([0.707, 0.707, 0.0, 0.0]),
            angular_velocity=np.array([0.1, 0.2, 0.3]),
        )
        ekf.reset(target)
        result = ekf.get_state()
        np.testing.assert_allclose(result.position, target.position)
        np.testing.assert_allclose(result.velocity, target.velocity)
        np.testing.assert_allclose(result.angular_velocity, target.angular_velocity)

    def test_get_state_returns_drone_state(self):
        """get_state() should return a DroneState instance."""
        ekf = StateEstimator()
        state = ekf.get_state()
        assert isinstance(state, DroneState)
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)
        assert state.orientation.shape == (4,)
        assert state.angular_velocity.shape == (3,)
