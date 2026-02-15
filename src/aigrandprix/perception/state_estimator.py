"""Extended Kalman Filter (EKF) state estimator for drone racing.

Implements a Multiplicative EKF (MEKF) fusing IMU measurements
(accelerometer + gyroscope) with gate detection position updates.

Nominal state: position(3), velocity(3), quaternion(4), angular_velocity(3), accel_bias(3)
Error state:   dp(3), dv(3), dtheta(3), domega(3), dab(3) = 15D
"""

from __future__ import annotations

import numpy as np

from aigrandprix.control.drone_controller import DroneState
from aigrandprix.utils.math_utils import (
    quat_multiply,
    quat_normalize,
    quat_to_rotation_matrix,
    skew,
)

GRAVITY = np.array([0.0, 0.0, -9.81])


class StateEstimator:
    """Multiplicative EKF for drone state estimation."""

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}

        # Nominal state
        self._position = np.zeros(3)
        self._velocity = np.zeros(3)
        self._quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self._angular_velocity = np.zeros(3)
        self._accel_bias = np.zeros(3)

        # Error-state covariance (15x15)
        self._P = np.eye(15) * 1.0

        # Process noise (15x15 diagonal)
        self._Q = np.diag([
            config.get("process_noise_pos", 0.1),
            config.get("process_noise_pos", 0.1),
            config.get("process_noise_pos", 0.1),
            config.get("process_noise_vel", 1.0),
            config.get("process_noise_vel", 1.0),
            config.get("process_noise_vel", 1.0),
            config.get("process_noise_orient", 0.01),
            config.get("process_noise_orient", 0.01),
            config.get("process_noise_orient", 0.01),
            config.get("process_noise_gyro", 0.1),
            config.get("process_noise_gyro", 0.1),
            config.get("process_noise_gyro", 0.1),
            config.get("process_noise_accel_bias", 0.001),
            config.get("process_noise_accel_bias", 0.001),
            config.get("process_noise_accel_bias", 0.001),
        ])

        # Measurement noise for gate position updates
        meas_noise = config.get("measurement_noise_pos", 0.5)
        self._R_pos = np.eye(3) * meas_noise ** 2

    def reset(self, state: DroneState) -> None:
        """Reset estimator to a known state."""
        self._position = state.position.copy()
        self._velocity = state.velocity.copy()
        self._quaternion = quat_normalize(state.orientation.copy())
        self._angular_velocity = state.angular_velocity.copy()
        self._accel_bias = np.zeros(3)
        self._P = np.eye(15) * 1.0

    def predict(
        self,
        accel_meas: np.ndarray,
        gyro_meas: np.ndarray,
        dt: float,
    ) -> None:
        """Propagate state forward using IMU measurements.

        Args:
            accel_meas: accelerometer reading in body frame (m/s^2)
            gyro_meas: gyroscope reading in body frame (rad/s)
            dt: timestep in seconds
        """
        # Correct accelerometer for bias
        accel_body = accel_meas - self._accel_bias

        # Rotation matrix (body → world)
        R = quat_to_rotation_matrix(self._quaternion)

        # World-frame acceleration (rotate to world + gravity)
        accel_world = R @ accel_body + GRAVITY

        # Integrate nominal state
        self._position = self._position + self._velocity * dt + 0.5 * accel_world * dt ** 2
        self._velocity = self._velocity + accel_world * dt

        # Integrate quaternion with gyroscope
        omega = gyro_meas
        self._angular_velocity = omega.copy()
        angle = np.linalg.norm(omega) * dt
        if angle > 1e-10:
            axis = omega / np.linalg.norm(omega)
            half = angle / 2.0
            dq = np.array([np.cos(half), *(axis * np.sin(half))])
        else:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        self._quaternion = quat_normalize(quat_multiply(self._quaternion, dq))

        # Build 15x15 error-state Jacobian F
        F = np.eye(15)
        # dp/dv
        F[0:3, 3:6] = np.eye(3) * dt
        # dv/dtheta: -R [a_body]x dt
        F[3:6, 6:9] = -R @ skew(accel_body) * dt
        # dv/dab: -R dt
        F[3:6, 12:15] = -R * dt
        # dtheta/domega: I * dt (small angle)
        F[6:9, 9:12] = np.eye(3) * dt

        # Propagate covariance
        self._P = F @ self._P @ F.T + self._Q * dt

    def update_gate_detection(
        self,
        gate_world_pos: np.ndarray,
        gate_camera_pos: np.ndarray,
    ) -> None:
        """Update state with a gate position measurement.

        Args:
            gate_world_pos: known gate position in world frame
            gate_camera_pos: measured gate position in camera/body frame (from PnP)
        """
        R = quat_to_rotation_matrix(self._quaternion)

        # Expected measurement: gate in body frame
        expected = R.T @ (gate_world_pos - self._position)

        # Innovation
        innovation = gate_camera_pos - expected

        # Measurement Jacobian H (3x15)
        H = np.zeros((3, 15))
        # d(measurement)/d(position) = -R^T
        H[0:3, 0:3] = -R.T
        # d(measurement)/d(attitude error): R^T [gate_world - pos]x
        H[0:3, 6:9] = R.T @ skew(gate_world_pos - self._position)

        # Kalman gain
        S = H @ self._P @ H.T + self._R_pos
        K = self._P @ H.T @ np.linalg.inv(S)

        # Error state update
        dx = K @ innovation

        # Apply corrections to nominal state
        self._position = self._position + dx[0:3]
        self._velocity = self._velocity + dx[3:6]

        # Attitude correction via small-angle quaternion
        dtheta = dx[6:9]
        angle = np.linalg.norm(dtheta)
        if angle > 1e-10:
            axis = dtheta / angle
            half = angle / 2.0
            dq = np.array([np.cos(half), *(axis * np.sin(half))])
        else:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        self._quaternion = quat_normalize(quat_multiply(self._quaternion, dq))

        self._angular_velocity = self._angular_velocity + dx[9:12]
        self._accel_bias = self._accel_bias + dx[12:15]

        # Joseph-form covariance update for numerical stability
        IKH = np.eye(15) - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ self._R_pos @ K.T

    def get_state(self) -> DroneState:
        """Return current state estimate as DroneState."""
        return DroneState(
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            orientation=self._quaternion.copy(),
            angular_velocity=self._angular_velocity.copy(),
        )
