"""Drone flight controller with multiple control strategies.

Modeled after the AirSim Drone Racing Lab API hierarchy:
- Low-level: angle rates + throttle (FPV-style)
- Medium-level: velocity / position setpoints
- High-level: spline trajectory following (minimum-jerk)

This abstraction ensures we can adapt to whatever API the DCL platform exposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ControlLevel(Enum):
    """Control API level matching typical drone racing platforms."""

    ANGLE_RATE = "angle_rate"  # roll_rate, pitch_rate, yaw_rate, throttle
    VELOCITY = "velocity"  # vx, vy, vz, yaw_rate
    POSITION = "position"  # x, y, z, yaw
    SPLINE = "spline"  # waypoints -> minimum-jerk trajectory


@dataclass
class DroneState:
    """Current drone state."""

    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    orientation: np.ndarray  # quaternion [w, x, y, z]
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def yaw(self) -> float:
        """Extract yaw angle from quaternion."""
        w, x, y, z = self.orientation
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


@dataclass
class ControlCommand:
    """Low-level control command for the drone."""

    thrust: float  # normalized [0, 1]
    roll_rate: float  # rad/s
    pitch_rate: float  # rad/s
    yaw_rate: float  # rad/s


@dataclass
class VelocityCommand:
    """Medium-level velocity setpoint."""

    vx: float
    vy: float
    vz: float
    yaw_rate: float = 0.0


@dataclass
class PositionCommand:
    """Medium-level position setpoint."""

    x: float
    y: float
    z: float
    yaw: float = 0.0


@dataclass
class PIDGains:
    """PID controller gains."""

    kp: np.ndarray = field(default_factory=lambda: np.array([6.0, 6.0, 8.0]))
    ki: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.2]))
    kd: np.ndarray = field(default_factory=lambda: np.array([3.5, 3.5, 4.5]))


class DroneController:
    """Multi-level drone flight controller.

    Supports the full control hierarchy:
    - Low-level: angle rate commands (for RL policies)
    - Medium-level: velocity/position tracking (for classical control)
    - High-level: spline trajectory following (for racing)

    Designed to be API-agnostic so we can swap between simulators
    and the eventual DCL competition platform.
    """

    def __init__(self, gains: PIDGains | None = None, dt: float = 0.01) -> None:
        self.gains = gains or PIDGains()
        self.dt = dt
        self._integral_error = np.zeros(3)
        self._prev_error = np.zeros(3)
        self._vel_integral_error = np.zeros(3)
        self._vel_prev_error = np.zeros(3)
        self.max_speed = 15.0  # m/s
        self.max_thrust = 1.0
        self.max_rate = 8.0  # rad/s

    def track_position(self, state: DroneState, target_pos: np.ndarray) -> ControlCommand:
        """PID position controller -> low-level commands."""
        error = target_pos - state.position

        self._integral_error += error * self.dt
        self._integral_error = np.clip(self._integral_error, -2.0, 2.0)

        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error.copy()

        accel = (
            self.gains.kp * error
            + self.gains.ki * self._integral_error
            + self.gains.kd * derivative
        )

        return self._accel_to_command(accel, state)

    def track_velocity(self, state: DroneState, target_vel: np.ndarray) -> ControlCommand:
        """Velocity tracking controller -> low-level commands."""
        vel_error = target_vel - state.velocity

        self._vel_integral_error += vel_error * self.dt
        self._vel_integral_error = np.clip(self._vel_integral_error, -2.0, 2.0)

        vel_derivative = (vel_error - self._vel_prev_error) / self.dt
        self._vel_prev_error = vel_error.copy()

        accel = (
            self.gains.kp * 0.8 * vel_error
            + self.gains.ki * 0.5 * self._vel_integral_error
            + self.gains.kd * 0.3 * vel_derivative
        )

        return self._accel_to_command(accel, state)

    def track_trajectory_point(
        self,
        state: DroneState,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_accel: np.ndarray | None = None,
    ) -> ControlCommand:
        """Combined position + velocity + feedforward tracking.

        This is the key controller for racing: uses the trajectory's
        planned velocity and acceleration as feedforward terms for
        tighter tracking at high speeds.
        """
        pos_error = target_pos - state.position
        vel_error = target_vel - state.velocity

        # Feedforward acceleration from trajectory
        ff_accel = target_accel if target_accel is not None else np.zeros(3)

        # PD on position + P on velocity + feedforward
        accel = (
            self.gains.kp * pos_error
            + self.gains.kd * vel_error
            + ff_accel
        )

        return self._accel_to_command(accel, state)

    def compute_yaw_rate(self, state: DroneState, target_pos: np.ndarray) -> float:
        """Compute yaw rate to face the target position."""
        direction = target_pos - state.position
        target_yaw = np.arctan2(direction[1], direction[0])
        yaw_error = target_yaw - state.yaw

        # Normalize to [-pi, pi]
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

        return float(np.clip(yaw_error * 3.0, -self.max_rate / 2, self.max_rate / 2))

    def _accel_to_command(self, accel: np.ndarray, state: DroneState) -> ControlCommand:
        """Convert desired acceleration to low-level angle rate + thrust commands."""
        gravity = np.array([0.0, 0.0, 9.81])
        total_accel = accel + gravity

        thrust = float(np.clip(np.linalg.norm(total_accel) / 20.0, 0, self.max_thrust))

        if np.linalg.norm(total_accel) > 0.01:
            z_body = total_accel / np.linalg.norm(total_accel)
        else:
            z_body = np.array([0, 0, 1])

        roll_target = np.arcsin(np.clip(-z_body[1], -1, 1))
        pitch_target = np.arctan2(z_body[0], z_body[2])

        roll_rate = float(np.clip(roll_target * 5.0, -self.max_rate, self.max_rate))
        pitch_rate = float(np.clip(pitch_target * 5.0, -self.max_rate, self.max_rate))

        return ControlCommand(
            thrust=thrust,
            roll_rate=roll_rate,
            pitch_rate=pitch_rate,
            yaw_rate=0.0,
        )

    def reset(self) -> None:
        """Reset all controller state."""
        self._integral_error = np.zeros(3)
        self._prev_error = np.zeros(3)
        self._vel_integral_error = np.zeros(3)
        self._vel_prev_error = np.zeros(3)
