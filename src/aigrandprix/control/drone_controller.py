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

from aigrandprix.utils.math_utils import quat_to_euler


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
        return float(quat_to_euler(self.orientation)[2])


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

    def __init__(
        self,
        gains: PIDGains | None = None,
        dt: float = 0.01,
        config: dict | None = None,
    ) -> None:
        self.physics_mode = "simplified"
        self.max_accel = 15.0
        self._gain_scheduling_enabled = False
        self._gs_kp_scale = 0.6
        self._gs_kd_scale = 1.6
        self._gs_ki_scale = 0.2

        # Attitude PD gains for full dynamics (Mellinger-style cascaded control)
        self._att_kp = 8.0
        self._att_kd = 2.0
        self._att_kp_yaw = 3.0
        self._att_kd_yaw = 1.0

        if config is not None and gains is None:
            pid_cfg = config.get("pid", {})
            if pid_cfg:
                self.gains = PIDGains(
                    kp=np.array(pid_cfg.get("kp", [6.0, 6.0, 8.0])),
                    ki=np.array(pid_cfg.get("ki", [0.1, 0.1, 0.2])),
                    kd=np.array(pid_cfg.get("kd", [3.5, 3.5, 4.5])),
                )
                gs_cfg = pid_cfg.get("gain_scheduling", {})
                self._gain_scheduling_enabled = gs_cfg.get("enabled", False)
                self._gs_kp_scale = gs_cfg.get("kp_high_speed_scale", 0.6)
                self._gs_kd_scale = gs_cfg.get("kd_high_speed_scale", 1.6)
                self._gs_ki_scale = gs_cfg.get("ki_high_speed_scale", 0.2)
            else:
                self.gains = PIDGains()
            self.max_rate = config.get("max_rate", 8.0)
            self.max_thrust = config.get("max_thrust", 1.0)
            self.physics_mode = config.get("physics_mode", "simplified")
            self.max_accel = config.get("max_accel", 15.0)
            att_cfg = config.get("attitude", {})
            self._att_kp = att_cfg.get("kp", 8.0)
            self._att_kd = att_cfg.get("kd", 2.0)
            self._att_kp_yaw = att_cfg.get("kp_yaw", 3.0)
            self._att_kd_yaw = att_cfg.get("kd_yaw", 1.0)
        else:
            self.gains = gains or PIDGains()
            self.max_rate = 8.0
            self.max_thrust = 1.0
        self.dt = dt
        self._integral_error = np.zeros(3)
        self._prev_error = np.zeros(3)
        self._vel_integral_error = np.zeros(3)
        self._vel_prev_error = np.zeros(3)
        self.max_speed = 15.0  # m/s

    def _get_scheduled_gains(self, speed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply gain scheduling based on current speed.

        At high speed: reduce kp (less overshoot), increase kd (stability),
        reduce ki (no windup).
        """
        if not self._gain_scheduling_enabled:
            return self.gains.kp.copy(), self.gains.ki.copy(), self.gains.kd.copy()

        alpha = np.clip(speed / self.max_speed, 0.0, 1.0)
        # Interpolate from base gains toward high-speed scales
        kp = self.gains.kp * (1.0 - alpha * (1.0 - self._gs_kp_scale))
        kd = self.gains.kd * (1.0 - alpha * (1.0 - self._gs_kd_scale))
        ki = self.gains.ki * (1.0 - alpha * (1.0 - self._gs_ki_scale))
        return kp, ki, kd

    def _limit_accel(self, accel: np.ndarray) -> np.ndarray:
        """Clamp acceleration per-component to max_accel.

        Component-wise clipping is used because in simplified physics
        each axis maps independently to a separate actuator.
        """
        return np.clip(accel, -self.max_accel, self.max_accel)

    def track_position(self, state: DroneState, target_pos: np.ndarray) -> ControlCommand:
        """PID position controller -> low-level commands."""
        error = target_pos - state.position

        self._integral_error += error * self.dt
        self._integral_error = np.clip(self._integral_error, -2.0, 2.0)

        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error.copy()

        kp, ki, kd = self._get_scheduled_gains(state.speed)

        accel = kp * error + ki * self._integral_error + kd * derivative
        accel = self._limit_accel(accel)

        return self._accel_to_command(accel, state)

    def track_velocity(self, state: DroneState, target_vel: np.ndarray) -> ControlCommand:
        """Velocity tracking controller -> low-level commands."""
        vel_error = target_vel - state.velocity

        self._vel_integral_error += vel_error * self.dt
        self._vel_integral_error = np.clip(self._vel_integral_error, -2.0, 2.0)

        vel_derivative = (vel_error - self._vel_prev_error) / self.dt
        self._vel_prev_error = vel_error.copy()

        kp, ki, kd = self._get_scheduled_gains(state.speed)

        accel = kp * 0.8 * vel_error + ki * 0.5 * self._vel_integral_error + kd * 0.3 * vel_derivative
        accel = self._limit_accel(accel)

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
        ff_accel = target_accel.copy() if target_accel is not None else np.zeros(3)

        # In simplified physics the axes are decoupled and z-range is very
        # narrow (+0.19 up, -19.81 down). Trajectory feedforward z-accel
        # is typically far beyond what the physics can deliver, destabilizing
        # thrust. Zero the z-feedforward and let PD handle altitude.
        if self.physics_mode == "simplified":
            ff_accel[2] = 0.0
            # Clamp target z-velocity to prevent unrecoverable dives.
            # Max climb is +0.19 m/s², so downward velocity builds fast
            # and is very hard to arrest. Asymmetric clamp: allow more
            # upward velocity tracking than downward.
            vel_error[2] = np.clip(vel_error[2], -1.0, 5.0)

        kp, _ki, kd = self._get_scheduled_gains(state.speed)

        # PD on position + P on velocity + feedforward
        accel = kp * pos_error + kd * vel_error + ff_accel
        accel = self._limit_accel(accel)

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
        """Convert desired acceleration to low-level commands. Dispatches by physics_mode."""
        if self.physics_mode == "simplified":
            return self._accel_to_command_simplified(accel, state)
        return self._accel_to_command_full(accel, state)

    def _accel_to_command_simplified(self, accel: np.ndarray, state: DroneState) -> ControlCommand:
        """Exact inverse of gym_env simplified physics.

        gym_env does:
            accel_x = pitch_rate * 0.5
            accel_y = -roll_rate * 0.5
            accel_z = (thrust - 0.5) * 20.0 - 9.81
        So the inverse is:
            thrust = (accel_z + 9.81) / 20.0 + 0.5
            pitch_rate = accel_x * 2.0
            roll_rate = -accel_y * 2.0
        """
        thrust = float(np.clip((accel[2] + 9.81) / 20.0 + 0.5, 0.0, self.max_thrust))
        # In simplified physics, never command thrust far below hover.
        # thrust=0.99 is hover, anything below 0.8 causes rapid descent
        # that's nearly impossible to recover from.
        thrust = max(thrust, 0.65)
        pitch_rate = float(np.clip(accel[0] * 2.0, -self.max_rate, self.max_rate))
        roll_rate = float(np.clip(-accel[1] * 2.0, -self.max_rate, self.max_rate))

        # Altitude safety overrides for simplified physics.
        # Max climb accel is only +0.19 m/s², but descent accel can be
        # -19.8 m/s². This extreme asymmetry means any descent velocity
        # is very hard to arrest. We must intervene early and aggressively.
        z = state.position[2]
        vz = state.velocity[2]
        if vz < -2.5:
            # Fast descent at any altitude: override to max thrust
            thrust = 1.0
        elif z < 2.0 and vz < -0.8:
            thrust = 1.0
        elif z < 1.0 and vz < -0.2:
            thrust = 1.0
        elif z < 0.5:
            thrust = 1.0  # floor protection

        return ControlCommand(
            thrust=thrust,
            roll_rate=roll_rate,
            pitch_rate=pitch_rate,
            yaw_rate=0.0,
        )

    def _accel_to_command_full(self, accel: np.ndarray, state: DroneState) -> ControlCommand:
        """Mellinger-style cascaded position-attitude controller for full dynamics.

        1. Compute desired attitude from desired acceleration
        2. Compute attitude error (desired - current)
        3. PD on attitude error to produce angular rate commands
        """
        gravity = np.array([0.0, 0.0, 9.81])
        total_accel = accel + gravity

        thrust = float(np.clip(np.linalg.norm(total_accel) / (2.0 * 9.81), 0, self.max_thrust))

        # Desired attitude from desired acceleration direction
        if np.linalg.norm(total_accel) > 0.01:
            z_body = total_accel / np.linalg.norm(total_accel)
        else:
            z_body = np.array([0, 0, 1])

        roll_desired = np.arcsin(np.clip(-z_body[1], -1, 1))
        pitch_desired = np.arctan2(z_body[0], z_body[2])

        # Current attitude from quaternion
        current_rpy = quat_to_euler(state.orientation)
        roll_current = current_rpy[0]
        pitch_current = current_rpy[1]

        # Attitude error
        roll_error = roll_desired - roll_current
        pitch_error = pitch_desired - pitch_current

        # PD on attitude error: rate = kp * error - kd * angular_velocity
        roll_rate = float(np.clip(
            self._att_kp * roll_error - self._att_kd * state.angular_velocity[0],
            -self.max_rate, self.max_rate,
        ))
        pitch_rate = float(np.clip(
            self._att_kp * pitch_error - self._att_kd * state.angular_velocity[1],
            -self.max_rate, self.max_rate,
        ))
        yaw_rate = float(np.clip(
            -self._att_kp_yaw * current_rpy[2] - self._att_kd_yaw * state.angular_velocity[2],
            -self.max_rate, self.max_rate,
        ))

        return ControlCommand(
            thrust=thrust,
            roll_rate=roll_rate,
            pitch_rate=pitch_rate,
            yaw_rate=yaw_rate,
        )

    def reset(self) -> None:
        """Reset all controller state."""
        self._integral_error = np.zeros(3)
        self._prev_error = np.zeros(3)
        self._vel_integral_error = np.zeros(3)
        self._vel_prev_error = np.zeros(3)
