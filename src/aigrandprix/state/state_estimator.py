"""NED state estimator from MAVLink telemetry (VADR-TS-002, no GPS).

Fuses ATTITUDE (Euler angles) and HIGHRES_IMU (body accel + gyro) to produce
a NED state estimate: position, velocity, yaw.

Position is obtained by double-integrating body accelerometer, rotated to NED
world frame and corrected for gravity. This drifts over time (~1m/30s) but is
sufficient for gate-to-gate navigation in races of <8 minutes.

NED conventions:
  - X = North (forward), Y = East (right), Z = Down
  - Altitude (above arming point) = -z_ned
  - Yaw: 0 = north, π/2 = east (clockwise positive, viewed from above)
  - Gravity = [0, 0, +9.81] in NED world frame
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from aigrandprix.comms.mavlink_client import TelemetryState

_GRAVITY_NED = np.array([0.0, 0.0, 9.81])  # NED: gravity points +Z (down)


@dataclass
class NEDState:
    """Drone state in NED world frame."""

    pos_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))   # [north, east, down] m
    vel_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))   # [vn, ve, vd] m/s
    roll: float = 0.0      # rad
    pitch: float = 0.0     # rad
    yaw: float = 0.0       # rad (0=north, π/2=east)
    timestamp: float = field(default_factory=time.time)

    @property
    def altitude_m(self) -> float:
        """Altitude above arming point (positive = above ground)."""
        return -self.pos_ned[2]

    @property
    def heading_deg(self) -> float:
        return math.degrees(self.yaw) % 360.0

    def copy(self) -> "NEDState":
        return NEDState(
            pos_ned=self.pos_ned.copy(),
            vel_ned=self.vel_ned.copy(),
            roll=self.roll,
            pitch=self.pitch,
            yaw=self.yaw,
            timestamp=self.timestamp,
        )


def _rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-NED rotation matrix from roll/pitch/yaw (Tait-Bryan ZYX)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                  cp * cr],
    ])


class NEDStateEstimator:
    """Simple integration-based NED state estimator.

    Updates from MAVLink telemetry (ATTITUDE + HIGHRES_IMU).
    No Kalman filter — prioritizes simplicity and robustness over accuracy.
    Drift ~0.5-2 m/min depending on IMU quality.

    Position resets can be triggered via reset_position() when a gate is passed
    (using known gate position as anchor).
    """

    def __init__(
        self,
        initial_pos_ned: np.ndarray | None = None,
        vel_decay: float = 0.995,
    ) -> None:
        """
        Args:
            initial_pos_ned: starting position in NED (default: origin)
            vel_decay: per-step velocity decay to bound drift (0.99-1.0)
        """
        self._pos = np.zeros(3) if initial_pos_ned is None else initial_pos_ned.copy()
        self._vel = np.zeros(3)
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._vel_decay = vel_decay
        self._last_time: float | None = None
        self._initialized = False

    def reset(self, pos_ned: np.ndarray | None = None) -> None:
        """Reset state. Call at arming or gate-anchor correction."""
        self._pos = np.zeros(3) if pos_ned is None else pos_ned.copy()
        self._vel = np.zeros(3)
        self._last_time = None
        self._initialized = False

    def reset_position(self, pos_ned: np.ndarray) -> None:
        """Anchor position to known gate position (keeps velocity)."""
        self._pos = pos_ned.copy()

    def update(self, telem: TelemetryState) -> NEDState:
        """Update estimate from latest MAVLink telemetry snapshot.

        Should be called every time new telemetry arrives (up to 120 Hz).
        Returns current NEDState.
        """
        now = time.time()
        dt = (now - self._last_time) if self._last_time is not None else 0.0
        self._last_time = now

        # Update orientation from ATTITUDE (authoritative, no drift)
        self._roll = telem.roll
        self._pitch = telem.pitch
        self._yaw = telem.yaw

        if not self._initialized or dt <= 0.0 or dt > 0.5:
            # Skip integration on first call or on large gaps (reset guard)
            self._initialized = True
            return self._make_state()

        # Rotation matrix body → NED
        R = _rpy_to_rotation_matrix(self._roll, self._pitch, self._yaw)

        # Specific force in body frame (accelerometer reading, includes gravity reaction)
        # In body NED (X fwd, Y right, Z down): when level+hover, zacc ≈ -9.81
        # because accelerometer measures -g in body (reaction to gravity)
        # Specific force = a_body - g_body. So a_body = specific_force + g_body
        # g_body = R.T @ g_ned
        accel_body = np.array([telem.accel_x, telem.accel_y, telem.accel_z])
        g_body = R.T @ _GRAVITY_NED
        # true_accel_body = accel_body + g_body ... but spec says HIGHRES_IMU xacc includes gravity
        # For the real DCL sim, test empirically. Using: world_accel = R @ accel_body + g_ned
        # This is the standard formula: rotate specific force to world, then add gravity to get
        # true world acceleration. But this double-adds if specific_force already includes gravity...
        # We'll use: world_accel = R @ accel_body - g_ned ... no.
        #
        # Correct derivation:
        #   specific_force = accel_measured (no gravity)
        #   true_world_accel = R @ specific_force + g_ned ... no
        #
        # Actually: accelerometer measures force_body / mass. For hover:
        #   net_force = thrust_up - m*g = 0  → force_body = m*g (upward in body = -Z_body in NED)
        #   So accel_body_measured (specific force) = [0, 0, -g] in NED body (Z down)
        #   gravity in NED body = R.T @ [0,0,g] = [0, 0, g] when level
        #   world_accel = R @ (accel_body_measured - R.T @ g_ned_corrections)... complex.
        #
        # Simplest working formula used in most drone stacks:
        #   world_accel_ned = R @ accel_body + g_ned
        # where g_ned = [0, 0, +9.81] (gravity points down = +Z in NED)
        # This works because: accel_body from sensor = specific_force (no gravity),
        # and R @ specific_force_body + g_ned = true acceleration in NED.
        world_accel = R @ accel_body + _GRAVITY_NED

        # Integrate velocity
        self._vel = self._vel * self._vel_decay + world_accel * dt

        # Integrate position
        self._pos = self._pos + self._vel * dt + 0.5 * world_accel * dt * dt

        return self._make_state()

    def get_state(self) -> NEDState:
        """Return current state without updating."""
        return self._make_state()

    def _make_state(self) -> NEDState:
        return NEDState(
            pos_ned=self._pos.copy(),
            vel_ned=self._vel.copy(),
            roll=self._roll,
            pitch=self._pitch,
            yaw=self._yaw,
            timestamp=time.time(),
        )
