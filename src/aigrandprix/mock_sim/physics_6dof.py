"""6-DOF rigid body physics for DCL mock simulator (VADR-TS-002 §3.2).

Physics update frequency: 120 Hz (dt = 1/120 s).
State: position (NED), velocity (NED), roll/pitch/yaw.
Control: SET_POSITION_TARGET_LOCAL_NED → inner PD controller → velocity/attitude.

NED conventions: X=north, Y=east, Z=down. Altitude = -Z.
Gravity: +9.81 m/s² in +Z (NED down).

The inner controller mimics what a real autopilot would do with position targets.
It is NOT competition logic — it is the mock sim's internal stabilizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PhysicsState:
    """Full 6DOF state in NED frame."""

    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -2.0]))  # NED (m)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))                 # NED (m/s)
    roll: float = 0.0   # rad
    pitch: float = 0.0  # rad
    yaw: float = 0.0    # rad (NED heading, 0=north, π/2=east)
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0
    time_s: float = 0.0

    def copy(self) -> "PhysicsState":
        return PhysicsState(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            roll=self.roll, pitch=self.pitch, yaw=self.yaw,
            roll_rate=self.roll_rate, pitch_rate=self.pitch_rate, yaw_rate=self.yaw_rate,
            time_s=self.time_s,
        )


@dataclass
class ControlCommand:
    """Desired state from SET_POSITION_TARGET_LOCAL_NED."""

    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -2.0]))  # NED
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))                  # NED
    yaw: float = 0.0  # rad

    def is_hover(self) -> bool:
        return bool(np.allclose(self.vel, 0) and np.allclose(self.pos, np.array([0.0, 0.0, -2.0])))


_DT = 1.0 / 120.0           # 120 Hz physics
# Terminal velocity = sqrt(MAX_ACCEL / DRAG). Prior 0.10/40 capped drone at 20 m/s
# regardless of commanded speed. Rebalanced to racing-realistic 6g thrust + low drag
# so terminal = sqrt(60/0.03) = 44.7 m/s — commanded 38 now actually achievable.
_DRAG = 0.03                 # aerodynamic drag coefficient
_MAX_SPEED = 52.0            # m/s — hard speed clip (below terminal 51.6 so terminal binds)
_MAX_ACCEL = 80.0            # m/s² ≈ 8g, upper end for racing quad; terminal = sqrt(80/0.03) = 51.6
_ATTITUDE_RATE_MAX = 4.0     # rad/s max attitude rate
_G = 9.81


class Physics6DOF:
    """Simple 6DOF drone physics with inner PD position/velocity controller.

    The inner loop implements the "flight controller" that the real DCL sim
    would run internally when given a SET_POSITION_TARGET_LOCAL_NED command.

    Gains are tuned for smooth, stable behavior (not racing performance).
    """

    def __init__(
        self,
        initial_pos: np.ndarray | None = None,
        initial_yaw: float = 0.0,
        kp_pos: float = 3.5,
        kd_pos: float = 8.0,   # raised 2→8: damps overshoot at high speed (low-drag regime)
        kp_att: float = 8.0,
    ) -> None:
        pos = initial_pos.copy() if initial_pos is not None else np.array([0.0, 0.0, -2.0])
        self._state = PhysicsState(pos=pos, yaw=initial_yaw)
        self._cmd = ControlCommand(pos=pos.copy())
        self._kp_pos = kp_pos
        self._kd_pos = kd_pos
        self._kp_att = kp_att
        self._t = 0.0
        self._last_world_accel = np.zeros(3)

    @property
    def state(self) -> PhysicsState:
        return self._state.copy()

    def set_command(self, cmd: ControlCommand) -> None:
        self._cmd = cmd

    def step(self) -> PhysicsState:
        """Advance physics by one dt (1/120 s). Returns new state."""
        dt = _DT
        s = self._state
        cmd = self._cmd

        # ── Desired acceleration from position + velocity error (PD)
        pos_err = cmd.pos - s.pos
        vel_err = cmd.vel - s.vel
        desired_acc = self._kp_pos * pos_err + self._kd_pos * vel_err

        # Clamp to physical limits
        desired_acc = np.clip(desired_acc, -_MAX_ACCEL, _MAX_ACCEL)

        # Gravity compensation in desired accel
        # NED Z: up thrust = -Z accel. Gravity adds +Z. Net accel = desired - [0,0,g]...
        # For velocity control: just apply desired as external force, gravity already baked in
        # We model as: accel_world = desired_acc (controller already compensates gravity)

        # ── Drag
        speed = np.linalg.norm(s.vel)
        if speed > 0.01:
            drag = -_DRAG * s.vel * speed
        else:
            drag = np.zeros(3)

        total_acc = desired_acc + drag
        self._last_world_accel = total_acc.copy()

        # ── Integrate velocity and position
        new_vel = s.vel + total_acc * dt
        # Speed limit
        spd = np.linalg.norm(new_vel)
        if spd > _MAX_SPEED:
            new_vel = new_vel / spd * _MAX_SPEED

        new_pos = s.pos + new_vel * dt

        # ── Ground clamp: altitude = -z, z ≤ 0 means on ground
        if new_pos[2] > 0.05:     # z > 0.05 = below ground + small tolerance
            new_pos[2] = 0.05
            new_vel[2] = min(new_vel[2], 0.0)

        # ── Attitude (simplified: pitch/roll proportional to horizontal accel)
        # pitch: positive nose-up (pitch-down in NED body). accel[0] = north = forward.
        # When accelerating north (forward), pitch forward = positive pitch.
        # In NED: positive pitch = nose-down (positive X tilts toward +Z = down)
        target_pitch = math.atan2(-desired_acc[0], _G)  # nose-up when accel forward
        target_roll = math.atan2(desired_acc[1], _G)    # right-side-up when accel right

        target_pitch = float(np.clip(target_pitch, -math.radians(45), math.radians(45)))
        target_roll = float(np.clip(target_roll, -math.radians(45), math.radians(45)))

        # Yaw tracking
        yaw_err = _wrap_angle(cmd.yaw - s.yaw)
        new_yaw_rate = float(np.clip(self._kp_att * yaw_err, -_ATTITUDE_RATE_MAX, _ATTITUDE_RATE_MAX))
        new_yaw = s.yaw + new_yaw_rate * dt
        new_yaw = float(_wrap_angle(new_yaw))

        # Attitude tracks target with time constant ~0.1s
        att_tc = 0.1
        att_alpha = dt / (att_tc + dt)
        new_pitch = s.pitch + att_alpha * (target_pitch - s.pitch)
        new_roll = s.roll + att_alpha * (target_roll - s.roll)

        # Estimate rates from change
        new_pitch_rate = (new_pitch - s.pitch) / dt
        new_roll_rate = (new_roll - s.roll) / dt

        self._t += dt
        self._state = PhysicsState(
            pos=new_pos,
            vel=new_vel,
            roll=float(new_roll),
            pitch=float(new_pitch),
            yaw=float(new_yaw),
            roll_rate=float(new_roll_rate),
            pitch_rate=float(new_pitch_rate),
            yaw_rate=float(new_yaw_rate),
            time_s=self._t,
        )
        return self._state.copy()

    def body_frame_accel(self) -> np.ndarray:
        """Return simulated body-frame specific force (what HIGHRES_IMU.xacc etc. would report).

        specific_force_body = R.T @ a_world - g_body
        This lets the state estimator recover: world_accel = R @ specific_force + g_ned = a_world.
        """
        s = self._state
        R = _rpy_to_R(s.roll, s.pitch, s.yaw)
        g_ned = np.array([0.0, 0.0, _G])
        g_body = R.T @ g_ned
        return R.T @ self._last_world_accel - g_body


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-NED rotation matrix (ZYX Tait-Bryan)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                  cp * cr],
    ])


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return float((a + math.pi) % (2 * math.pi) - math.pi)
