"""Quadrotor dynamics model with Newton-Euler equations and RK4 integration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aigrandprix.control.drone_controller import DroneState
from aigrandprix.utils.math_utils import (
    angular_vel_to_quat_derivative,
    quat_normalize,
    quat_rotate,
    quat_to_euler,
)


@dataclass
class DroneParams:
    """Physical parameters for a generic racing quadrotor (X-config)."""

    mass: float = 0.75  # kg
    arm_length: float = 0.125  # m
    Ixx: float = 0.0025  # kg*m^2
    Iyy: float = 0.0025
    Izz: float = 0.0045
    thrust_coeff: float = 1.0e-5  # N / rpm^2
    drag_coeff: float = 1.0e-7  # N*m / rpm^2
    motor_max_rpm: float = 25000
    linear_drag: float = 0.01
    gravity: float = 9.81
    motor_time_constant: float = 0.02  # s

    @classmethod
    def from_config(cls, config: dict) -> DroneParams:
        return cls(**{k: config[k] for k in config if k in cls.__dataclass_fields__})

    @property
    def inertia_matrix(self) -> np.ndarray:
        return np.diag([self.Ixx, self.Iyy, self.Izz])

    @property
    def inertia_inv(self) -> np.ndarray:
        return np.diag([1.0 / self.Ixx, 1.0 / self.Iyy, 1.0 / self.Izz])


class QuadrotorDynamics:
    """Full Newton-Euler quadrotor dynamics with RK4 integration.

    X-config motor layout (top view, NED-like body frame):
        Motor 0: front-right (+x, -y)  CW
        Motor 1: rear-left  (-x, +y)   CW
        Motor 2: front-left (+x, +y)   CCW
        Motor 3: rear-right (-x, -y)   CCW
    """

    def __init__(self, params: DroneParams | None = None) -> None:
        self.params = params or DroneParams()
        p = self.params
        L = p.arm_length / np.sqrt(2.0)  # effective arm for X-config
        k_t = p.thrust_coeff
        k_d = p.drag_coeff

        # Mixing matrix: [F_total, tau_x, tau_y, tau_z] = M @ [f1, f2, f3, f4]
        #   f_i is thrust from motor i
        self._mixing = np.array([
            [1,    1,    1,    1],        # total thrust
            [L,   -L,    L,   -L],        # roll torque
            [L,   -L,   -L,    L],        # pitch torque
            [-k_d / k_t, -k_d / k_t, k_d / k_t, k_d / k_t],  # yaw (reaction torque)
        ])
        self._mixing_inv = np.linalg.pinv(self._mixing)

    def state_derivative(
        self, state: DroneState, motor_thrusts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute (pos_dot, vel_dot, quat_dot, omega_dot) via Newton-Euler."""
        p = self.params
        q = state.orientation
        omega = state.angular_velocity

        # Wrench from motors
        wrench = self._mixing @ motor_thrusts
        total_thrust = wrench[0]
        torques = wrench[1:]

        # Body-frame thrust vector (along +z body axis)
        body_thrust = np.array([0.0, 0.0, total_thrust])
        # Rotate to world frame
        thrust_world = quat_rotate(q, body_thrust)

        # Linear dynamics (world frame)
        gravity = np.array([0.0, 0.0, -p.gravity * p.mass])
        drag = -p.linear_drag * state.velocity
        vel_dot = (thrust_world + gravity + drag) / p.mass

        pos_dot = state.velocity.copy()

        # Rotational dynamics (body frame)
        I = p.inertia_matrix
        I_inv = p.inertia_inv
        gyroscopic = np.cross(omega, I @ omega)
        omega_dot = I_inv @ (torques - gyroscopic)

        quat_dot = angular_vel_to_quat_derivative(q, omega)

        return pos_dot, vel_dot, quat_dot, omega_dot

    def step(self, state: DroneState, motor_thrusts: np.ndarray, dt: float) -> DroneState:
        """Integrate state forward by dt using RK4."""
        def pack(s: DroneState) -> np.ndarray:
            return np.concatenate([s.position, s.velocity, s.orientation, s.angular_velocity])

        def unpack(y: np.ndarray) -> DroneState:
            return DroneState(
                position=y[0:3].copy(),
                velocity=y[3:6].copy(),
                orientation=quat_normalize(y[6:10]),
                angular_velocity=y[10:13].copy(),
            )

        def deriv(s: DroneState) -> np.ndarray:
            pd, vd, qd, wd = self.state_derivative(s, motor_thrusts)
            return np.concatenate([pd, vd, qd, wd])

        y = pack(state)
        k1 = deriv(state)
        k2 = deriv(unpack(y + 0.5 * dt * k1))
        k3 = deriv(unpack(y + 0.5 * dt * k2))
        k4 = deriv(unpack(y + dt * k3))

        y_new = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return unpack(y_new)

    def command_to_motor_thrusts(
        self,
        thrust_norm: float,
        roll_rate: float,
        pitch_rate: float,
        yaw_rate: float,
    ) -> np.ndarray:
        """Map gym_env-style commands to individual motor thrusts.

        thrust_norm=0.5 -> hover (total_force = mass * g).
        """
        p = self.params
        hover_thrust = p.mass * p.gravity
        total_thrust = thrust_norm * 2.0 * hover_thrust  # 0.5 → hover

        # Simple proportional mapping from rate commands to desired torques
        tau_x = roll_rate * p.Ixx * 5.0
        tau_y = pitch_rate * p.Iyy * 5.0
        tau_z = yaw_rate * p.Izz * 2.0

        desired = np.array([total_thrust, tau_x, tau_y, tau_z])
        thrusts = self._mixing_inv @ desired
        return np.clip(thrusts, 0.0, None)  # motors can only push, not pull
