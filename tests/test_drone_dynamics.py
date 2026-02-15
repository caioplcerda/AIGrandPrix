"""Tests for the quadrotor dynamics model."""

from __future__ import annotations

import numpy as np
import pytest

from aigrandprix.control.drone_controller import DroneState
from aigrandprix.control.drone_dynamics import DroneParams, QuadrotorDynamics


def _hover_state() -> DroneState:
    return DroneState(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.zeros(3),
        orientation=np.array([1.0, 0, 0, 0]),
        angular_velocity=np.zeros(3),
    )


class TestDroneParams:
    def test_from_config(self):
        cfg = {"mass": 1.0, "arm_length": 0.2, "Ixx": 0.003}
        params = DroneParams.from_config(cfg)
        assert params.mass == 1.0
        assert params.arm_length == 0.2
        assert params.Ixx == 0.003
        # defaults for fields not in config
        assert params.gravity == 9.81

    def test_inertia_matrix(self):
        p = DroneParams()
        I = p.inertia_matrix
        assert I.shape == (3, 3)
        np.testing.assert_allclose(np.diag(I), [p.Ixx, p.Iyy, p.Izz])


class TestQuadrotorDynamics:
    def test_mixing_matrix_invertible(self):
        dyn = QuadrotorDynamics()
        # pinv @ mixing should be close to identity
        result = dyn._mixing_inv @ dyn._mixing
        np.testing.assert_allclose(result, np.eye(4), atol=1e-10)

    def test_hover_equilibrium(self):
        """At hover thrust, vertical acceleration should be ~0."""
        dyn = QuadrotorDynamics()
        state = _hover_state()
        thrusts = dyn.command_to_motor_thrusts(0.5, 0, 0, 0)
        _, vel_dot, _, _ = dyn.state_derivative(state, thrusts)
        # Vertical acceleration should be near zero (hover)
        assert abs(vel_dot[2]) < 0.5, f"Vertical accel at hover: {vel_dot[2]}"

    def test_free_fall(self):
        """Zero motor thrust -> drone accelerates downward at ~g."""
        dyn = QuadrotorDynamics()
        state = _hover_state()
        thrusts = np.zeros(4)
        _, vel_dot, _, _ = dyn.state_derivative(state, thrusts)
        assert vel_dot[2] < -9.0, f"Expected ~-9.81, got {vel_dot[2]}"

    def test_rk4_does_not_diverge(self):
        """Running many steps with hover thrust should keep state bounded."""
        dyn = QuadrotorDynamics()
        state = _hover_state()
        thrusts = dyn.command_to_motor_thrusts(0.5, 0, 0, 0)
        for _ in range(500):
            state = dyn.step(state, thrusts, dt=0.01)
        # Position should stay reasonable (not diverge)
        assert np.all(np.abs(state.position) < 100), f"State diverged: {state.position}"
        # Quaternion should stay normalized
        assert abs(np.linalg.norm(state.orientation) - 1.0) < 1e-6

    def test_command_to_thrusts_positive(self):
        """Motor thrusts should be non-negative."""
        dyn = QuadrotorDynamics()
        thrusts = dyn.command_to_motor_thrusts(0.5, 1.0, -1.0, 0.5)
        assert np.all(thrusts >= 0), f"Negative thrust: {thrusts}"

    def test_hover_maintains_altitude(self):
        """Hovering for 1 second should keep altitude roughly constant."""
        dyn = QuadrotorDynamics()
        state = _hover_state()
        initial_z = state.position[2]
        thrusts = dyn.command_to_motor_thrusts(0.5, 0, 0, 0)
        for _ in range(100):
            state = dyn.step(state, thrusts, dt=0.01)
        assert abs(state.position[2] - initial_z) < 1.0, (
            f"Altitude drifted: {state.position[2]} vs {initial_z}"
        )
