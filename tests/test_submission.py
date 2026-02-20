"""Tests for DCL submission interface."""

from __future__ import annotations

import numpy as np
import pytest

from aigrandprix.control.drone_controller import ControlCommand
from aigrandprix.submission import AutonomousRacer, SafeActionWrapper
from aigrandprix.submission.type_converters import (
    command_to_array,
    gates_from_race_info,
    observation_to_internal,
)


# --- AutonomousRacer lifecycle ---

class TestAutonomousRacer:
    def _make_race_info(self, num_gates: int = 3) -> dict:
        gates = []
        x = 5.0
        for i in range(num_gates):
            gates.append({"position": [x, 0.0, 2.0], "normal": [1.0, 0.0, 0.0]})
            x += 5.0
        return {"gates": gates, "dt": 0.02}

    def _make_observation(self, pos=(0, 0, 1), gate_idx=None) -> dict:
        obs = {
            "image": np.zeros((480, 640, 3), dtype=np.uint8),
            "position": list(pos),
            "velocity": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "elapsed_time": 0.0,
        }
        if gate_idx is not None:
            obs["current_gate_index"] = gate_idx
        return obs

    def test_lifecycle(self):
        """Init -> N frames -> shutdown without error."""
        racer = AutonomousRacer()
        racer.initialize(self._make_race_info())
        for _ in range(10):
            action = racer.compute(self._make_observation())
            assert action.shape == (4,)
        racer.shutdown()

    def test_valid_action_bounds(self):
        """Actions should always be within bounds."""
        racer = AutonomousRacer()
        racer.initialize(self._make_race_info())
        for _ in range(5):
            action = racer.compute(self._make_observation())
            assert action[0] >= 0.0 and action[0] <= 1.0  # thrust
            assert abs(action[1]) <= 8.0  # roll_rate
            assert abs(action[2]) <= 8.0  # pitch_rate
            assert abs(action[3]) <= 4.0  # yaw_rate
        racer.shutdown()

    def test_compute_before_initialize_raises(self):
        """compute() before initialize() should raise RuntimeError."""
        racer = AutonomousRacer()
        with pytest.raises(RuntimeError):
            racer.compute(self._make_observation())

    def test_gate_sync_from_observation(self):
        """External gate index should sync agent's internal counter."""
        racer = AutonomousRacer()
        racer.initialize(self._make_race_info(5))
        # Send observation with gate_idx=2 (skipping gates 0,1)
        obs = self._make_observation(gate_idx=2)
        racer.compute(obs)
        assert racer._agent._gates_passed == 2
        racer.shutdown()

    def test_run_loop_integration(self):
        """run_loop with InternalSimAdapter should complete without error."""
        from aigrandprix.simulation.gym_env import DroneRacingEnv
        from aigrandprix.simulation.sim_interface import InternalSimAdapter

        env = DroneRacingEnv(num_gates=3, max_steps=200)
        sim = InternalSimAdapter(env)
        racer = AutonomousRacer()
        result = racer.run_loop(sim)
        assert "gates_passed" in result
        assert "steps" in result
        assert result["steps"] > 0


# --- Type converters ---

class TestTypeConverters:
    def test_observation_defaults(self):
        """Missing keys should use defaults."""
        image, state, elapsed, gate_idx = observation_to_internal({})
        assert image.shape == (480, 640, 3)
        np.testing.assert_array_equal(state.position, [0.0, 0.0, 1.0])
        assert elapsed is None
        assert gate_idx is None

    def test_observation_full(self):
        """All keys provided."""
        obs = {
            "image": np.ones((100, 100, 3), dtype=np.uint8),
            "position": [1.0, 2.0, 3.0],
            "velocity": [0.1, 0.2, 0.3],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.1, 0.0],
            "elapsed_time": 1.5,
            "current_gate_index": 2,
        }
        image, state, elapsed, gate_idx = observation_to_internal(obs)
        assert image.shape == (100, 100, 3)
        np.testing.assert_allclose(state.position, [1.0, 2.0, 3.0])
        assert elapsed == 1.5
        assert gate_idx == 2

    def test_command_to_array_shape(self):
        """command_to_array should return (4,) array."""
        cmd = ControlCommand(thrust=0.5, roll_rate=1.0, pitch_rate=-1.0, yaw_rate=0.0)
        arr = command_to_array(cmd)
        assert arr.shape == (4,)
        assert arr.dtype == np.float32
        np.testing.assert_allclose(arr, [0.5, 1.0, -1.0, 0.0])

    def test_gates_from_dict_format(self):
        """Parse gates from dict format."""
        info = {"gates": [
            {"position": [1, 0, 2], "normal": [1, 0, 0]},
            {"position": [5, 0, 2], "normal": [0, 1, 0]},
        ]}
        gates = gates_from_race_info(info)
        assert len(gates) == 2
        np.testing.assert_allclose(gates[0].position, [1, 0, 2])

    def test_gates_from_list_format(self):
        """Parse gates from bare position lists."""
        info = {"gates": [[1, 0, 2], [5, 0, 2]]}
        gates = gates_from_race_info(info)
        assert len(gates) == 2

    def test_gates_alternative_key(self):
        """gate_positions key should work too."""
        info = {"gate_positions": [[1, 0, 2]]}
        gates = gates_from_race_info(info)
        assert len(gates) == 1


# --- Safety wrapper ---

class TestSafeActionWrapper:
    def test_valid_action_passes_through(self):
        """Normal compute should pass through."""
        wrapper = SafeActionWrapper()
        action = wrapper.safe_compute(
            lambda: np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_allclose(action, [0.5, 0.0, 0.0, 0.0])

    def test_exception_returns_hover(self):
        """Exception in compute should return hover."""
        wrapper = SafeActionWrapper()
        def bad_compute():
            raise ValueError("boom")
        action = wrapper.safe_compute(bad_compute)
        np.testing.assert_allclose(action, SafeActionWrapper.HOVER_ACTION)
        assert wrapper._fallback_count == 1

    def test_nan_returns_hover(self):
        """NaN in action should trigger hover fallback."""
        wrapper = SafeActionWrapper()
        action = wrapper.safe_compute(
            lambda: np.array([float("nan"), 0.0, 0.0, 0.0])
        )
        np.testing.assert_allclose(action, SafeActionWrapper.HOVER_ACTION)

    def test_inf_returns_hover(self):
        """Inf in action should trigger hover fallback."""
        wrapper = SafeActionWrapper()
        action = wrapper.safe_compute(
            lambda: np.array([0.5, float("inf"), 0.0, 0.0])
        )
        np.testing.assert_allclose(action, SafeActionWrapper.HOVER_ACTION)

    def test_clipping(self):
        """Out-of-bounds actions should be clipped."""
        wrapper = SafeActionWrapper()
        action = wrapper.safe_compute(
            lambda: np.array([1.5, -20.0, 10.0, 0.0], dtype=np.float32)
        )
        assert action[0] == 1.0  # clipped thrust
        assert action[1] == -8.0  # clipped roll_rate
        assert action[2] == 8.0  # clipped pitch_rate

    def test_latency_stats(self):
        """Stats should track latency."""
        wrapper = SafeActionWrapper()
        for _ in range(10):
            wrapper.safe_compute(
                lambda: np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
            )
        stats = wrapper.get_stats()
        assert stats["total_calls"] == 10
        assert stats["fallback_count"] == 0
        assert stats["p50_ms"] >= 0
        assert stats["p95_ms"] >= 0

    def test_empty_stats(self):
        """Stats with no calls."""
        wrapper = SafeActionWrapper()
        stats = wrapper.get_stats()
        assert stats["total_calls"] == 0
        assert stats["p50_ms"] == 0.0
