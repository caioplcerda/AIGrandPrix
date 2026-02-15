"""Tests for the abstract simulator interface and internal adapter."""

from __future__ import annotations

import numpy as np
import pytest

from aigrandprix.simulation.gym_env import DroneRacingEnv
from aigrandprix.simulation.sim_interface import InternalSimAdapter


@pytest.fixture
def adapter():
    env = DroneRacingEnv(num_gates=3, max_steps=500)
    adp = InternalSimAdapter(env)
    yield adp
    adp.close()


class TestInternalSimAdapter:
    def test_reset_returns_info(self, adapter: InternalSimAdapter):
        info = adapter.reset(seed=42)
        assert isinstance(info, dict)

    def test_step_returns_tuple(self, adapter: InternalSimAdapter):
        adapter.reset(seed=42)
        action = np.array([0.5, 0, 0, 0], dtype=np.float32)
        terminated, truncated, info = adapter.step(action)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_get_state_matches_env(self, adapter: InternalSimAdapter):
        adapter.reset(seed=42)
        state = adapter.get_state()
        np.testing.assert_array_almost_equal(state.position, [0, 0, 1])
        np.testing.assert_array_almost_equal(state.velocity, [0, 0, 0])

    def test_get_gates_returns_correct_count(self, adapter: InternalSimAdapter):
        adapter.reset(seed=42)
        gates = adapter.get_gate_poses()
        assert len(gates) == 3
        for g in gates:
            assert g.normal.shape == (3,)
            # Normal should be unit length
            assert abs(np.linalg.norm(g.normal) - 1.0) < 0.01
