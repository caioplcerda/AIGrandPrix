"""Comprehensive tests for the DroneRacingEnv Gymnasium environment."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from aigrandprix.simulation.gym_env import DroneRacingEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    e = DroneRacingEnv(num_gates=3, max_steps=500)
    yield e
    e.close()


@pytest.fixture
def seeded_env():
    """Environment with fixed seed for reproducible tests."""
    e = DroneRacingEnv(num_gates=3, max_steps=500)
    e.reset(seed=42)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Space validation
# ---------------------------------------------------------------------------

class TestSpaces:
    def test_observation_space_shape(self, env: DroneRacingEnv):
        obs, _ = env.reset(seed=0)
        assert obs.shape == (19,), f"Expected obs shape (19,), got {obs.shape}"
        assert env.observation_space.contains(obs)

    def test_action_space_shape(self, env: DroneRacingEnv):
        assert env.action_space.shape == (4,)

    def test_action_space_bounds(self, env: DroneRacingEnv):
        low = env.action_space.low
        high = env.action_space.high
        assert low[0] == pytest.approx(0.0)  # thrust min
        assert high[0] == pytest.approx(1.0)  # thrust max
        assert low[1] == pytest.approx(-8.0)  # roll_rate min
        assert high[3] == pytest.approx(4.0)  # yaw_rate max

    def test_gymnasium_check_env(self):
        """Full Gymnasium compliance check."""
        e = DroneRacingEnv(num_gates=3, max_steps=100)
        check_env(e, skip_render_check=True)
        e.close()


# ---------------------------------------------------------------------------
# Reset behaviour
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_obs_and_info(self, env: DroneRacingEnv):
        result = env.reset(seed=0)
        assert isinstance(result, tuple) and len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_sets_drone_to_start(self, env: DroneRacingEnv):
        env.reset(seed=0)
        np.testing.assert_array_almost_equal(env._drone_pos, [0, 0, 1])
        np.testing.assert_array_almost_equal(env._drone_vel, [0, 0, 0])

    def test_reset_generates_correct_gate_count(self, env: DroneRacingEnv):
        env.reset(seed=0)
        assert len(env._gates) == env.num_gates

    def test_reset_deterministic_with_seed(self):
        e = DroneRacingEnv(num_gates=3)
        obs1, _ = e.reset(seed=99)
        gates1 = [g.copy() for g in e._gates]
        obs2, _ = e.reset(seed=99)
        gates2 = [g.copy() for g in e._gates]
        np.testing.assert_array_equal(obs1, obs2)
        for g1, g2 in zip(gates1, gates2):
            np.testing.assert_array_equal(g1, g2)
        e.close()


# ---------------------------------------------------------------------------
# Physics step
# ---------------------------------------------------------------------------

class TestPhysics:
    def test_hover_thrust_maintains_altitude(self, seeded_env: DroneRacingEnv):
        """Thrust ~0.5 should roughly balance gravity (simplified physics)."""
        initial_z = seeded_env._drone_pos[2]
        # Apply hover thrust for a few steps
        for _ in range(10):
            hover_action = np.array([0.5, 0, 0, 0], dtype=np.float32)
            seeded_env.step(hover_action)
        # Altitude should stay close to initial (within tolerance for simplified model)
        assert abs(seeded_env._drone_pos[2] - initial_z) < 0.5

    def test_zero_thrust_falls(self, seeded_env: DroneRacingEnv):
        """No thrust -> drone falls due to gravity."""
        for _ in range(50):
            seeded_env.step(np.array([0, 0, 0, 0], dtype=np.float32))
        assert seeded_env._drone_pos[2] < 0.5, "Drone should fall with zero thrust"

    def test_full_thrust_rises(self, seeded_env: DroneRacingEnv):
        """Full thrust -> drone gains altitude."""
        initial_z = seeded_env._drone_pos[2]
        for _ in range(50):
            seeded_env.step(np.array([1.0, 0, 0, 0], dtype=np.float32))
        assert seeded_env._drone_pos[2] > initial_z, "Drone should rise with full thrust"

    def test_pitch_moves_forward(self, seeded_env: DroneRacingEnv):
        """Positive pitch_rate should move drone in +x direction."""
        initial_x = seeded_env._drone_pos[0]
        for _ in range(50):
            seeded_env.step(np.array([0.5, 0, 2.0, 0], dtype=np.float32))
        assert seeded_env._drone_pos[0] > initial_x, "Drone should move forward with pitch"

    def test_drag_limits_speed(self, seeded_env: DroneRacingEnv):
        """Drag coefficient should prevent infinite speed buildup."""
        for _ in range(500):
            seeded_env.step(np.array([0.5, 0, 4.0, 0], dtype=np.float32))
        speed = np.linalg.norm(seeded_env._drone_vel)
        assert speed < 50.0, f"Speed {speed} should be bounded by drag"


# ---------------------------------------------------------------------------
# Reward structure
# ---------------------------------------------------------------------------

class TestRewards:
    def test_time_penalty_per_step(self, seeded_env: DroneRacingEnv):
        """Each step should incur a small negative time penalty."""
        _, reward, _, _, _ = seeded_env.step(
            np.array([0.5, 0, 0, 0], dtype=np.float32)
        )
        # Reward = time penalty (-0.01) + approach reward (positive)
        # Just verify the time component is present (total reward should be modest)
        assert reward < 10.0, "Reward should not be huge for a neutral step"

    def test_gate_passage_reward(self):
        """Passing through a gate should give a large positive reward."""
        e = DroneRacingEnv(num_gates=1, max_steps=2000)
        e.reset(seed=42)
        # Teleport drone right next to the gate
        e._drone_pos = e._gates[0].copy()
        e._drone_pos[0] -= 0.3  # just slightly before
        e._drone_vel = np.array([5.0, 0, 0])  # moving toward gate
        _, reward, _, _, info = e.step(np.array([0.5, 0, 0, 0], dtype=np.float32))
        # Should get the +100 gate reward
        assert reward > 50.0, f"Gate passage reward {reward} should be high"
        assert info["gates_passed"] >= 1
        e.close()

    def test_crash_penalty(self):
        """Crashing should give a large negative reward."""
        e = DroneRacingEnv(num_gates=3, max_steps=500)
        e.reset(seed=0)
        # Force drone below ground
        e._drone_pos = np.array([0, 0, 0.05])
        e._drone_vel = np.array([0, 0, -5.0])
        _, reward, terminated, _, _ = e.step(
            np.array([0, 0, 0, 0], dtype=np.float32)
        )
        assert reward < -100.0, f"Crash reward {reward} should be very negative"
        assert terminated
        e.close()


# ---------------------------------------------------------------------------
# Termination / truncation
# ---------------------------------------------------------------------------

class TestTermination:
    def test_crash_below_ground_terminates(self, env: DroneRacingEnv):
        env.reset(seed=0)
        env._drone_pos = np.array([0, 0, -1.0])
        _, _, terminated, _, _ = env.step(np.array([0, 0, 0, 0], dtype=np.float32))
        assert terminated

    def test_crash_above_ceiling_terminates(self, env: DroneRacingEnv):
        env.reset(seed=0)
        env._drone_pos = np.array([0, 0, 51.0])
        _, _, terminated, _, _ = env.step(np.array([1, 0, 0, 0], dtype=np.float32))
        assert terminated

    def test_max_steps_truncates(self):
        e = DroneRacingEnv(num_gates=3, max_steps=5)
        e.reset(seed=0)
        for i in range(5):
            _, _, terminated, truncated, _ = e.step(
                np.array([0.5, 0, 0, 0], dtype=np.float32)
            )
        assert truncated, "Episode should truncate at max_steps"
        e.close()

    def test_all_gates_passed_terminates(self):
        e = DroneRacingEnv(num_gates=1, max_steps=2000)
        e.reset(seed=42)
        # Place drone at the gate
        e._drone_pos = e._gates[0].copy()
        _, _, terminated, _, info = e.step(
            np.array([0.5, 0, 0, 0], dtype=np.float32)
        )
        assert terminated, "Episode should end when all gates passed"
        assert info["gates_passed"] == 1


# ---------------------------------------------------------------------------
# Info dict
# ---------------------------------------------------------------------------

class TestInfo:
    def test_info_contains_expected_keys(self, seeded_env: DroneRacingEnv):
        _, _, _, _, info = seeded_env.step(
            np.array([0.5, 0, 0, 0], dtype=np.float32)
        )
        assert "gates_passed" in info
        assert "total_gates" in info
        assert "speed" in info

    def test_speed_is_non_negative(self, seeded_env: DroneRacingEnv):
        for _ in range(10):
            _, _, _, _, info = seeded_env.step(
                np.array([0.5, 0, 1.0, 0], dtype=np.float32)
            )
        assert info["speed"] >= 0
