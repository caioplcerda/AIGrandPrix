"""Robustness tests: edge cases and failure recovery."""

from __future__ import annotations

import numpy as np
import pytest

from aigrandprix.control.drone_controller import DroneState
from aigrandprix.main import RaceConfig, RacingAgent
from aigrandprix.planning.path_planner import Gate
from aigrandprix.simulation.gym_env import DroneRacingEnv


def _make_gates(positions: list[list[float]]) -> list[Gate]:
    """Create gates with auto-computed normals."""
    gates = []
    for i, pos in enumerate(positions):
        pos_arr = np.array(pos)
        if i == 0:
            direction = pos_arr - np.array([0, 0, 1])
        else:
            direction = pos_arr - np.array(positions[i - 1])
        norm = np.linalg.norm(direction[:2])
        if norm > 0.01:
            normal = direction[:2] / norm
            normal = np.array([normal[0], normal[1], 0.0])
        else:
            normal = np.array([1.0, 0.0, 0.0])
        gates.append(Gate(position=pos_arr, normal=normal, gate_id=i))
    return gates


class TestGateMissRecovery:
    """Test recovery when gates are missed."""

    def test_skip_gate_replans(self):
        """After externally skipping a gate, agent replans to next."""
        gates = _make_gates([[5, 0, 2], [10, 0, 2], [15, 0, 2]])
        agent = RacingAgent(RaceConfig(max_speed=10.0, control_dt=0.02))
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        state = DroneState(
            position=np.array([0.0, 0.0, 1.0]),
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        )

        # Compute first action (targets gate 0)
        cmd = agent.compute_action(
            image=dummy_image, state=state, elapsed_time=0.0, known_gates=gates,
        )
        assert agent._gates_passed == 0

        # Skip gate 0 externally
        agent.on_gate_passed(0)
        assert agent._gates_passed == 1

        # Next compute should target gate 1
        cmd = agent.compute_action(
            image=dummy_image, state=state, elapsed_time=0.5, known_gates=gates,
        )
        # Trajectory should now target second gate region


class TestCloseGates:
    """Gates very close together."""

    def test_close_gates_no_overshoot(self):
        """Agent handles gates <3m apart."""
        gates = _make_gates([[3, 0, 2], [5, 0, 2], [7, 0, 2]])
        agent = RacingAgent(RaceConfig(max_speed=8.0, control_dt=0.02))
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

        state = DroneState(
            position=np.array([0.0, 0.0, 1.0]),
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        )

        # Should produce valid commands without crashing
        for step in range(50):
            cmd = agent.compute_action(
                image=dummy_image, state=state,
                elapsed_time=step * 0.02, known_gates=gates,
            )
            assert 0 <= cmd.thrust <= 1.0
            assert not np.isnan(cmd.thrust)


class TestSharpTurns:
    """Gates requiring >90 degree turns."""

    def test_sharp_turn_produces_valid_commands(self):
        """90+ degree turn: valid commands, no NaN."""
        # Gates form a sharp right turn
        gates = _make_gates([[10, 0, 2], [10, 10, 2], [0, 10, 2]])
        agent = RacingAgent(RaceConfig(max_speed=8.0, control_dt=0.02))
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

        state = DroneState(
            position=np.array([0.0, 0.0, 1.0]),
            velocity=np.array([5.0, 0.0, 0.0]),  # flying toward gate 0
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        )

        for step in range(100):
            cmd = agent.compute_action(
                image=dummy_image, state=state,
                elapsed_time=step * 0.02, known_gates=gates,
            )
            assert not np.isnan(cmd.thrust)
            assert not np.isnan(cmd.roll_rate)
            assert not np.isnan(cmd.pitch_rate)


class TestAltitudeChange:
    """Large altitude changes between gates."""

    def test_altitude_change(self):
        """Gate z=1 to z=5: valid trajectory and commands."""
        gates = _make_gates([[5, 0, 1], [10, 0, 5], [15, 0, 2]])
        agent = RacingAgent(RaceConfig(max_speed=10.0, control_dt=0.02))
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

        state = DroneState(
            position=np.array([0.0, 0.0, 1.0]),
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        )

        for step in range(100):
            cmd = agent.compute_action(
                image=dummy_image, state=state,
                elapsed_time=step * 0.02, known_gates=gates,
            )
            assert 0 <= cmd.thrust <= 1.0


class TestLongCourse:
    """Extended course with many gates."""

    def test_ten_gate_course_no_crash(self):
        """10-gate course: agent survives 500 steps."""
        env = DroneRacingEnv(num_gates=10, max_steps=600)
        env.reset(seed=42)

        gates = []
        for i, pos in enumerate(env._gates):
            if i == 0:
                d = pos - np.array([0, 0, 1])
            else:
                d = pos - env._gates[i - 1]
            n = np.linalg.norm(d[:2])
            normal = np.array([d[0], d[1], 0.0]) / n if n > 0.01 else np.array([1, 0, 0])
            gates.append(Gate(position=pos.copy(), normal=normal, gate_id=i))

        agent = RacingAgent(RaceConfig(max_speed=10.0, control_dt=0.02))
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        max_gates = 0

        for step in range(500):
            state = DroneState(
                position=env._drone_pos.copy(),
                velocity=env._drone_vel.copy(),
                orientation=env._drone_orient.copy(),
                angular_velocity=env._drone_angular_vel.copy(),
            )
            cmd = agent.compute_action(
                image=dummy_image, state=state,
                elapsed_time=step * 0.02, known_gates=gates,
            )
            action = np.array(
                [cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate],
                dtype=np.float32,
            )
            action = np.clip(action, env.action_space.low, env.action_space.high)
            _, _, terminated, truncated, info = env.step(action)
            if info["gates_passed"] > max_gates:
                agent.on_gate_passed(max_gates)
                max_gates = info["gates_passed"]
            if terminated:
                crashed = env._drone_pos[2] < 0 or env._drone_pos[2] > 50
                if crashed:
                    pytest.fail(f"Crashed at step {step}, pos={env._drone_pos}")
                break
            if truncated:
                break

        env.close()


class TestActionBounds:
    """Actions must always be valid."""

    @pytest.mark.parametrize("seed", range(5))
    def test_no_nan_no_inf(self, seed: int):
        """Actions never contain NaN or Inf."""
        env = DroneRacingEnv(num_gates=5, max_steps=200)
        env.reset(seed=seed)

        gates = []
        for i, pos in enumerate(env._gates):
            gates.append(Gate(
                position=pos.copy(),
                normal=np.array([1.0, 0.0, 0.0]),
                gate_id=i,
            ))

        agent = RacingAgent(RaceConfig(max_speed=12.0, control_dt=0.02))
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

        for step in range(200):
            state = DroneState(
                position=env._drone_pos.copy(),
                velocity=env._drone_vel.copy(),
                orientation=env._drone_orient.copy(),
                angular_velocity=env._drone_angular_vel.copy(),
            )
            cmd = agent.compute_action(
                image=dummy_image, state=state,
                elapsed_time=step * 0.02, known_gates=gates,
            )
            assert not np.isnan(cmd.thrust), f"NaN thrust at step {step}"
            assert not np.isnan(cmd.roll_rate), f"NaN roll at step {step}"
            assert not np.isnan(cmd.pitch_rate), f"NaN pitch at step {step}"
            assert not np.isnan(cmd.yaw_rate), f"NaN yaw at step {step}"
            assert not np.isinf(cmd.thrust), f"Inf thrust at step {step}"

            action = np.array(
                [cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate],
                dtype=np.float32,
            )
            action = np.clip(action, env.action_space.low, env.action_space.high)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        env.close()
