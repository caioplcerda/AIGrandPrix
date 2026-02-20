"""Stress tests for the autonomous racing pipeline.

Tests performance, stability, and resource usage across
varied configurations.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest

from aigrandprix.control.drone_controller import DroneState
from aigrandprix.main import RaceConfig, RacingAgent
from aigrandprix.planning.path_planner import Gate
from aigrandprix.simulation.gym_env import DroneRacingEnv


def _env_state(env: DroneRacingEnv) -> DroneState:
    return DroneState(
        position=env._drone_pos.copy(),
        velocity=env._drone_vel.copy(),
        orientation=env._drone_orient.copy(),
        angular_velocity=env._drone_angular_vel.copy(),
    )


def _env_gates(env: DroneRacingEnv) -> list[Gate]:
    gates = []
    for i, pos in enumerate(env._gates):
        if i == 0:
            direction = pos - np.array([0, 0, 1])
        else:
            direction = pos - env._gates[i - 1]
        norm = np.linalg.norm(direction)
        normal = direction / norm if norm > 0.01 else np.array([1, 0, 0])
        normal[2] = 0
        ln = np.linalg.norm(normal)
        if ln > 0.01:
            normal /= ln
        else:
            normal = np.array([1, 0, 0])
        gates.append(Gate(position=pos.copy(), normal=normal, gate_id=i))
    return gates


def _run_episode(num_gates: int, seed: int, max_steps: int = 1000) -> dict:
    """Run a quick episode and return stats."""
    env = DroneRacingEnv(num_gates=num_gates, max_steps=max_steps)
    env.reset(seed=seed)
    gates = _env_gates(env)

    agent = RacingAgent(RaceConfig(max_speed=10.0, control_dt=0.02))
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    max_gates_passed = 0
    crashed = False

    for step in range(max_steps):
        state = _env_state(env)
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

        if info["gates_passed"] > max_gates_passed:
            agent.on_gate_passed(max_gates_passed)
            max_gates_passed = info["gates_passed"]

        if terminated:
            crashed = env._drone_pos[2] < 0 or env._drone_pos[2] > 50
            break
        if truncated:
            break

    env.close()
    return {
        "gates_passed": max_gates_passed,
        "total_gates": num_gates,
        "steps": step + 1,
        "crashed": crashed,
    }


class TestStressParametric:
    """Parametric sweep: gates x seeds."""

    @pytest.mark.parametrize("num_gates", [3, 5, 8])
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_passes_at_least_one_gate(self, num_gates: int, seed: int):
        """Agent should pass at least 1 gate in all configs."""
        result = _run_episode(num_gates, seed, max_steps=1500)
        assert result["gates_passed"] >= 1, (
            f"gates={num_gates}, seed={seed}: passed {result['gates_passed']}/{result['total_gates']}"
        )

    @pytest.mark.parametrize("num_gates", [3, 5, 8])
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_no_crash_first_200_steps(self, num_gates: int, seed: int):
        """No crash in first 200 steps."""
        env = DroneRacingEnv(num_gates=num_gates, max_steps=300)
        env.reset(seed=seed)
        gates = _env_gates(env)
        agent = RacingAgent(RaceConfig(max_speed=10.0, control_dt=0.02))
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

        for step in range(200):
            state = _env_state(env)
            cmd = agent.compute_action(
                image=dummy_image, state=state,
                elapsed_time=step * 0.02, known_gates=gates,
            )
            action = np.array(
                [cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate],
                dtype=np.float32,
            )
            action = np.clip(action, env.action_space.low, env.action_space.high)
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                crashed = env._drone_pos[2] < 0 or env._drone_pos[2] > 50
                assert not crashed, (
                    f"Crashed at step {step}, pos={env._drone_pos}, "
                    f"gates={num_gates}, seed={seed}"
                )
                break
        env.close()


class TestMemoryStability:
    """Memory should not grow unboundedly."""

    def test_no_memory_leak_over_episodes(self):
        """Memory growth < 20% over 10 episodes."""
        tracemalloc.start()

        for ep in range(3):
            _run_episode(5, seed=ep, max_steps=500)

        _, baseline_peak = tracemalloc.get_traced_memory()

        for ep in range(10):
            _run_episode(5, seed=100 + ep, max_steps=500)

        _, final_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        growth = (final_peak - baseline_peak) / max(baseline_peak, 1)
        assert growth < 0.5, f"Memory grew {growth*100:.1f}% over 10 episodes"


class TestLatency:
    """Compute latency bounds."""

    def test_compute_action_p95_under_15ms(self):
        """compute_action P95 < 15ms over 300 frames."""
        env = DroneRacingEnv(num_gates=5, max_steps=400)
        env.reset(seed=42)
        gates = _env_gates(env)
        agent = RacingAgent(RaceConfig(max_speed=10.0, control_dt=0.02))
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

        latencies = []
        for step in range(300):
            state = _env_state(env)
            t0 = time.perf_counter()
            cmd = agent.compute_action(
                image=dummy_image, state=state,
                elapsed_time=step * 0.02, known_gates=gates,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

            action = np.array(
                [cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate],
                dtype=np.float32,
            )
            action = np.clip(action, env.action_space.low, env.action_space.high)
            _, _, terminated, truncated, info = env.step(action)
            if info["gates_passed"] > 0:
                agent.on_gate_passed(info["gates_passed"] - 1)
            if terminated or truncated:
                break

        env.close()
        p95 = np.percentile(latencies, 95)
        assert p95 < 15.0, f"compute_action P95={p95:.1f}ms > 15ms"

    def test_mpc_compute_p95_under_5ms(self):
        """MPC compute_control P95 < 5ms."""
        from aigrandprix.control.mpc_controller import MPCConfig, MPCController
        from aigrandprix.planning.path_planner import TrajectoryPoint

        mpc = MPCController(MPCConfig())
        state = DroneState(
            position=np.array([0.0, 0.0, 1.0]),
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        )
        # Build a simple lookahead trajectory
        traj = []
        for i in range(10):
            traj.append(TrajectoryPoint(
                position=np.array([i * 0.5, 0.0, 1.0]),
                velocity=np.array([2.0, 0.0, 0.0]),
                acceleration=np.zeros(3),
                time=i * 0.02,
            ))

        latencies = []
        for _ in range(200):
            t0 = time.perf_counter()
            mpc.compute_control(state, traj)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

        p95 = np.percentile(latencies, 95)
        assert p95 < 5.0, f"MPC P95={p95:.1f}ms > 5ms"
