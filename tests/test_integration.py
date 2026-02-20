"""Integration tests: full racing pipeline running in the gym environment.

Tests the RacingAgent (planner + controller) driving through gates
inside DroneRacingEnv without using vision (known gate positions).
"""

from __future__ import annotations

import numpy as np
import pytest

from aigrandprix.control.drone_controller import DroneState
from aigrandprix.main import RaceConfig, RacingAgent
from aigrandprix.planning.path_planner import Gate
from aigrandprix.simulation.gym_env import DroneRacingEnv


def _env_state_to_drone_state(env: DroneRacingEnv) -> DroneState:
    """Extract DroneState from the environment's internal state."""
    return DroneState(
        position=env._drone_pos.copy(),
        velocity=env._drone_vel.copy(),
        orientation=env._drone_orient.copy(),
        angular_velocity=env._drone_angular_vel.copy(),
    )


def _env_gates_to_gate_list(env: DroneRacingEnv) -> list[Gate]:
    """Convert environment gates to Gate dataclass list."""
    gates = []
    for i, pos in enumerate(env._gates):
        # Approximate normal as direction from previous gate (or from origin)
        if i == 0:
            direction = pos - np.array([0, 0, 1])
        else:
            direction = pos - env._gates[i - 1]
        norm = np.linalg.norm(direction)
        normal = direction / norm if norm > 0.01 else np.array([1, 0, 0])
        normal[2] = 0  # keep normal horizontal
        ln = np.linalg.norm(normal)
        if ln > 0.01:
            normal /= ln
        else:
            normal = np.array([1, 0, 0])
        gates.append(Gate(position=pos.copy(), normal=normal, gate_id=i))
    return gates


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestRacingPipeline:
    """Test the full plan -> control -> simulate loop."""

    def test_agent_passes_at_least_one_gate(self):
        """The racing agent should pass at least one gate in a full episode."""
        env = DroneRacingEnv(num_gates=3, max_steps=1500)
        obs, _ = env.reset(seed=42)
        gates = _env_gates_to_gate_list(env)

        config = RaceConfig(max_speed=10.0, control_dt=0.02)
        agent = RacingAgent(config=config)

        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        elapsed = 0.0
        dt = 0.02
        max_gates_passed = 0

        for step in range(1500):
            state = _env_state_to_drone_state(env)
            cmd = agent.compute_action(
                image=dummy_image,
                state=state,
                elapsed_time=elapsed,
                known_gates=gates,
            )

            action = np.array(
                [cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate],
                dtype=np.float32,
            )
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            elapsed += dt

            # Track gate passage and notify agent
            if info["gates_passed"] > max_gates_passed:
                agent.on_gate_passed(max_gates_passed)
                max_gates_passed = info["gates_passed"]

            if terminated or truncated:
                break

        env.close()
        assert max_gates_passed >= 1, (
            f"Agent should pass at least 1 gate, but passed {max_gates_passed}"
        )

    def test_agent_does_not_crash_immediately(self):
        """Agent should survive at least 100 steps without crashing."""
        env = DroneRacingEnv(num_gates=3, max_steps=2000)
        obs, _ = env.reset(seed=7)
        gates = _env_gates_to_gate_list(env)

        agent = RacingAgent(RaceConfig(max_speed=8.0, control_dt=0.02))
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

        for step in range(100):
            state = _env_state_to_drone_state(env)
            cmd = agent.compute_action(
                image=dummy_image,
                state=state,
                elapsed_time=step * 0.02,
                known_gates=gates,
            )
            action = np.array(
                [cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate],
                dtype=np.float32,
            )
            action = np.clip(action, env.action_space.low, env.action_space.high)
            _, _, terminated, _, _ = env.step(action)
            assert not terminated, f"Agent crashed at step {step}"

        env.close()

    def test_agent_resets_cleanly(self):
        """Agent should be resettable and work across multiple episodes."""
        env = DroneRacingEnv(num_gates=2, max_steps=200)

        for episode in range(3):
            obs, _ = env.reset(seed=episode)
            gates = _env_gates_to_gate_list(env)
            agent = RacingAgent(RaceConfig(control_dt=0.02))
            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

            for step in range(50):
                state = _env_state_to_drone_state(env)
                cmd = agent.compute_action(
                    image=dummy_image,
                    state=state,
                    elapsed_time=step * 0.02,
                    known_gates=gates,
                )
                action = np.array(
                    [cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate],
                    dtype=np.float32,
                )
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

        env.close()


class TestControllerInEnv:
    """Test the controller directly in the simulation loop."""

    def test_position_tracking_converges(self):
        """Controller tracking a fixed position should converge."""
        from aigrandprix.control.drone_controller import DroneController

        env = DroneRacingEnv(num_gates=1, max_steps=500)
        env.reset(seed=0)

        controller = DroneController(dt=0.02)
        target = np.array([3.0, 0.0, 2.0])

        for step in range(300):
            state = _env_state_to_drone_state(env)
            cmd = controller.track_position(state, target)
            action = np.array(
                [cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate],
                dtype=np.float32,
            )
            action = np.clip(action, env.action_space.low, env.action_space.high)
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                break

        dist = np.linalg.norm(env._drone_pos - target)
        assert dist < 3.0, f"Controller should converge toward target, dist={dist:.2f}"
        env.close()


class TestFullDynamicsIntegration:
    """Test full dynamics mode in the simulation loop."""

    def test_full_dynamics_no_crash_100_steps(self):
        """Full dynamics: no crash for 100 steps."""
        env = DroneRacingEnv(num_gates=3, max_steps=200, use_dynamics_model=True)
        obs, _ = env.reset(seed=42)
        gates = _env_gates_to_gate_list(env)
        agent = RacingAgent(
            RaceConfig(max_speed=8.0, control_dt=0.02),
            full_config={"control": {"physics_mode": "full", "attitude": {"kp": 8.0, "kd": 2.0}}, "simulation": {"use_dynamics_model": True}},
        )
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        for step in range(100):
            state = _env_state_to_drone_state(env)
            cmd = agent.compute_action(image=dummy_image, state=state, elapsed_time=step * 0.02, known_gates=gates)
            action = np.array([cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate], dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            _, _, terminated, _, _ = env.step(action)
            assert not terminated, f"Full dynamics crashed at step {step}"
        env.close()

    def test_full_dynamics_passes_gate(self):
        """Full dynamics: pass at least 1 gate."""
        env = DroneRacingEnv(num_gates=3, max_steps=2000, use_dynamics_model=True)
        obs, _ = env.reset(seed=42)
        gates = _env_gates_to_gate_list(env)
        agent = RacingAgent(
            RaceConfig(max_speed=8.0, control_dt=0.02),
            full_config={"control": {"physics_mode": "full", "attitude": {"kp": 8.0, "kd": 2.0}}, "simulation": {"use_dynamics_model": True}},
        )
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        max_gates = 0
        for step in range(2000):
            state = _env_state_to_drone_state(env)
            cmd = agent.compute_action(image=dummy_image, state=state, elapsed_time=step * 0.02, known_gates=gates)
            action = np.array([cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate], dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            _, _, terminated, truncated, info = env.step(action)
            if info["gates_passed"] > max_gates:
                agent.on_gate_passed(max_gates)
                max_gates = info["gates_passed"]
            if terminated or truncated:
                break
        env.close()
        assert max_gates >= 1, f"Full dynamics should pass at least 1 gate, got {max_gates}"


class TestPlannerInEnv:
    """Test the planner generating valid trajectories for env gates."""

    def test_planner_generates_trajectory_for_env_gates(self):
        """Planner should produce a trajectory for gates from the env."""
        from aigrandprix.planning.path_planner import PathPlanner

        env = DroneRacingEnv(num_gates=5, max_steps=2000)
        env.reset(seed=42)
        gates = _env_gates_to_gate_list(env)

        planner = PathPlanner(max_speed=15.0, max_accel=10.0)
        traj = planner.plan_through_gates(gates, start_pos=np.array([0, 0, 1]))

        assert len(traj) > 0, "Should generate non-empty trajectory"
        # First point should be near start
        assert np.linalg.norm(traj[0].position - np.array([0, 0, 1])) < 1.0
        # Trajectory should pass near each gate
        for gate in gates:
            min_dist = min(np.linalg.norm(pt.position - gate.position) for pt in traj)
            assert min_dist < 3.0, (
                f"Trajectory should pass near gate {gate.gate_id}, min_dist={min_dist:.2f}"
            )

        env.close()
