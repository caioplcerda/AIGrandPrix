"""Gymnasium environment for drone racing training.

Wraps drone simulation into a standard Gymnasium interface for
reinforcement learning training with stable-baselines3.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aigrandprix.utils.math_utils import euler_to_quat, quat_to_euler


class DroneRacingEnv(gym.Env):
    """Drone racing Gymnasium environment.

    Observation space: drone state + next gate relative position
    Action space: continuous [thrust, roll_rate, pitch_rate, yaw_rate]

    Reward: progress toward next gate - time penalty - crash penalty
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Fixed normalization scales (shape 19): pos, vel, orient, angular_vel, gate_rel, gate_normal
    _OBS_SCALES = np.array(
        [50, 50, 50, 30, 30, 30, 1, 1, 1, 1, 10, 10, 10, 50, 50, 50, 1, 1, 1],
        dtype=np.float32,
    )

    def __init__(
        self,
        num_gates: int = 5,
        max_steps: int = 2000,
        render_mode: str | None = None,
        normalize_obs: bool = False,
        normalize_actions: bool = False,
        use_dynamics_model: bool = False,
        config: dict | None = None,
    ) -> None:
        super().__init__()
        self.num_gates = num_gates
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.normalize_obs = normalize_obs
        self.normalize_actions = normalize_actions
        self.use_dynamics_model = use_dynamics_model

        # Action space
        if normalize_actions:
            self.action_space = spaces.Box(
                low=-np.ones(4, dtype=np.float32),
                high=np.ones(4, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([0.0, -8.0, -8.0, -4.0]),
                high=np.array([1.0, 8.0, 8.0, 4.0]),
                dtype=np.float32,
            )

        # Observation space
        if normalize_obs:
            self.observation_space = spaces.Box(
                low=-np.ones(19, dtype=np.float32),
                high=np.ones(19, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(19,),
                dtype=np.float32,
            )

        # Dynamics model (lazy init)
        self._dynamics = None
        if use_dynamics_model:
            self._init_dynamics(config)

        self._step_count = 0
        self._current_gate_idx = 0
        self._gates: list[np.ndarray] = []
        self._drone_pos = np.zeros(3)
        self._drone_vel = np.zeros(3)
        self._drone_orient = np.array([1.0, 0, 0, 0])
        self._drone_angular_vel = np.zeros(3)
        # Euler angles tracked from integrated angular rates (roll, pitch, yaw)
        self._drone_rpy = np.zeros(3)

    def _init_dynamics(self, config: dict | None) -> None:
        from aigrandprix.control.drone_dynamics import DroneParams, QuadrotorDynamics

        if config and "drone" in config:
            params = DroneParams.from_config(config["drone"])
        else:
            params = DroneParams()
        self._dynamics = QuadrotorDynamics(params)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self._step_count = 0
        self._current_gate_idx = 0

        # Generate random gate sequence
        self._gates = self._generate_gates()

        # Reset drone to start position
        self._drone_pos = np.array([0.0, 0.0, 1.0])
        self._drone_vel = np.zeros(3)
        self._drone_orient = np.array([1.0, 0, 0, 0])
        self._drone_angular_vel = np.zeros(3)
        self._drone_rpy = np.zeros(3)

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1

        # Denormalize actions if needed
        if self.normalize_actions:
            thrust = (action[0] + 1.0) / 2.0
            roll_r = action[1] * 8.0
            pitch_r = action[2] * 8.0
            yaw_r = action[3] * 4.0
        else:
            thrust, roll_r, pitch_r, yaw_r = action

        if self.use_dynamics_model:
            self._step_dynamics(thrust, roll_r, pitch_r, yaw_r)
        else:
            self._step_simplified(thrust, roll_r, pitch_r, yaw_r)

        # Check gate passage
        reward = 0.0
        if self._current_gate_idx < len(self._gates):
            gate_pos = self._gates[self._current_gate_idx]
            dist = np.linalg.norm(self._drone_pos - gate_pos)

            # Progress reward
            reward = -0.01  # time penalty

            if dist < 1.0:  # passed through gate
                reward += 100.0
                self._current_gate_idx += 1

            # Approach reward
            reward += max(0, 1.0 - dist * 0.1)

        # Termination conditions
        crashed = self._drone_pos[2] < 0.0 or self._drone_pos[2] > 50.0
        completed = self._current_gate_idx >= len(self._gates)
        truncated = self._step_count >= self.max_steps

        if crashed:
            reward -= 200.0

        terminated = crashed or completed

        info = {
            "gates_passed": self._current_gate_idx,
            "total_gates": len(self._gates),
            "speed": float(np.linalg.norm(self._drone_vel)),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _step_simplified(
        self, thrust: float, roll_r: float, pitch_r: float, yaw_r: float
    ) -> None:
        """Original simplified physics — byte-for-byte identical to pre-refactor."""
        dt = 0.02

        # Integrate angular rates -> Euler angles (roll, pitch, yaw)
        self._drone_rpy += np.array([roll_r, pitch_r, yaw_r]) * dt
        # Dampen angles toward zero (aerodynamic restoring moment)
        self._drone_rpy *= 0.98
        # Clamp to physically reasonable range
        self._drone_rpy[0] = np.clip(self._drone_rpy[0], -np.pi / 3, np.pi / 3)  # roll ±60°
        self._drone_rpy[1] = np.clip(self._drone_rpy[1], -np.pi / 3, np.pi / 3)  # pitch ±60°
        self._drone_angular_vel = np.array([roll_r, pitch_r, yaw_r])

        # Update quaternion from Euler angles for observation consistency
        self._drone_orient = euler_to_quat(self._drone_rpy)

        # Simplified dynamics
        accel = np.array([
            pitch_r * 0.5,
            -roll_r * 0.5,
            (thrust - 0.5) * 20.0 - 9.81,
        ])
        self._drone_vel += accel * dt
        self._drone_vel *= 0.99  # drag
        self._drone_pos += self._drone_vel * dt

    def _step_dynamics(
        self, thrust: float, roll_r: float, pitch_r: float, yaw_r: float
    ) -> None:
        """Full Newton-Euler dynamics via QuadrotorDynamics."""
        from aigrandprix.control.drone_controller import DroneState

        dt = 0.02
        state = DroneState(
            position=self._drone_pos.copy(),
            velocity=self._drone_vel.copy(),
            orientation=self._drone_orient.copy(),
            angular_velocity=self._drone_angular_vel.copy(),
        )
        motor_thrusts = self._dynamics.command_to_motor_thrusts(thrust, roll_r, pitch_r, yaw_r)
        new_state = self._dynamics.step(state, motor_thrusts, dt)

        self._drone_pos = new_state.position
        self._drone_vel = new_state.velocity
        self._drone_orient = new_state.orientation
        self._drone_angular_vel = new_state.angular_velocity
        self._drone_rpy = quat_to_euler(new_state.orientation)

    def _get_obs(self) -> np.ndarray:
        if self._current_gate_idx < len(self._gates):
            gate_pos = self._gates[self._current_gate_idx]
            gate_rel = gate_pos - self._drone_pos
            gate_normal = np.array([1, 0, 0])  # placeholder
        else:
            gate_rel = np.zeros(3)
            gate_normal = np.zeros(3)

        obs = np.concatenate([
            self._drone_pos,
            self._drone_vel,
            self._drone_orient,
            self._drone_angular_vel,
            gate_rel,
            gate_normal,
        ]).astype(np.float32)

        if self.normalize_obs:
            obs = np.clip(obs / self._OBS_SCALES, -1.0, 1.0)

        return obs

    def _generate_gates(self) -> list[np.ndarray]:
        """Generate a random gate course."""
        gates = []
        pos = np.array([5.0, 0.0, 2.0])
        for _ in range(self.num_gates):
            pos = pos + self.np_random.uniform(
                low=[3.0, -3.0, -1.0],
                high=[8.0, 3.0, 1.0],
            )
            pos[2] = np.clip(pos[2], 1.0, 5.0)
            gates.append(pos.copy())
        return gates
