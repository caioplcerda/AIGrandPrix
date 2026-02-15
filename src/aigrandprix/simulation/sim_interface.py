"""Abstract simulator interface and internal adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from aigrandprix.control.drone_controller import DroneState
from aigrandprix.planning.path_planner import Gate


class SimInterface(ABC):
    """Abstract base class for drone racing simulators."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> dict:
        """Reset the simulator. Returns info dict."""

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[bool, bool, dict]:
        """Advance one timestep. Returns (terminated, truncated, info)."""

    @abstractmethod
    def get_state(self) -> DroneState:
        """Get current drone state."""

    @abstractmethod
    def get_image(self) -> np.ndarray:
        """Get current camera image (H, W, 3) BGR."""

    @abstractmethod
    def get_gate_poses(self) -> list[Gate]:
        """Get list of all racing gates with positions and normals."""

    @abstractmethod
    def get_next_gate_index(self) -> int:
        """Get the index of the next gate to pass."""

    @property
    @abstractmethod
    def dt(self) -> float:
        """Simulation timestep in seconds."""

    @abstractmethod
    def close(self) -> None:
        """Clean up simulator resources."""


class InternalSimAdapter(SimInterface):
    """Wraps DroneRacingEnv behind the SimInterface ABC."""

    def __init__(self, env) -> None:
        from aigrandprix.simulation.gym_env import DroneRacingEnv

        if not isinstance(env, DroneRacingEnv):
            raise TypeError(f"Expected DroneRacingEnv, got {type(env)}")
        self._env = env

    def reset(self, seed: int | None = None) -> dict:
        _, info = self._env.reset(seed=seed)
        return info

    def step(self, action: np.ndarray) -> tuple[bool, bool, dict]:
        _, _, terminated, truncated, info = self._env.step(action)
        return terminated, truncated, info

    def get_state(self) -> DroneState:
        return DroneState(
            position=self._env._drone_pos.copy(),
            velocity=self._env._drone_vel.copy(),
            orientation=self._env._drone_orient.copy(),
            angular_velocity=self._env._drone_angular_vel.copy(),
        )

    def get_image(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def get_gate_poses(self) -> list[Gate]:
        gates = []
        for i, pos in enumerate(self._env._gates):
            if i == 0:
                direction = pos - np.array([0.0, 0.0, 1.0])
            else:
                direction = pos - self._env._gates[i - 1]
            norm = np.linalg.norm(direction)
            normal = direction / norm if norm > 0.01 else np.array([1.0, 0.0, 0.0])
            normal[2] = 0.0
            ln = np.linalg.norm(normal)
            if ln > 0.01:
                normal = normal / ln
            else:
                normal = np.array([1.0, 0.0, 0.0])
            gates.append(Gate(position=pos.copy(), normal=normal, gate_id=i))
        return gates

    def get_next_gate_index(self) -> int:
        return self._env._current_gate_idx

    @property
    def dt(self) -> float:
        return 0.02

    def close(self) -> None:
        self._env.close()
