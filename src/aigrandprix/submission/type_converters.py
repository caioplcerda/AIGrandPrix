"""Type conversion between DCL formats and internal types.

This is one of only 2 files that need updating when DCL specs arrive.
All format-specific conversion logic is centralized here.
"""

from __future__ import annotations

import numpy as np

from aigrandprix.control.drone_controller import ControlCommand, DroneState
from aigrandprix.planning.path_planner import Gate


def observation_to_internal(
    obs: dict,
) -> tuple[np.ndarray, DroneState, float | None, int | None]:
    """Convert observation dict to internal types.

    Args:
        obs: Observation from DCL platform or internal sim.
            Expected keys: image, position, velocity, orientation,
            angular_velocity, elapsed_time, current_gate_index

    Returns:
        (image, DroneState, elapsed_time_or_None, gate_index_or_None)
    """
    # Image (default: blank if missing)
    image = obs.get("image", np.zeros((480, 640, 3), dtype=np.uint8))
    if not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.uint8)

    # Drone state
    position = np.array(obs.get("position", [0.0, 0.0, 1.0]), dtype=float)
    velocity = np.array(obs.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    orientation = np.array(obs.get("orientation", [1.0, 0.0, 0.0, 0.0]), dtype=float)
    angular_velocity = np.array(obs.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=float)

    state = DroneState(
        position=position,
        velocity=velocity,
        orientation=orientation,
        angular_velocity=angular_velocity,
    )

    elapsed = obs.get("elapsed_time")
    gate_idx = obs.get("current_gate_index")

    return image, state, elapsed, gate_idx


def command_to_array(cmd: ControlCommand) -> np.ndarray:
    """Convert ControlCommand to action array.

    Returns:
        np.ndarray of shape (4,): [thrust, roll_rate, pitch_rate, yaw_rate]
    """
    return np.array(
        [cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate],
        dtype=np.float32,
    )


def gates_from_race_info(race_info: dict) -> list[Gate]:
    """Extract gate list from race info dict.

    Supports multiple formats for flexibility:
    - {"gates": [{"position": [x,y,z], "normal": [nx,ny,nz]}, ...]}
    - {"gates": [[x,y,z], ...]}  (positions only, default normal)
    - {"gate_positions": [[x,y,z], ...]}  (alternative key)

    Args:
        race_info: Race configuration dict

    Returns:
        List of Gate objects
    """
    gates_data = race_info.get("gates", race_info.get("gate_positions", []))

    gates = []
    for i, g in enumerate(gates_data):
        if isinstance(g, dict):
            pos = np.array(g["position"], dtype=float)
            normal = np.array(g.get("normal", [1.0, 0.0, 0.0]), dtype=float)
            width = g.get("width", 1.0)
            height = g.get("height", 1.0)
        else:
            pos = np.array(g, dtype=float)
            normal = np.array([1.0, 0.0, 0.0])
            width = 1.0
            height = 1.0

        # Normalize normal vector
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-6:
            normal = normal / norm_len

        gates.append(Gate(
            position=pos,
            normal=normal,
            gate_id=i,
            width=width,
            height=height,
        ))

    return gates
