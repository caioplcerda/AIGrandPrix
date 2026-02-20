"""DCL platform adapter — stub with clear TODO markers.

When DCL releases their interface specs, fill in the 8 methods below.
This is one of only 2 files that need updating when specs arrive.
"""

from __future__ import annotations

import numpy as np

from aigrandprix.control.drone_controller import DroneState
from aigrandprix.planning.path_planner import Gate
from aigrandprix.simulation.sim_interface import SimInterface


class DCLAdapter(SimInterface):
    """Adapter between DCL competition platform and our SimInterface.

    TODO: When DCL releases their API specification, implement each method
    to bridge their data format to ours. See type_converters.py for the
    conversion helpers.
    """

    def __init__(self, dcl_handle: object | None = None) -> None:
        """Initialize with DCL platform handle.

        Args:
            dcl_handle: The DCL platform API object (type TBD).
        """
        self._dcl = dcl_handle
        # TODO: Store any DCL-specific state needed

    def reset(self, seed: int | None = None) -> dict:
        """Reset the DCL race environment.

        TODO: Call DCL's reset/restart API.
        Returns info dict with race metadata.
        """
        raise NotImplementedError("DCL reset() — fill when specs arrive")

    def step(self, action: np.ndarray) -> tuple[bool, bool, dict]:
        """Send action to DCL and advance one timestep.

        Args:
            action: [thrust, roll_rate, pitch_rate, yaw_rate]

        TODO: Convert action to DCL format, call their step API,
        return (terminated, truncated, info).
        """
        raise NotImplementedError("DCL step() — fill when specs arrive")

    def get_state(self) -> DroneState:
        """Get drone state from DCL telemetry.

        TODO: Read DCL's state output and convert to DroneState.
        """
        raise NotImplementedError("DCL get_state() — fill when specs arrive")

    def get_image(self) -> np.ndarray:
        """Get camera image from DCL.

        TODO: Read DCL's camera feed, convert to BGR (H, W, 3).
        """
        raise NotImplementedError("DCL get_image() — fill when specs arrive")

    def get_gate_poses(self) -> list[Gate]:
        """Get gate positions from DCL race config.

        TODO: Read DCL's gate/course data and convert to Gate objects.
        """
        raise NotImplementedError("DCL get_gate_poses() — fill when specs arrive")

    def get_next_gate_index(self) -> int:
        """Get index of next gate to pass.

        TODO: Read DCL's gate progress tracking.
        """
        raise NotImplementedError("DCL get_next_gate_index() — fill when specs arrive")

    @property
    def dt(self) -> float:
        """Simulation timestep.

        TODO: Read from DCL config or hardcode to match their physics rate.
        """
        return 0.02  # Placeholder — update when specs arrive

    def close(self) -> None:
        """Clean up DCL resources.

        TODO: Call DCL's cleanup/disconnect API.
        """
        pass  # Placeholder
