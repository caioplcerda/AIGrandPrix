"""Main entry point for the AI Grand Prix autonomous racing system.

This module ties together perception, planning, and control into
the main racing loop that will interface with the DCL platform.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from aigrandprix.control.drone_controller import ControlCommand, DroneController, DroneState
from aigrandprix.perception.gate_detector import GateDetector
from aigrandprix.planning.path_planner import Gate, PathPlanner


@dataclass
class RaceConfig:
    """Configuration for a race run."""

    max_speed: float = 15.0
    max_accel: float = 10.0
    detection_method: str = "hybrid"
    replan_interval: float = 0.5  # seconds between replanning
    control_dt: float = 0.01  # control loop timestep

    @classmethod
    def from_config(cls, config: dict) -> RaceConfig:
        race = config.get("race", {})
        perception = config.get("perception", {})
        return cls(
            max_speed=race.get("max_speed", 15.0),
            max_accel=race.get("max_accel", 10.0),
            detection_method=perception.get("method", "hybrid"),
            replan_interval=race.get("replan_interval", 0.5),
            control_dt=race.get("control_dt", 0.01),
        )


class RacingAgent:
    """Main autonomous racing agent.

    Coordinates perception, planning, and control modules to
    fly through gates as fast as possible.
    """

    def __init__(
        self,
        config: RaceConfig | None = None,
        full_config: dict | None = None,
    ) -> None:
        self.config = config or RaceConfig()

        perception_cfg = full_config.get("perception", {}) if full_config else None
        planning_cfg = full_config.get("planning", {}) if full_config else None
        control_cfg = full_config.get("control", {}) if full_config else None

        self.detector = GateDetector(
            method=self.config.detection_method,
            config=perception_cfg,
        )
        self.planner = PathPlanner(
            max_speed=self.config.max_speed,
            max_accel=self.config.max_accel,
            config=planning_cfg,
        )
        self.controller = DroneController(
            dt=self.config.control_dt,
            config=control_cfg,
        )

        self._trajectory = []
        self._trajectory_idx = 0
        self._gates_passed = 0
        self._last_replan_time = 0.0

    def compute_action(
        self,
        image: np.ndarray,
        state: DroneState,
        elapsed_time: float,
        known_gates: list[Gate] | None = None,
    ) -> ControlCommand:
        """Compute the next control action given current sensor data.

        This is the main function called each timestep during the race.

        Args:
            image: Current camera image (BGR)
            state: Current drone state
            elapsed_time: Time since race start
            known_gates: Pre-loaded gate positions (if available)

        Returns:
            Control command for the drone
        """
        # Detect gates from vision
        detections = self.detector.detect(image)

        # Decide if we need to replan
        should_replan = (
            not self._trajectory
            or elapsed_time - self._last_replan_time > self.config.replan_interval
            or self._trajectory_idx >= len(self._trajectory)
        )

        if should_replan and known_gates:
            remaining_gates = known_gates[self._gates_passed:]
            self._trajectory = self.planner.replan_from_current(
                current_pos=state.position,
                current_vel=state.velocity,
                remaining_gates=remaining_gates,
            )
            self._trajectory_idx = 0
            self._last_replan_time = elapsed_time

        # Follow trajectory using feedforward tracking for racing performance
        if self._trajectory and self._trajectory_idx < len(self._trajectory):
            target = self._trajectory[self._trajectory_idx]
            command = self.controller.track_trajectory_point(
                state,
                target_pos=target.position,
                target_vel=target.velocity,
                target_accel=target.acceleration,
            )
            # Compute yaw to face next gate
            if self._gates_passed < len(known_gates or []):
                next_gate = known_gates[self._gates_passed]
                command.yaw_rate = self.controller.compute_yaw_rate(state, next_gate.position)
            self._trajectory_idx += 1
        else:
            # Emergency: hold position
            command = self.controller.track_position(state, state.position)

        return command

    def on_gate_passed(self, gate_id: int) -> None:
        """Called when a gate has been passed."""
        self._gates_passed += 1

    def reset(self) -> None:
        """Reset agent for a new race."""
        self._trajectory = []
        self._trajectory_idx = 0
        self._gates_passed = 0
        self._last_replan_time = 0.0
        self.controller.reset()
