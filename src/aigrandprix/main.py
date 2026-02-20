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

        # MPC controller (created when controller_type == "mpc")
        self._mpc = None
        self._controller_type = "pid"
        if control_cfg:
            self._controller_type = control_cfg.get("controller_type", "pid")
            if self._controller_type == "mpc":
                from aigrandprix.control.mpc_controller import MPCConfig, MPCController
                mpc_cfg = MPCConfig.from_config(control_cfg.get("mpc", {}))
                self._mpc = MPCController(
                    config=mpc_cfg,
                    max_speed=self.config.max_speed,
                    max_rate=control_cfg.get("max_rate", 8.0),
                    max_thrust=control_cfg.get("max_thrust", 1.0),
                )

        # Optional Phase 1 perception modules (None when not configured)
        self.depth_estimator = None
        self.state_estimator = None

        if full_config:
            p_cfg = full_config.get("perception", {})
            # Depth estimator: create when camera or gate dimensions are specified
            if p_cfg.get("camera") or p_cfg.get("gate_real_width"):
                from aigrandprix.perception.depth_estimation import DepthEstimator
                self.depth_estimator = DepthEstimator(
                    gate_width=p_cfg.get("gate_real_width", 1.0),
                    gate_height=p_cfg.get("gate_real_height", 1.0),
                    config=p_cfg,
                )
            # State estimator: create when explicitly enabled
            se_cfg = p_cfg.get("state_estimator", {})
            if se_cfg.get("enabled", False):
                from aigrandprix.perception.state_estimator import StateEstimator
                self.state_estimator = StateEstimator(config=se_cfg)

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
        current_gate_index: int | None = None,
    ) -> ControlCommand:
        """Compute the next control action given current sensor data.

        This is the main function called each timestep during the race.

        Args:
            image: Current camera image (BGR)
            state: Current drone state
            elapsed_time: Time since race start
            known_gates: Pre-loaded gate positions (if available)
            current_gate_index: External gate progress (from DCL platform)

        Returns:
            Control command for the drone
        """
        # Use state estimator if available
        active_state = state
        if self.state_estimator is not None:
            self.state_estimator.predict(
                accel_meas=np.array([0.0, 0.0, 9.81]),  # placeholder IMU
                gyro_meas=state.angular_velocity,
                dt=self.config.control_dt,
            )
            active_state = self.state_estimator.get_state()

        # External gate sync (from DCL platform)
        if current_gate_index is not None and current_gate_index > self._gates_passed:
            for _ in range(current_gate_index - self._gates_passed):
                self.on_gate_passed(self._gates_passed)

        # Detect gates with tracking if available
        if self.depth_estimator is not None and hasattr(self.detector, "detect_and_track"):
            detections, tracks = self.detector.detect_and_track(
                image, elapsed_time, self.depth_estimator,
            )
            # Refine detection distances via PnP
            for det in detections:
                pose = self.depth_estimator.estimate_pose_from_detection(det)
                if pose is not None:
                    det.distance = pose.distance
        else:
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
                current_pos=active_state.position,
                current_vel=active_state.velocity,
                remaining_gates=remaining_gates,
            )
            self._trajectory_idx = 0
            self._last_replan_time = elapsed_time

        # Follow trajectory
        if self._trajectory and self._trajectory_idx < len(self._trajectory):
            if self._controller_type == "mpc" and self._mpc is not None:
                # MPC: extract lookahead window
                horizon = self._mpc.config.horizon
                lookahead = self._trajectory[self._trajectory_idx:self._trajectory_idx + horizon]
                # Pad with last point if near end
                while len(lookahead) < horizon and lookahead:
                    lookahead.append(lookahead[-1])
                command = self._mpc.compute_control(active_state, lookahead)
                # Add yaw rate from PID controller
                if self._gates_passed < len(known_gates or []):
                    next_gate = known_gates[self._gates_passed]
                    command.yaw_rate = self.controller.compute_yaw_rate(
                        active_state, next_gate.position,
                    )
                self._trajectory_idx += 1
            else:
                # PID: existing path (now with corrected physics mapping)
                target = self._trajectory[self._trajectory_idx]
                command = self.controller.track_trajectory_point(
                    active_state,
                    target_pos=target.position,
                    target_vel=target.velocity,
                    target_accel=target.acceleration,
                )
                # Compute yaw to face next gate
                if self._gates_passed < len(known_gates or []):
                    next_gate = known_gates[self._gates_passed]
                    command.yaw_rate = self.controller.compute_yaw_rate(
                        active_state, next_gate.position,
                    )
                self._trajectory_idx += 1
        else:
            # Emergency: hold position
            command = self.controller.track_position(active_state, active_state.position)

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
        if self._mpc is not None:
            self._mpc.reset()
        if self.state_estimator is not None:
            self.state_estimator.reset(DroneState(
                position=np.zeros(3),
                velocity=np.zeros(3),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity=np.zeros(3),
            ))
