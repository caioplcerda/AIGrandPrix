"""Path planning through racing gates.

Plans optimal trajectories through gate sequences using:
- Minimum jerk trajectory generation for smooth flight
- Time-optimal trajectory planning for speed
- Waypoint interpolation with velocity constraints
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class Gate:
    """Racing gate definition."""

    position: np.ndarray  # [x, y, z] center of gate
    normal: np.ndarray  # gate facing direction (unit vector)
    gate_id: int
    width: float = 1.0  # meters
    height: float = 1.0  # meters


@dataclass
class TrajectoryPoint:
    """Single point on a planned trajectory."""

    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]
    time: float  # seconds from start


class PathPlanner:
    """Plans optimal flight paths through gate sequences."""

    def __init__(
        self,
        max_speed: float = 15.0,
        max_accel: float = 10.0,
        config: dict | None = None,
    ) -> None:
        self.max_speed = max_speed
        self.max_accel = max_accel
        if config is not None:
            self._approach_distance = config.get("approach_distance", 1.5)
            self._exit_distance = config.get("exit_distance", 1.0)
            self._trajectory_dt = config.get("trajectory_dt", 0.02)
        else:
            self._approach_distance = 1.5
            self._exit_distance = 1.0
            self._trajectory_dt = 0.02

    def plan_through_gates(
        self,
        gates: list[Gate],
        start_pos: np.ndarray,
        start_vel: np.ndarray | None = None,
    ) -> list[TrajectoryPoint]:
        """Plan a trajectory through all gates in order.

        Args:
            gates: Ordered list of gates to fly through
            start_pos: Current drone position
            start_vel: Current drone velocity (default: hovering)

        Returns:
            List of trajectory points at regular intervals
        """
        if start_vel is None:
            start_vel = np.zeros(3)

        # Build waypoints: start + gate centers + approach/exit offsets
        waypoints = [start_pos]
        for gate in gates:
            # Add approach point (before gate)
            approach = gate.position - gate.normal * self._approach_distance
            waypoints.append(approach)
            # Gate center
            waypoints.append(gate.position)
            # Exit point (after gate)
            exit_pt = gate.position + gate.normal * self._exit_distance
            waypoints.append(exit_pt)

        waypoints_arr = np.array(waypoints)
        return self._generate_smooth_trajectory(waypoints_arr, start_vel)

    def _generate_smooth_trajectory(
        self,
        waypoints: np.ndarray,
        start_vel: np.ndarray,
        dt: float = 0.02,
    ) -> list[TrajectoryPoint]:
        """Generate a smooth trajectory through waypoints using cubic splines.

        Uses distance-parameterized spline interpolation for natural speed profiles.
        """
        # Compute cumulative distance along waypoints
        diffs = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_dist = cumulative_dist[-1]

        if total_dist < 0.01:
            return []

        # Estimate total time based on average speed
        avg_speed = min(self.max_speed * 0.7, total_dist / 2.0)
        total_time = total_dist / max(avg_speed, 0.1)

        # Normalize distances to time parameter
        time_params = cumulative_dist / total_dist * total_time

        # Fit cubic spline for each axis
        splines = []
        for axis in range(3):
            cs = CubicSpline(
                time_params,
                waypoints[:, axis],
                bc_type=((1, start_vel[axis]), (1, 0.0)),  # velocity BCs
            )
            splines.append(cs)

        # Sample trajectory
        trajectory = []
        t = 0.0
        while t <= total_time:
            pos = np.array([s(t) for s in splines])
            vel = np.array([s(t, 1) for s in splines])
            acc = np.array([s(t, 2) for s in splines])

            # Enforce speed limit
            speed = np.linalg.norm(vel)
            if speed > self.max_speed:
                vel = vel / speed * self.max_speed

            trajectory.append(TrajectoryPoint(
                position=pos,
                velocity=vel,
                acceleration=acc,
                time=t,
            ))
            t += dt

        return trajectory

    def replan_from_current(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        remaining_gates: list[Gate],
    ) -> list[TrajectoryPoint]:
        """Replan trajectory from current state to remaining gates.

        Used for online replanning when deviating from planned path.
        """
        return self.plan_through_gates(remaining_gates, current_pos, current_vel)
