"""Path planning through racing gates.

Plans optimal trajectories through gate sequences using:
- Minimum jerk trajectory generation for smooth flight
- Minimum snap trajectory for race-optimal flight
- Time-optimal trajectory planning for speed
- Waypoint interpolation with velocity constraints
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


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
            self._trajectory_method = config.get("trajectory_method", "min_snap")
            self._racing_line_config = config.get("racing_line", {})
            self._time_alloc_config = config.get("time_allocation", {})
            self._min_snap_config = config.get("min_snap", {})
        else:
            self._approach_distance = 1.5
            self._exit_distance = 1.0
            self._trajectory_dt = 0.02
            self._trajectory_method = "cubic_spline"
            self._racing_line_config = {}
            self._time_alloc_config = {}
            self._min_snap_config = {}

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

        # Racing line optimization (if enabled)
        planning_gates = gates
        if self._racing_line_config.get("enabled", False):
            from aigrandprix.planning.race_strategy import RacingLineOptimizer

            optimizer = RacingLineOptimizer(
                gate_margin=self._racing_line_config.get("gate_margin", 0.15),
                curvature_weight=self._racing_line_config.get("curvature_weight", 2.0),
            )
            planning_gates = optimizer.optimize_gates(gates, start_pos)

        waypoints_arr = self._build_waypoints(planning_gates, start_pos)

        # Dispatch to selected trajectory method
        method = self._trajectory_method
        if method == "min_snap":
            return self._generate_min_snap_trajectory(waypoints_arr, start_vel)
        elif method == "min_jerk":
            return self._generate_min_jerk_trajectory(waypoints_arr, start_vel)
        else:
            return self._generate_cubic_spline_trajectory(waypoints_arr, start_vel)

    def _build_waypoints(
        self,
        gates: list[Gate],
        start_pos: np.ndarray,
    ) -> np.ndarray:
        """Build waypoint array from start position and gate sequence.

        Approach/exit waypoints inherit the gate's z but are clamped
        to a minimum altitude to prevent ground crashes in simplified physics.
        """
        min_z = 0.8  # minimum safe altitude
        waypoints = [start_pos]
        for gate in gates:
            approach = gate.position - gate.normal * self._approach_distance
            approach[2] = max(approach[2], min_z)
            waypoints.append(approach)
            gate_pos = gate.position.copy()
            gate_pos[2] = max(gate_pos[2], min_z)
            waypoints.append(gate_pos)
            exit_pt = gate.position + gate.normal * self._exit_distance
            exit_pt[2] = max(exit_pt[2], min_z)
            waypoints.append(exit_pt)
        return np.array(waypoints)

    def _estimate_segment_times(self, waypoints: np.ndarray) -> np.ndarray:
        """Estimate segment durations from distances and max speed."""
        diffs = np.diff(waypoints, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        # Use a fraction of max speed as average segment speed
        avg_speed = max(self.max_speed * 0.6, 1.0)
        times = distances / avg_speed
        # Enforce minimum segment time
        times = np.maximum(times, 0.1)
        return times

    def _generate_cubic_spline_trajectory(
        self,
        waypoints: np.ndarray,
        start_vel: np.ndarray,
        dt: float = 0.02,
    ) -> list[TrajectoryPoint]:
        """Generate a smooth trajectory through waypoints using cubic splines."""
        diffs = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_dist = cumulative_dist[-1]

        if total_dist < 0.01:
            return []

        avg_speed = min(self.max_speed * 0.7, total_dist / 2.0)
        total_time = total_dist / max(avg_speed, 0.1)
        time_params = cumulative_dist / total_dist * total_time

        splines = []
        for axis in range(3):
            cs = CubicSpline(
                time_params,
                waypoints[:, axis],
                bc_type=((1, start_vel[axis]), (1, 0.0)),
            )
            splines.append(cs)

        trajectory = []
        t = 0.0
        while t <= total_time:
            pos = np.array([s(t) for s in splines])
            vel = np.array([s(t, 1) for s in splines])
            acc = np.array([s(t, 2) for s in splines])

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

    def _generate_min_jerk_trajectory(
        self,
        waypoints: np.ndarray,
        start_vel: np.ndarray,
    ) -> list[TrajectoryPoint]:
        """Generate trajectory using minimum-jerk polynomial solver."""
        from aigrandprix.planning.trajectory_opt import (
            MinJerkSolver,
            TimeAllocator,
            sample_trajectory,
        )

        segment_times = self._estimate_segment_times(waypoints)

        # Time allocation optimization
        if self._time_alloc_config.get("enabled", False):
            solver = MinJerkSolver()
            allocator = TimeAllocator(
                min_segment_time=self._time_alloc_config.get("min_segment_time", 0.1),
                max_iterations=self._time_alloc_config.get("max_iterations", 50),
                tolerance=self._time_alloc_config.get("tolerance", 0.01),
            )
            segment_times = allocator.optimize(
                waypoints, segment_times,
                max_velocity=self.max_speed,
                max_acceleration=self.max_accel,
                solver=solver,
                start_vel=start_vel,
            )

        try:
            solver = MinJerkSolver()
            segments = solver.solve_trajectory(
                waypoints, segment_times, start_vel=start_vel,
            )
            return sample_trajectory(segments, self._trajectory_dt, self.max_speed)
        except np.linalg.LinAlgError:
            logger.warning("MinJerk solver failed, falling back to cubic spline")
            return self._generate_cubic_spline_trajectory(waypoints, start_vel)

    def _generate_min_snap_trajectory(
        self,
        waypoints: np.ndarray,
        start_vel: np.ndarray,
    ) -> list[TrajectoryPoint]:
        """Generate trajectory using minimum-snap polynomial solver."""
        from aigrandprix.planning.trajectory_opt import (
            MinJerkSolver,
            MinSnapSolver,
            TimeAllocator,
            sample_trajectory,
        )

        segment_times = self._estimate_segment_times(waypoints)

        # Time allocation optimization
        if self._time_alloc_config.get("enabled", False):
            solver_for_alloc: MinJerkSolver | MinSnapSolver = MinSnapSolver(
                regularization=self._min_snap_config.get("regularization", 1e-6),
            )
            allocator = TimeAllocator(
                min_segment_time=self._time_alloc_config.get("min_segment_time", 0.1),
                max_iterations=self._time_alloc_config.get("max_iterations", 50),
                tolerance=self._time_alloc_config.get("tolerance", 0.01),
            )
            try:
                segment_times = allocator.optimize(
                    waypoints, segment_times,
                    max_velocity=self.max_speed,
                    max_acceleration=self.max_accel,
                    solver=solver_for_alloc,
                    start_vel=start_vel,
                )
            except np.linalg.LinAlgError:
                logger.warning("Time allocation with min-snap failed, using initial times")

        try:
            solver = MinSnapSolver(
                regularization=self._min_snap_config.get("regularization", 1e-6),
            )
            segments = solver.solve_trajectory(
                waypoints, segment_times, start_vel=start_vel,
            )
            return sample_trajectory(segments, self._trajectory_dt, self.max_speed)
        except np.linalg.LinAlgError:
            logger.warning("MinSnap solver failed, falling back to min-jerk")
            return self._generate_min_jerk_trajectory(waypoints, start_vel)

    def replan_from_current(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        remaining_gates: list[Gate],
    ) -> list[TrajectoryPoint]:
        """Replan trajectory from current state to remaining gates.

        Uses cubic spline for speed (real-time replanning).
        Reduces approach/exit distances since we're already en route.
        """
        # Save and override for fast replanning
        saved_method = self._trajectory_method
        saved_approach = self._approach_distance
        saved_exit = self._exit_distance
        if self._trajectory_method != "cubic_spline":
            self._trajectory_method = "cubic_spline"
        # Shorter approach for replanning — we're already on course
        self._approach_distance = 0.5
        self._exit_distance = 0.5

        result = self.plan_through_gates(remaining_gates, current_pos, current_vel)

        self._trajectory_method = saved_method
        self._approach_distance = saved_approach
        self._exit_distance = saved_exit
        return result
