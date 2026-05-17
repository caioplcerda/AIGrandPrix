"""NED-aware path planner for DCL competition (VADR-TS-002).

Wraps PathPlanner with NED coordinate conventions:
  - X = North, Y = East, Z = Down
  - Altitude above ground = -Z (z ≤ 0 means at or below ground)
  - Minimum safe altitude: z ≤ -MIN_ALT_M (default -0.5m)
  - Gate inner opening: 1.5×1.5 m (VADR-TS-002 §3.7)

Outputs:
  - Waypoints in NED for SET_POSITION_TARGET_LOCAL_NED
  - Each waypoint includes (pos_ned, vel_ned, yaw) for the MAVLink command
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aigrandprix.planning.path_planner import PathPlanner, TrajectoryPoint

_MIN_ALTITUDE_M = 0.5     # drone must stay ≥0.5m above arming point
_MAX_Z_NED = -_MIN_ALTITUDE_M  # NED Z: must be ≤ this value (more negative = higher)


@dataclass
class GateNED:
    """Racing gate in NED world frame (VADR-TS-002 §3.7)."""

    position: np.ndarray   # center [north, east, down] in meters
    normal: np.ndarray     # unit vector in NED pointing from approach side toward gate
    gate_id: int
    inner_half: float = 0.75  # half of inner opening (1.5m / 2)

    def __post_init__(self) -> None:
        n = np.linalg.norm(self.normal)
        if n > 1e-6:
            self.normal = self.normal / n


@dataclass
class WaypointNED:
    """NED waypoint ready for SET_POSITION_TARGET_LOCAL_NED."""

    pos: np.ndarray     # [north, east, down] (m)
    vel: np.ndarray     # [vn, ve, vd] (m/s) — feedforward velocity
    yaw: float          # heading (rad), 0=north, π/2=east
    time: float         # seconds from start (informational)


def _heading_to_gate(from_pos: np.ndarray, gate_pos: np.ndarray) -> float:
    """Compute NED yaw angle pointing from from_pos toward gate_pos."""
    dn = gate_pos[0] - from_pos[0]  # north delta
    de = gate_pos[1] - from_pos[1]  # east delta
    return math.atan2(de, dn)       # NED yaw: 0=north, π/2=east


def _clamp_altitude(pt: np.ndarray) -> np.ndarray:
    """Clamp NED point to minimum altitude (z ≤ _MAX_Z_NED)."""
    out = pt.copy()
    out[2] = min(out[2], _MAX_Z_NED)
    return out


class PathPlannerNED:
    """Trajectory planner in NED frame for DCL competition.

    Converts GateNED gates to NED waypoints suitable for
    SET_POSITION_TARGET_LOCAL_NED commands.

    The underlying trajectory math is coordinate-agnostic (from PathPlanner),
    but altitude safety uses NED semantics (z ≤ -MIN_ALT).
    """

    def __init__(
        self,
        max_speed: float = 8.0,
        max_accel: float = 6.0,
        approach_distance: float = 2.0,
        exit_distance: float = 1.0,
        trajectory_method: str = "cubic_spline",
    ) -> None:
        self._max_speed = max_speed
        self._max_accel = max_accel
        self._approach_dist = approach_distance
        self._exit_dist = exit_distance
        self._planner = PathPlanner(
            max_speed=max_speed,
            max_accel=max_accel,
            config={
                "approach_distance": approach_distance,
                "exit_distance": exit_distance,
                "trajectory_dt": 0.05,
                "trajectory_method": trajectory_method,
            },
        )

    def plan(
        self,
        gates: list[GateNED],
        start_pos: np.ndarray,
        start_vel: np.ndarray | None = None,
    ) -> list[WaypointNED]:
        """Plan NED trajectory through all gates.

        Args:
            gates: ordered list of gates in NED
            start_pos: drone position in NED
            start_vel: current velocity in NED (default: hover)

        Returns:
            List of WaypointNED commands to send to MAVLink client.
        """
        if not gates:
            return []

        if start_vel is None:
            start_vel = np.zeros(3)

        # Build waypoints array in NED with altitude clamping
        wps = self._build_waypoints_ned(gates, start_pos)

        # Use underlying trajectory generator (coordinate-agnostic)
        traj = self._planner._generate_cubic_spline_trajectory(wps, start_vel)  # noqa: SLF001
        if not traj:
            return []

        return self._to_waypoints_ned(traj, gates, start_pos)

    def replan(
        self,
        remaining_gates: list[GateNED],
        current_pos: np.ndarray,
        current_vel: np.ndarray,
    ) -> list[WaypointNED]:
        """Fast replan from current position (for real-time use, cubic spline only)."""
        return self.plan(remaining_gates, current_pos, current_vel)

    def next_position_target(
        self,
        waypoints: list[WaypointNED],
        current_pos: np.ndarray,
        lookahead_m: float = 3.0,
    ) -> WaypointNED | None:
        """Select the next waypoint to command (lookahead-based).

        Finds the first waypoint ahead of current position by lookahead_m.
        Returns None when past the last waypoint.
        """
        if not waypoints:
            return None

        # Find nearest waypoint index
        dists = [float(np.linalg.norm(wp.pos - current_pos)) for wp in waypoints]
        nearest_idx = int(np.argmin(dists))

        # Look ahead from nearest
        for i in range(nearest_idx, len(waypoints)):
            d = float(np.linalg.norm(waypoints[i].pos - current_pos))
            if d >= lookahead_m:
                return waypoints[i]

        # Past all waypoints — return last
        return waypoints[-1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_waypoints_ned(
        self, gates: list[GateNED], start_pos: np.ndarray
    ) -> np.ndarray:
        pts = [_clamp_altitude(start_pos)]
        for g in gates:
            approach = _clamp_altitude(g.position - g.normal * self._approach_dist)
            pts.append(approach)
            pts.append(_clamp_altitude(g.position.copy()))
            exit_pt = _clamp_altitude(g.position + g.normal * self._exit_dist)
            pts.append(exit_pt)
        return np.array(pts)

    def _to_waypoints_ned(
        self,
        traj: list[TrajectoryPoint],
        gates: list[GateNED],
        start_pos: np.ndarray,
    ) -> list[WaypointNED]:
        result = []
        for i, tp in enumerate(traj):
            pos = _clamp_altitude(tp.position)
            vel = tp.velocity

            # Compute yaw toward next gate (track heading)
            if gates:
                target_gate_pos = gates[min(len(gates) - 1, i // max(1, len(traj) // len(gates)))].position
                yaw = _heading_to_gate(pos, target_gate_pos)
            else:
                yaw = _heading_to_gate(start_pos, pos) if i > 0 else 0.0

            result.append(WaypointNED(pos=pos, vel=vel, yaw=yaw, time=tp.time))
        return result
