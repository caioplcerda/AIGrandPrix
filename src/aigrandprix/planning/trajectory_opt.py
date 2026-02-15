"""Polynomial trajectory optimization for drone racing.

Provides minimum-jerk (5th order) and minimum-snap (7th order) trajectory
solvers, plus time-allocation optimization for faster gate passage.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy.optimize import minimize

from aigrandprix.planning.path_planner import TrajectoryPoint


@dataclass
class PolynomialSegment:
    """Single polynomial segment in ascending coefficient order.

    coeffs[i] is the coefficient of t^i: p(t) = a0 + a1*t + a2*t^2 + ...
    """

    coeffs: np.ndarray  # [a0, a1, ..., an]
    duration: float

    def evaluate(self, t: float, derivative: int = 0) -> float:
        """Evaluate the polynomial (or its derivative) at time t."""
        c = self.coeffs.copy()
        # Differentiate `derivative` times
        for _ in range(derivative):
            if len(c) <= 1:
                return 0.0
            c = np.array([c[i] * i for i in range(1, len(c))])
        # Evaluate using Horner's method (polyval wants descending order)
        return float(np.polyval(c[::-1], t))

    def max_abs_value(self, derivative: int, n_samples: int = 100) -> float:
        """Sample the segment and return peak absolute magnitude of a derivative."""
        ts = np.linspace(0.0, self.duration, n_samples)
        vals = np.array([self.evaluate(t, derivative) for t in ts])
        return float(np.max(np.abs(vals)))


class MinJerkSolver:
    """Closed-form 5th-order polynomial minimising jerk per segment per axis."""

    def solve_segment(
        self,
        x0: float, v0: float, a0: float,
        xf: float, vf: float, af: float,
        T: float,
    ) -> PolynomialSegment:
        """Solve for a single 5th-order segment.

        Boundary conditions at t=0: position=x0, velocity=v0, acceleration=a0
        Boundary conditions at t=T: position=xf, velocity=vf, acceleration=af
        """
        # First 3 coefficients from initial conditions
        c0 = x0
        c1 = v0
        c2 = a0 / 2.0

        # Solve 3x3 system for [c3, c4, c5]
        T2, T3, T4, T5 = T**2, T**3, T**4, T**5
        A = np.array([
            [T3, T4, T5],
            [3.0 * T2, 4.0 * T3, 5.0 * T4],
            [6.0 * T, 12.0 * T2, 20.0 * T3],
        ])
        b = np.array([
            xf - c0 - c1 * T - c2 * T2,
            vf - c1 - 2.0 * c2 * T,
            af - 2.0 * c2,
        ])
        x = np.linalg.solve(A, b)

        coeffs = np.array([c0, c1, c2, x[0], x[1], x[2]])
        return PolynomialSegment(coeffs=coeffs, duration=T)

    def solve_trajectory(
        self,
        waypoints: np.ndarray,
        segment_times: np.ndarray,
        start_vel: np.ndarray | None = None,
        start_accel: np.ndarray | None = None,
        end_vel: np.ndarray | None = None,
        end_accel: np.ndarray | None = None,
    ) -> list[list[PolynomialSegment]]:
        """Solve a multi-segment trajectory for all 3 axes.

        Returns segments[axis][segment_idx].
        """
        n_waypoints = len(waypoints)
        n_segments = n_waypoints - 1
        n_axes = waypoints.shape[1]

        if start_vel is None:
            start_vel = np.zeros(n_axes)
        if start_accel is None:
            start_accel = np.zeros(n_axes)
        if end_vel is None:
            end_vel = np.zeros(n_axes)
        if end_accel is None:
            end_accel = np.zeros(n_axes)

        # Estimate interior velocities: central difference
        interior_vels = np.zeros((n_waypoints, n_axes))
        interior_vels[0] = start_vel
        interior_vels[-1] = end_vel
        for i in range(1, n_waypoints - 1):
            dt_sum = segment_times[i - 1] + segment_times[i]
            if dt_sum > 1e-12:
                interior_vels[i] = (waypoints[i + 1] - waypoints[i - 1]) / dt_sum

        all_segments: list[list[PolynomialSegment]] = []

        for axis in range(n_axes):
            axis_segments: list[PolynomialSegment] = []
            prev_accel = start_accel[axis]

            for seg in range(n_segments):
                x0 = waypoints[seg, axis]
                xf = waypoints[seg + 1, axis]
                v0 = interior_vels[seg, axis]
                vf = interior_vels[seg + 1, axis]
                a0 = prev_accel
                # For the last segment, use end acceleration
                if seg == n_segments - 1:
                    af = end_accel[axis]
                else:
                    # Will be computed after solving; use 0 as target
                    af = 0.0

                T = segment_times[seg]
                segment = self.solve_segment(x0, v0, a0, xf, vf, af, T)
                axis_segments.append(segment)

                # Propagate end acceleration for C2 continuity
                prev_accel = segment.evaluate(T, derivative=2)

            all_segments.append(axis_segments)

        return all_segments


class MinSnapSolver:
    """7th-order polynomial minimising snap (4th derivative), solved via QP/KKT."""

    def __init__(self, regularization: float = 1e-6) -> None:
        self.regularization = regularization

    def _build_segment_cost_matrix(self, T: float) -> np.ndarray:
        """Build the 8x8 Hessian for snap cost on one segment of duration T."""
        H = np.zeros((8, 8))
        for j in range(4, 8):
            for k in range(4, 8):
                num = factorial(j) * factorial(k)
                den_j = factorial(j - 4)
                den_k = factorial(k - 4)
                power = j + k - 7
                H[j, k] = (num / (den_j * den_k)) * (T ** power) / power
        return H

    def _derivative_constraint_row(
        self,
        seg_idx: int,
        n_segments: int,
        t: float,
        derivative: int,
    ) -> np.ndarray:
        """Build a constraint row for a derivative value at time t in segment seg_idx."""
        row = np.zeros(8 * n_segments)
        base = 8 * seg_idx
        for j in range(derivative, 8):
            coeff = 1.0
            for d in range(derivative):
                coeff *= (j - d)
            row[base + j] = coeff * (t ** (j - derivative))
        return row

    def solve_axis(
        self,
        waypoints_1d: np.ndarray,
        segment_times: np.ndarray,
        v_start: float = 0.0,
        a_start: float = 0.0,
        v_end: float = 0.0,
        a_end: float = 0.0,
    ) -> list[PolynomialSegment]:
        """Solve min-snap for one axis. Returns list of PolynomialSegment."""
        N = len(waypoints_1d)  # number of waypoints
        n_seg = N - 1
        n_vars = 8 * n_seg

        # Build block-diagonal cost matrix
        Q = np.zeros((n_vars, n_vars))
        for i in range(n_seg):
            T = segment_times[i]
            H = self._build_segment_cost_matrix(T)
            Q[8 * i:8 * (i + 1), 8 * i:8 * (i + 1)] = H
        # Regularization
        Q += self.regularization * np.eye(n_vars)

        # Build equality constraints: A_eq @ x = b_eq
        rows = []
        b_vals = []

        # 1) Position at start of each segment (N waypoints -> N constraints)
        # Start of first segment
        row = self._derivative_constraint_row(0, n_seg, 0.0, 0)
        rows.append(row)
        b_vals.append(waypoints_1d[0])

        # End of each segment = next waypoint
        for i in range(n_seg):
            T = segment_times[i]
            row = self._derivative_constraint_row(i, n_seg, T, 0)
            rows.append(row)
            b_vals.append(waypoints_1d[i + 1])

        # 2) Continuity at interior joints: position, velocity, acceleration, jerk
        for i in range(n_seg - 1):
            T = segment_times[i]
            for deriv in range(4):  # pos, vel, acc, jerk
                # End of segment i == start of segment i+1
                row_end = self._derivative_constraint_row(i, n_seg, T, deriv)
                row_start = self._derivative_constraint_row(i + 1, n_seg, 0.0, deriv)
                row = row_end - row_start
                rows.append(row)
                b_vals.append(0.0)

        # 3) Boundary conditions: vel and acc at start and end
        # Start velocity
        row = self._derivative_constraint_row(0, n_seg, 0.0, 1)
        rows.append(row)
        b_vals.append(v_start)

        # Start acceleration
        row = self._derivative_constraint_row(0, n_seg, 0.0, 2)
        rows.append(row)
        b_vals.append(a_start)

        # End velocity
        row = self._derivative_constraint_row(n_seg - 1, n_seg, segment_times[-1], 1)
        rows.append(row)
        b_vals.append(v_end)

        # End acceleration
        row = self._derivative_constraint_row(n_seg - 1, n_seg, segment_times[-1], 2)
        rows.append(row)
        b_vals.append(a_end)

        A_eq = np.array(rows)
        b_eq = np.array(b_vals)

        n_constraints = A_eq.shape[0]

        # Solve KKT system: [[Q, A^T], [A, 0]] [x; lambda] = [0; b]
        KKT = np.zeros((n_vars + n_constraints, n_vars + n_constraints))
        KKT[:n_vars, :n_vars] = Q
        KKT[:n_vars, n_vars:] = A_eq.T
        KKT[n_vars:, :n_vars] = A_eq

        rhs = np.zeros(n_vars + n_constraints)
        rhs[n_vars:] = b_eq

        solution = np.linalg.solve(KKT, rhs)
        x = solution[:n_vars]

        # Extract segments
        segments = []
        for i in range(n_seg):
            coeffs = x[8 * i:8 * (i + 1)]
            segments.append(PolynomialSegment(coeffs=coeffs, duration=segment_times[i]))

        return segments

    def solve_trajectory(
        self,
        waypoints: np.ndarray,
        segment_times: np.ndarray,
        start_vel: np.ndarray | None = None,
        start_accel: np.ndarray | None = None,
        end_vel: np.ndarray | None = None,
        end_accel: np.ndarray | None = None,
    ) -> list[list[PolynomialSegment]]:
        """Solve min-snap for all axes. Returns segments[axis][segment_idx]."""
        n_axes = waypoints.shape[1]
        if start_vel is None:
            start_vel = np.zeros(n_axes)
        if start_accel is None:
            start_accel = np.zeros(n_axes)
        if end_vel is None:
            end_vel = np.zeros(n_axes)
        if end_accel is None:
            end_accel = np.zeros(n_axes)

        all_segments: list[list[PolynomialSegment]] = []
        for axis in range(n_axes):
            segs = self.solve_axis(
                waypoints[:, axis],
                segment_times,
                v_start=start_vel[axis],
                a_start=start_accel[axis],
                v_end=end_vel[axis],
                a_end=end_accel[axis],
            )
            all_segments.append(segs)

        return all_segments


def sample_trajectory(
    segments: list[list[PolynomialSegment]],
    dt: float,
    max_speed: float = float("inf"),
) -> list[TrajectoryPoint]:
    """Sample a multi-axis polynomial trajectory at regular intervals.

    Args:
        segments: segments[axis][segment_idx]
        dt: sampling interval in seconds
        max_speed: clamp velocity magnitude to this value
    """
    n_axes = len(segments)
    n_segments = len(segments[0])

    trajectory: list[TrajectoryPoint] = []
    global_time = 0.0

    for seg_idx in range(n_segments):
        T = segments[0][seg_idx].duration
        local_t = 0.0

        while local_t <= T + 1e-9:
            t_eval = min(local_t, T)
            pos = np.array([segments[ax][seg_idx].evaluate(t_eval, 0) for ax in range(n_axes)])
            vel = np.array([segments[ax][seg_idx].evaluate(t_eval, 1) for ax in range(n_axes)])
            acc = np.array([segments[ax][seg_idx].evaluate(t_eval, 2) for ax in range(n_axes)])

            # Clamp speed
            speed = np.linalg.norm(vel)
            if speed > max_speed:
                vel = vel / speed * max_speed

            trajectory.append(TrajectoryPoint(
                position=pos,
                velocity=vel,
                acceleration=acc,
                time=global_time,
            ))

            local_t += dt
            global_time += dt

        # Avoid double-counting the endpoint / start of next segment
        # Rewind so next segment starts from where we left off
        overshoot = local_t - dt - T
        if overshoot > 1e-9:
            global_time -= overshoot

    return trajectory


class TimeAllocator:
    """Optimise segment durations to minimise total time under dynamic constraints."""

    def __init__(
        self,
        min_segment_time: float = 0.1,
        max_iterations: int = 50,
        tolerance: float = 0.01,
    ) -> None:
        self.min_segment_time = min_segment_time
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(
        self,
        waypoints: np.ndarray,
        initial_times: np.ndarray,
        max_velocity: float,
        max_acceleration: float,
        solver: MinJerkSolver | MinSnapSolver,
        start_vel: np.ndarray | None = None,
        end_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Optimise segment times to minimise total time respecting dynamic limits."""
        n_seg = len(initial_times)
        if start_vel is None:
            start_vel = np.zeros(waypoints.shape[1])
        if end_vel is None:
            end_vel = np.zeros(waypoints.shape[1])

        def _check_limits(times: np.ndarray) -> tuple[float, float]:
            """Return (max_speed, max_accel) along the trajectory."""
            segments = solver.solve_trajectory(
                waypoints, times,
                start_vel=start_vel, end_vel=end_vel,
            )
            max_spd = 0.0
            max_acc = 0.0
            n_axes = len(segments)
            n_segments = len(segments[0])
            for seg_idx in range(n_segments):
                T = segments[0][seg_idx].duration
                for t in np.linspace(0, T, 100):
                    vel = np.array([segments[ax][seg_idx].evaluate(t, 1) for ax in range(n_axes)])
                    acc = np.array([segments[ax][seg_idx].evaluate(t, 2) for ax in range(n_axes)])
                    spd = float(np.linalg.norm(vel))
                    ac = float(np.linalg.norm(acc))
                    if spd > max_spd:
                        max_spd = spd
                    if ac > max_acc:
                        max_acc = ac
            return max_spd, max_acc

        def objective(times: np.ndarray) -> float:
            return float(np.sum(times))

        def vel_constraint(times: np.ndarray) -> float:
            max_spd, _ = _check_limits(times)
            return max_velocity - max_spd  # must be >= 0

        def acc_constraint(times: np.ndarray) -> float:
            _, max_acc = _check_limits(times)
            return max_acceleration - max_acc  # must be >= 0

        bounds = [(self.min_segment_time, None)] * n_seg
        constraints = [
            {"type": "ineq", "fun": vel_constraint},
            {"type": "ineq", "fun": acc_constraint},
        ]

        result = minimize(
            objective,
            initial_times,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance},
        )

        return np.array(result.x)
