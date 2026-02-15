"""Tests for polynomial trajectory optimization solvers."""

import time

import numpy as np
import pytest

from aigrandprix.planning.trajectory_opt import (
    MinJerkSolver,
    MinSnapSolver,
    PolynomialSegment,
    TimeAllocator,
    sample_trajectory,
)


# ---------------------------------------------------------------------------
# PolynomialSegment
# ---------------------------------------------------------------------------

class TestPolynomialSegment:
    def test_evaluate_constant(self):
        seg = PolynomialSegment(coeffs=np.array([5.0]), duration=1.0)
        assert seg.evaluate(0.0) == pytest.approx(5.0)
        assert seg.evaluate(0.5) == pytest.approx(5.0)

    def test_evaluate_linear(self):
        # p(t) = 2 + 3t
        seg = PolynomialSegment(coeffs=np.array([2.0, 3.0]), duration=2.0)
        assert seg.evaluate(0.0) == pytest.approx(2.0)
        assert seg.evaluate(1.0) == pytest.approx(5.0)
        assert seg.evaluate(2.0) == pytest.approx(8.0)

    def test_evaluate_quadratic(self):
        # p(t) = 1 + 2t + 3t^2
        seg = PolynomialSegment(coeffs=np.array([1.0, 2.0, 3.0]), duration=1.0)
        assert seg.evaluate(1.0) == pytest.approx(6.0)

    def test_first_derivative(self):
        # p(t) = 1 + 2t + 3t^2  =>  p'(t) = 2 + 6t
        seg = PolynomialSegment(coeffs=np.array([1.0, 2.0, 3.0]), duration=2.0)
        assert seg.evaluate(0.0, derivative=1) == pytest.approx(2.0)
        assert seg.evaluate(1.0, derivative=1) == pytest.approx(8.0)

    def test_second_derivative(self):
        # p(t) = 1 + 2t + 3t^2  =>  p''(t) = 6
        seg = PolynomialSegment(coeffs=np.array([1.0, 2.0, 3.0]), duration=2.0)
        assert seg.evaluate(0.5, derivative=2) == pytest.approx(6.0)

    def test_higher_derivative_than_order_returns_zero(self):
        seg = PolynomialSegment(coeffs=np.array([1.0, 2.0]), duration=1.0)
        assert seg.evaluate(0.5, derivative=2) == pytest.approx(0.0)

    def test_max_abs_value(self):
        # p(t) = t^2 on [0, 2], max |p'(t)| = |2*2| = 4
        seg = PolynomialSegment(coeffs=np.array([0.0, 0.0, 1.0]), duration=2.0)
        max_vel = seg.max_abs_value(derivative=1, n_samples=200)
        assert max_vel == pytest.approx(4.0, abs=0.05)


# ---------------------------------------------------------------------------
# MinJerkSolver
# ---------------------------------------------------------------------------

class TestMinJerkSolver:
    def _simple_waypoints(self):
        return np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 3.0, 0.0],
            [15.0, 3.0, 2.0],
        ])

    def test_boundary_conditions(self):
        solver = MinJerkSolver()
        x0, v0, a0 = 0.0, 1.0, 0.5
        xf, vf, af = 10.0, 0.5, 0.0
        T = 2.0
        seg = solver.solve_segment(x0, v0, a0, xf, vf, af, T)

        assert seg.evaluate(0.0, 0) == pytest.approx(x0, abs=1e-10)
        assert seg.evaluate(0.0, 1) == pytest.approx(v0, abs=1e-10)
        assert seg.evaluate(0.0, 2) == pytest.approx(a0, abs=1e-10)
        assert seg.evaluate(T, 0) == pytest.approx(xf, abs=1e-10)
        assert seg.evaluate(T, 1) == pytest.approx(vf, abs=1e-10)
        assert seg.evaluate(T, 2) == pytest.approx(af, abs=1e-10)

    def test_c2_continuity_at_joints(self):
        solver = MinJerkSolver()
        wp = self._simple_waypoints()
        times = np.array([1.0, 1.5, 1.0])
        segments = solver.solve_trajectory(wp, times)

        for axis in range(3):
            for i in range(len(segments[axis]) - 1):
                T = segments[axis][i].duration
                # Position continuity
                end_pos = segments[axis][i].evaluate(T, 0)
                start_pos = segments[axis][i + 1].evaluate(0.0, 0)
                assert end_pos == pytest.approx(start_pos, abs=1e-6)
                # Velocity continuity
                end_vel = segments[axis][i].evaluate(T, 1)
                start_vel = segments[axis][i + 1].evaluate(0.0, 1)
                assert end_vel == pytest.approx(start_vel, abs=1e-6)
                # Acceleration continuity (C2)
                end_acc = segments[axis][i].evaluate(T, 2)
                start_acc = segments[axis][i + 1].evaluate(0.0, 2)
                assert end_acc == pytest.approx(start_acc, abs=1e-6)

    def test_produces_valid_trajectory_points(self):
        solver = MinJerkSolver()
        wp = self._simple_waypoints()
        times = np.array([1.0, 1.5, 1.0])
        segments = solver.solve_trajectory(wp, times)
        traj = sample_trajectory(segments, dt=0.05)

        assert len(traj) > 0
        assert traj[0].time == pytest.approx(0.0)
        for pt in traj:
            assert pt.position.shape == (3,)
            assert pt.velocity.shape == (3,)
            assert pt.acceleration.shape == (3,)

    def test_speed_limit_respected(self):
        solver = MinJerkSolver()
        wp = np.array([[0, 0, 0], [20, 0, 0], [40, 0, 0]], dtype=float)
        times = np.array([0.5, 0.5])  # very fast -> high speed
        segments = solver.solve_trajectory(wp, times)
        max_speed = 10.0
        traj = sample_trajectory(segments, dt=0.01, max_speed=max_speed)

        for pt in traj:
            speed = np.linalg.norm(pt.velocity)
            assert speed <= max_speed + 0.1


# ---------------------------------------------------------------------------
# MinSnapSolver
# ---------------------------------------------------------------------------

class TestMinSnapSolver:
    def _simple_waypoints(self):
        return np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 3.0, 0.0],
            [15.0, 3.0, 2.0],
        ])

    def test_boundary_conditions(self):
        solver = MinSnapSolver()
        wp = self._simple_waypoints()
        times = np.array([1.0, 1.5, 1.0])
        segments = solver.solve_trajectory(wp, times)

        # Check start position
        for axis in range(3):
            start_pos = segments[axis][0].evaluate(0.0, 0)
            assert start_pos == pytest.approx(wp[0, axis], abs=1e-6)

        # Check end position
        for axis in range(3):
            T = segments[axis][-1].duration
            end_pos = segments[axis][-1].evaluate(T, 0)
            assert end_pos == pytest.approx(wp[-1, axis], abs=1e-6)

        # Check intermediate waypoints
        for i in range(len(times)):
            T = segments[0][i].duration
            for axis in range(3):
                end_pos = segments[axis][i].evaluate(T, 0)
                assert end_pos == pytest.approx(wp[i + 1, axis], abs=1e-6)

    def test_c3_continuity_at_joints(self):
        solver = MinSnapSolver()
        wp = self._simple_waypoints()
        times = np.array([1.0, 1.5, 1.0])
        segments = solver.solve_trajectory(wp, times)

        for axis in range(3):
            for i in range(len(segments[axis]) - 1):
                T = segments[axis][i].duration
                for deriv in range(4):  # pos, vel, acc, jerk
                    end_val = segments[axis][i].evaluate(T, deriv)
                    start_val = segments[axis][i + 1].evaluate(0.0, deriv)
                    assert end_val == pytest.approx(start_val, abs=1e-4), (
                        f"Axis {axis}, joint {i}, derivative {deriv}: "
                        f"{end_val} != {start_val}"
                    )

    def test_smoother_snap_than_min_jerk(self):
        wp = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 2.0, 0.0],
            [10.0, 0.0, 1.0],
            [15.0, 2.0, 1.0],
            [20.0, 0.0, 0.0],
        ])
        times = np.array([1.0, 1.0, 1.0, 1.0])

        jerk_solver = MinJerkSolver()
        snap_solver = MinSnapSolver()

        jerk_segs = jerk_solver.solve_trajectory(wp, times)
        snap_segs = snap_solver.solve_trajectory(wp, times)

        # Compute max snap (4th derivative) for each
        def max_snap(segments):
            peak = 0.0
            for axis in range(3):
                for seg in segments[axis]:
                    val = seg.max_abs_value(derivative=4, n_samples=50)
                    if val > peak:
                        peak = val
            return peak

        jerk_snap = max_snap(jerk_segs)
        snap_snap = max_snap(snap_segs)

        # Min-snap should have lower or equal snap magnitude
        assert snap_snap <= jerk_snap * 1.1  # small tolerance

    def test_solve_time_under_50ms(self):
        # 10 segments (11 waypoints)
        wp = np.column_stack([
            np.linspace(0, 50, 11),
            np.sin(np.linspace(0, 2 * np.pi, 11)) * 5,
            np.ones(11) * 2,
        ])
        times = np.ones(10) * 0.8

        solver = MinSnapSolver()
        start = time.perf_counter()
        solver.solve_trajectory(wp, times)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.05, f"Solve took {elapsed:.3f}s, expected <0.05s"


# ---------------------------------------------------------------------------
# TimeAllocator
# ---------------------------------------------------------------------------

class TestTimeAllocator:
    def test_total_time_reduced(self):
        wp = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [15.0, 0.0, 0.0],
        ])
        # Start with conservative equal times
        initial_times = np.array([2.0, 2.0, 2.0])
        solver = MinJerkSolver()
        allocator = TimeAllocator(min_segment_time=0.1, max_iterations=50)

        optimized = allocator.optimize(
            wp, initial_times,
            max_velocity=15.0, max_acceleration=10.0,
            solver=solver,
        )

        initial_total = np.sum(initial_times)
        optimized_total = np.sum(optimized)
        reduction = (initial_total - optimized_total) / initial_total

        assert reduction > 0.15, (
            f"Expected >15% reduction, got {reduction * 100:.1f}% "
            f"({initial_total:.2f} -> {optimized_total:.2f})"
        )

    def test_respects_minimum_segment_time(self):
        wp = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        initial_times = np.array([1.0, 1.0])
        solver = MinJerkSolver()
        min_t = 0.2
        allocator = TimeAllocator(min_segment_time=min_t)

        optimized = allocator.optimize(
            wp, initial_times,
            max_velocity=15.0, max_acceleration=10.0,
            solver=solver,
        )

        for t in optimized:
            assert t >= min_t - 1e-6

    def test_speed_accel_limits_respected(self):
        wp = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 5.0, 0.0],
        ])
        initial_times = np.array([2.0, 2.0])
        max_vel = 12.0
        max_acc = 8.0
        solver = MinJerkSolver()
        allocator = TimeAllocator()

        optimized = allocator.optimize(
            wp, initial_times,
            max_velocity=max_vel, max_acceleration=max_acc,
            solver=solver,
        )

        # Verify limits by solving with optimized times
        segments = solver.solve_trajectory(wp, optimized)
        for seg_idx in range(len(segments[0])):
            T = segments[0][seg_idx].duration
            for t in np.linspace(0, T, 50):
                vel = np.array([segments[ax][seg_idx].evaluate(t, 1) for ax in range(3)])
                acc = np.array([segments[ax][seg_idx].evaluate(t, 2) for ax in range(3)])
                speed = np.linalg.norm(vel)
                accel = np.linalg.norm(acc)
                # Allow small tolerance from optimizer
                assert speed <= max_vel * 1.05, f"Speed {speed:.2f} > limit {max_vel}"
                assert accel <= max_acc * 1.05, f"Accel {accel:.2f} > limit {max_acc}"
