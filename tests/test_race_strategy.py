"""Tests for racing line optimization."""

import numpy as np
import pytest

from aigrandprix.planning.path_planner import Gate
from aigrandprix.planning.race_strategy import RacingLineOptimizer


def _make_straight_gates(n=3, spacing=5.0):
    """Gates in a straight line along X axis."""
    gates = []
    for i in range(n):
        gates.append(Gate(
            position=np.array([spacing * (i + 1), 0.0, 2.0]),
            normal=np.array([1.0, 0.0, 0.0]),
            gate_id=i,
            width=1.0,
            height=1.0,
        ))
    return gates


def _make_offset_gates():
    """Gates with lateral offsets to encourage corner-cutting."""
    return [
        Gate(position=np.array([5.0, 0.0, 2.0]), normal=np.array([1.0, 0.0, 0.0]), gate_id=0),
        Gate(position=np.array([10.0, 5.0, 2.0]), normal=np.array([1.0, 0.0, 0.0]), gate_id=1),
        Gate(position=np.array([15.0, 0.0, 2.0]), normal=np.array([1.0, 0.0, 0.0]), gate_id=2),
    ]


class TestGateFrame:
    def test_orthonormal_vectors(self):
        optimizer = RacingLineOptimizer()
        gate = Gate(
            position=np.array([5.0, 0.0, 2.0]),
            normal=np.array([1.0, 0.0, 0.0]),
            gate_id=0,
        )
        right, up = optimizer.gate_frame(gate)

        # Unit vectors
        assert np.linalg.norm(right) == pytest.approx(1.0, abs=1e-10)
        assert np.linalg.norm(up) == pytest.approx(1.0, abs=1e-10)

        # Orthogonal to each other and to normal
        assert abs(np.dot(right, up)) < 1e-10
        assert abs(np.dot(right, gate.normal)) < 1e-10
        assert abs(np.dot(up, gate.normal)) < 1e-10

    def test_vertical_normal_fallback(self):
        optimizer = RacingLineOptimizer()
        gate = Gate(
            position=np.array([0.0, 0.0, 5.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            gate_id=0,
        )
        right, up = optimizer.gate_frame(gate)

        assert np.linalg.norm(right) == pytest.approx(1.0, abs=1e-10)
        assert np.linalg.norm(up) == pytest.approx(1.0, abs=1e-10)
        assert abs(np.dot(right, up)) < 1e-10

    def test_angled_normal(self):
        optimizer = RacingLineOptimizer()
        normal = np.array([1.0, 1.0, 0.0])
        normal = normal / np.linalg.norm(normal)
        gate = Gate(position=np.zeros(3), normal=normal, gate_id=0)
        right, up = optimizer.gate_frame(gate)

        assert np.linalg.norm(right) == pytest.approx(1.0, abs=1e-10)
        assert np.linalg.norm(up) == pytest.approx(1.0, abs=1e-10)
        assert abs(np.dot(right, normal)) < 1e-10
        assert abs(np.dot(up, normal)) < 1e-10


class TestOptimizeGates:
    def test_crossing_within_gate_bounds(self):
        optimizer = RacingLineOptimizer(gate_margin=0.15)
        gates = _make_offset_gates()
        start = np.array([0.0, 0.0, 2.0])

        optimized = optimizer.optimize_gates(gates, start)

        for orig, opt in zip(gates, optimized):
            right, up = optimizer.gate_frame(orig)
            diff = opt.position - orig.position
            u = np.dot(diff, right)
            v = np.dot(diff, up)

            u_max = orig.width / 2.0 - optimizer.gate_margin
            v_max = orig.height / 2.0 - optimizer.gate_margin

            assert abs(u) <= u_max + 1e-6, f"u={u:.4f} exceeds bound {u_max:.4f}"
            assert abs(v) <= v_max + 1e-6, f"v={v:.4f} exceeds bound {v_max:.4f}"

    def test_optimized_path_shorter_or_equal(self):
        optimizer = RacingLineOptimizer()
        gates = _make_offset_gates()
        start = np.array([0.0, 0.0, 2.0])

        optimized = optimizer.optimize_gates(gates, start)

        # Path through center of gates
        center_pts = [start] + [g.position for g in gates]
        center_length = sum(
            np.linalg.norm(center_pts[i + 1] - center_pts[i])
            for i in range(len(center_pts) - 1)
        )

        opt_pts = [start] + [g.position for g in optimized]
        opt_length = sum(
            np.linalg.norm(opt_pts[i + 1] - opt_pts[i])
            for i in range(len(opt_pts) - 1)
        )

        assert opt_length <= center_length + 1e-6

    def test_collinear_gates_stay_near_center(self):
        optimizer = RacingLineOptimizer()
        gates = _make_straight_gates(3)
        start = np.array([0.0, 0.0, 2.0])

        optimized = optimizer.optimize_gates(gates, start)

        for orig, opt in zip(gates, optimized):
            dist = np.linalg.norm(opt.position - orig.position)
            assert dist < 0.1, f"Shifted {dist:.4f}m from center on collinear gates"

    def test_offset_gates_cut_corners(self):
        optimizer = RacingLineOptimizer()
        gates = _make_offset_gates()
        start = np.array([0.0, 0.0, 2.0])

        optimized = optimizer.optimize_gates(gates, start)

        # The middle gate (offset in Y) should have position adjusted
        mid_orig = gates[1].position
        mid_opt = optimized[1].position

        # The optimizer should shift at least slightly toward the line
        # between gate 0 and gate 2
        midpoint_line = (gates[0].position + gates[2].position) / 2.0
        dist_orig = np.linalg.norm(mid_orig - midpoint_line)
        dist_opt = np.linalg.norm(mid_opt - midpoint_line)

        # Optimized should be closer to or equal to the midpoint line
        assert dist_opt <= dist_orig + 1e-6

    def test_normals_unchanged(self):
        optimizer = RacingLineOptimizer()
        gates = _make_offset_gates()
        start = np.array([0.0, 0.0, 2.0])

        optimized = optimizer.optimize_gates(gates, start)

        for orig, opt in zip(gates, optimized):
            np.testing.assert_array_equal(opt.normal, orig.normal)

    def test_empty_gates(self):
        optimizer = RacingLineOptimizer()
        result = optimizer.optimize_gates([], np.zeros(3))
        assert result == []
