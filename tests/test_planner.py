"""Tests for path planner."""

import numpy as np

from aigrandprix.planning.path_planner import Gate, PathPlanner


def _make_gates(n=3):
    gates = []
    for i in range(n):
        gates.append(Gate(
            position=np.array([5.0 * (i + 1), 0.0, 2.0]),
            normal=np.array([1.0, 0.0, 0.0]),
            gate_id=i,
        ))
    return gates


def test_plan_produces_trajectory():
    planner = PathPlanner()
    gates = _make_gates(3)
    start = np.array([0.0, 0.0, 2.0])

    traj = planner.plan_through_gates(gates, start)

    assert len(traj) > 0
    assert traj[0].time == 0.0
    assert traj[-1].time > 0.0


def test_trajectory_starts_at_position():
    planner = PathPlanner()
    gates = _make_gates(2)
    start = np.array([0.0, 0.0, 2.0])

    traj = planner.plan_through_gates(gates, start)

    np.testing.assert_allclose(traj[0].position, start, atol=0.1)


def test_speed_limit_respected():
    planner = PathPlanner(max_speed=10.0)
    gates = _make_gates(3)
    start = np.zeros(3)

    traj = planner.plan_through_gates(gates, start)

    for pt in traj:
        speed = np.linalg.norm(pt.velocity)
        assert speed <= planner.max_speed + 0.1  # small tolerance
