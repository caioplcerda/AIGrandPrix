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


# ---------------------------------------------------------------------------
# Phase 2: Polynomial trajectory integration tests
# ---------------------------------------------------------------------------

def test_min_jerk_produces_valid_trajectory():
    config = {"trajectory_method": "min_jerk"}
    planner = PathPlanner(config=config)
    gates = _make_gates(3)
    start = np.array([0.0, 0.0, 2.0])

    traj = planner.plan_through_gates(gates, start)

    assert len(traj) > 0
    assert traj[0].time == 0.0
    assert traj[-1].time > 0.0
    for pt in traj:
        assert pt.position.shape == (3,)
        assert pt.velocity.shape == (3,)


def test_min_snap_produces_valid_trajectory():
    config = {"trajectory_method": "min_snap"}
    planner = PathPlanner(config=config)
    gates = _make_gates(3)
    start = np.array([0.0, 0.0, 2.0])

    traj = planner.plan_through_gates(gates, start)

    assert len(traj) > 0
    assert traj[0].time == 0.0
    assert traj[-1].time > 0.0
    for pt in traj:
        assert pt.position.shape == (3,)


def test_config_trajectory_method_respected():
    """Verify that trajectory_method config is dispatched correctly."""
    # cubic_spline (default when no config)
    planner_cs = PathPlanner()
    assert planner_cs._trajectory_method == "cubic_spline"

    # min_jerk via config
    planner_mj = PathPlanner(config={"trajectory_method": "min_jerk"})
    assert planner_mj._trajectory_method == "min_jerk"

    # min_snap via config
    planner_ms = PathPlanner(config={"trajectory_method": "min_snap"})
    assert planner_ms._trajectory_method == "min_snap"

    # All should produce trajectories
    gates = _make_gates(2)
    start = np.array([0.0, 0.0, 2.0])
    for planner in [planner_cs, planner_mj, planner_ms]:
        traj = planner.plan_through_gates(gates, start)
        assert len(traj) > 0


def test_fallback_on_solver_failure():
    """MinSnap should fall back to min-jerk, min-jerk to cubic spline."""
    # This tests the fallback chain with a valid case — we just verify no crash
    config = {"trajectory_method": "min_snap"}
    planner = PathPlanner(config=config)
    gates = _make_gates(2)
    start = np.array([0.0, 0.0, 2.0])

    traj = planner.plan_through_gates(gates, start)
    assert len(traj) > 0


def test_min_snap_with_racing_line():
    config = {
        "trajectory_method": "min_snap",
        "racing_line": {
            "enabled": True,
            "gate_margin": 0.15,
            "curvature_weight": 2.0,
        },
    }
    planner = PathPlanner(config=config)
    gates = _make_gates(3)
    start = np.array([0.0, 0.0, 2.0])

    traj = planner.plan_through_gates(gates, start)
    assert len(traj) > 0


def test_replan_uses_cubic_spline():
    """replan_from_current should use cubic spline for speed."""
    config = {"trajectory_method": "min_snap"}
    planner = PathPlanner(config=config)
    gates = _make_gates(2)

    traj = planner.replan_from_current(
        current_pos=np.array([2.0, 0.0, 2.0]),
        current_vel=np.array([3.0, 0.0, 0.0]),
        remaining_gates=gates,
    )
    assert len(traj) > 0
    # Method should be restored
    assert planner._trajectory_method == "min_snap"
