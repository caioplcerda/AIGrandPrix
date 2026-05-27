"""Hard course battery — 3D paths, diagonal normals, 10-gate circuit.

Extends test_curved_courses.py with more demanding scenarios:
  - chicane_3d: S-curve with 2.5m altitude swings, no diagonal normals
  - spiral_helix: 3D spiral path, +5m altitude gain, diagonal normals
  - compound_3d_diagonal: U-turn loop + altitude changes + diagonal normals
  - vq1_10gate_circuit: 10-gate realistic competition-scale closed circuit

All courses use identical parameters to run_vq1.py (max_speed=15, approach=6, lookahead=5).

Usage:
    python scripts/test_hard_courses.py
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hard_test")

_MAX_DURATION_S = 120.0
_LOOP_HZ = 50.0
_MAX_SPEED = 25.0
_REPLAN_INTERVAL_S = 0.25
_INNER_HALF = 0.75


def _gate_crossed(
    gate_pos: np.ndarray,
    gate_normal: np.ndarray,
    old_pos: np.ndarray,
    new_pos: np.ndarray,
    inner_half: float = _INNER_HALF,
) -> bool:
    n = gate_normal
    c = gate_pos
    d_old = float(np.dot(old_pos - c, n))
    d_new = float(np.dot(new_pos - c, n))
    if d_old * d_new > 0 or abs(d_old - d_new) < 1e-9:
        return False
    t = d_old / (d_old - d_new)
    cross = old_pos + t * (new_pos - old_pos)
    diff = cross - c
    up_ned = np.array([0.0, 0.0, -1.0])
    right_ned = np.cross(n, up_ned)
    rn = float(np.linalg.norm(right_ned))
    right_ned = right_ned / rn if rn > 1e-6 else np.array([0.0, 1.0, 0.0])
    lat = abs(float(np.dot(diff, right_ned)))
    vert = abs(float(np.dot(diff, up_ned)))
    return lat <= inner_half and vert <= inner_half


def run_course(
    course_name: str,
    gate_defs: list[tuple[np.ndarray, np.ndarray]],
    mavlink_port: int,
    vision_port: int,
) -> dict:
    from aigrandprix.mock_sim.dcl_mock_server import DCLMockServer, MockGate
    from aigrandprix.comms.mavlink_client import MAVLinkClient, PositionTargetNED
    from aigrandprix.state.state_estimator import NEDStateEstimator
    from aigrandprix.planning.path_planner_ned import GateNED, PathPlannerNED, WaypointNED

    n_gates = len(gate_defs)
    gates_mock = [
        MockGate(center_ned=pos.copy(), normal_ned=norm.copy(), gate_id=i)
        for i, (pos, norm) in enumerate(gate_defs)
    ]
    gate_neds = [
        GateNED(position=pos.copy(), normal=norm.copy(), gate_id=i)
        for i, (pos, norm) in enumerate(gate_defs)
    ]

    server_result: dict = {}
    server_err: list[str] = []
    stop_event = threading.Event()

    def run_server() -> None:
        try:
            server = DCLMockServer(
                gates=gates_mock,
                mavlink_port=mavlink_port,
                vision_port=vision_port,
                host="127.0.0.1",
            )
            r = server.run(max_duration_s=_MAX_DURATION_S)
            server_result.update(r)
        except Exception:
            import traceback
            server_err.append(traceback.format_exc())
        finally:
            stop_event.set()

    server_thread = threading.Thread(target=run_server, daemon=True, name=f"server-{course_name}")
    server_thread.start()
    time.sleep(0.2)

    if server_err:
        return {"course": course_name, "error": server_err[0], "passed": False}

    client = MAVLinkClient(host="127.0.0.1", mavlink_port=mavlink_port, heartbeat_rate_hz=5.0)
    client.connect()
    client.start()

    t_wait = time.time()
    while not client.connected and time.time() - t_wait < 15.0:
        time.sleep(0.05)

    if not client.connected:
        client.stop()
        return {"course": course_name, "error": "no heartbeat in 15s", "passed": False}

    estimator = NEDStateEstimator()
    planner = PathPlannerNED(max_speed=_MAX_SPEED, approach_distance=6.0, exit_distance=1.0)

    loop_dt = 1.0 / _LOOP_HZ
    next_gate_idx = 0
    waypoints: list[WaypointNED] = []
    last_replan = -_REPLAN_INTERVAL_S
    wp = WaypointNED(pos=np.zeros(3), vel=np.zeros(3), yaw=0.0, time=0.0)
    t_start = time.time()
    prev_pos: np.ndarray | None = None

    while True:
        t_loop = time.perf_counter()
        elapsed = time.time() - t_start

        if elapsed > _MAX_DURATION_S or stop_event.is_set():
            break

        if next_gate_idx >= n_gates:
            last_gate = gate_neds[-1]
            exit_pos = last_gate.position + last_gate.normal * 6.0
            client.send_position_target(PositionTargetNED(
                x=float(exit_pos[0]), y=float(exit_pos[1]), z=float(exit_pos[2]),
                vx=float(last_gate.normal[0] * _MAX_SPEED),
                vy=float(last_gate.normal[1] * _MAX_SPEED),
                vz=0.0, yaw=0.0,
            ))
            t_used = time.perf_counter() - t_loop
            if (s := loop_dt - t_used) > 0:
                time.sleep(s)
            continue

        telem = client.telemetry
        if not telem.connected:
            time.sleep(0.05)
            continue

        state = estimator.update(telem)
        curr_pos = state.pos_ned

        if prev_pos is not None and next_gate_idx < n_gates:
            g = gate_neds[next_gate_idx]
            if _gate_crossed(g.position, g.normal, prev_pos, curr_pos):
                estimator.reset_position(g.position)
                next_gate_idx += 1
                waypoints = []

        prev_pos = curr_pos.copy()

        remaining = gate_neds[next_gate_idx:]
        if remaining and (not waypoints or elapsed - last_replan > _REPLAN_INTERVAL_S):
            waypoints = planner.plan(remaining[:1], state.pos_ned, state.vel_ned)
            last_replan = elapsed

        if waypoints:
            wp = planner.next_position_target(waypoints, state.pos_ned, lookahead_m=5.0)
        else:
            wp = WaypointNED(pos=state.pos_ned.copy(), vel=np.zeros(3), yaw=state.yaw, time=elapsed)

        if wp is None:
            wp = WaypointNED(pos=state.pos_ned.copy(), vel=np.zeros(3), yaw=state.yaw, time=elapsed)

        # Approach-waypoint velocity blend: fly toward approach point at max_speed.
        # Fades to spline vel within 4m of approach point for precise gate traversal.
        _APPROACH_DIST = 6.0
        if remaining:
            g0 = remaining[0]
            approach_pt = g0.position - g0.normal * _APPROACH_DIST
            approach_vec = approach_pt - state.pos_ned
            dist_to_approach = float(np.linalg.norm(approach_vec))
            ahead = float(np.dot(approach_vec, g0.normal)) > 0.0
            if dist_to_approach > 0.1 and ahead:
                blend = min(1.0, max(0.0, (dist_to_approach - 4.0) / 8.0))
                approach_vel = approach_vec / dist_to_approach * _MAX_SPEED
                cmd_vel = blend * approach_vel + (1.0 - blend) * wp.vel
            else:
                cmd_vel = wp.vel
        else:
            cmd_vel = wp.vel

        client.send_position_target(PositionTargetNED(
            x=float(wp.pos[0]), y=float(wp.pos[1]), z=float(wp.pos[2]),
            vx=float(cmd_vel[0]), vy=float(cmd_vel[1]), vz=float(cmd_vel[2]),
            yaw=float(wp.yaw),
        ))

        t_used = time.perf_counter() - t_loop
        if (s := loop_dt - t_used) > 0:
            time.sleep(s)

    client.stop()
    server_thread.join(timeout=15.0)

    gates_passed = server_result.get("gates_passed", 0)
    total_time = server_result.get("total_time_s", 0.0)
    completed = server_result.get("completed", False)

    return {
        "course": course_name,
        "gates_passed": gates_passed,
        "total_gates": n_gates,
        "total_time_s": total_time,
        "completed": completed,
        "passed": gates_passed >= n_gates,
        "error": server_err[0] if server_err else None,
    }


# ── Course definitions ────────────────────────────────────────────────────────

def _norm(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=float)
    return a / np.linalg.norm(a)


HARD_COURSES = [
    # S-curve with aggressive altitude swings (+2.5m per gate) — no diagonal normals
    (
        "chicane_3d",  # baseline 3D
        [
            (np.array([12.0,  4.0, -1.5]), np.array([1.0, 0.0, 0.0])),
            (np.array([24.0, -4.0, -4.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([36.0,  4.0, -1.5]), np.array([1.0, 0.0, 0.0])),
            (np.array([48.0, -4.0, -4.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([60.0,  0.0, -2.5]), np.array([1.0, 0.0, 0.0])),
        ],
        14571, 15711,
    ),
    # 3D clockwise spiral — 6 gates, +5m altitude gain, diagonal normals
    (
        "spiral_helix",
        [
            (np.array([12.0,  0.0, -2.0]), _norm([1.0, 0.0, 0.0])),
            (np.array([18.0,  8.0, -3.5]), _norm([0.707, 0.707, 0.0])),
            (np.array([ 8.0, 16.0, -5.0]), _norm([-0.707, 0.707, 0.0])),
            (np.array([ 0.0, 10.0, -6.0]), _norm([-1.0, 0.0, 0.0])),
            (np.array([ 4.0,  2.0, -7.0]), _norm([0.707, -0.707, 0.0])),
            (np.array([12.0,  0.0, -7.5]), _norm([1.0, 0.0, 0.0])),
        ],
        14572, 15712,
    ),
    # U-turn loop: diagonal normals + altitude changes combined
    (
        "compound_3d_diagonal",
        [
            (np.array([12.0,  0.0, -2.5]), _norm([1.0, 0.0, 0.0])),
            (np.array([20.0,  6.0, -1.5]), _norm([0.707, 0.707, 0.0])),
            (np.array([26.0, 14.0, -3.5]), _norm([0.0, 1.0, 0.0])),
            (np.array([20.0, 22.0, -1.5]), _norm([-0.707, 0.707, 0.0])),
            (np.array([12.0, 22.0, -3.5]), _norm([-1.0, 0.0, 0.0])),
            (np.array([ 4.0, 14.0, -2.5]), _norm([-0.707, -0.707, 0.0])),
        ],
        14573, 15713,
    ),
    # VQ1-scale 10-gate closed circuit — mixed normals, altitude, S-curves
    (
        "vq1_10gate_circuit",
        [
            (np.array([12.0,  0.0, -2.5]), _norm([1.0, 0.0, 0.0])),
            (np.array([24.0,  5.0, -3.5]), _norm([1.0, 0.0, 0.0])),
            (np.array([36.0,  0.0, -1.5]), _norm([1.0, 0.0, 0.0])),
            (np.array([48.0, -5.0, -3.5]), _norm([0.707, -0.707, 0.0])),  # right turn
            (np.array([54.0,-16.0, -2.5]), _norm([0.0, -1.0, 0.0])),
            (np.array([46.0,-24.0, -3.5]), _norm([-0.707, -0.707, 0.0])),  # U-turn apex
            (np.array([30.0,-24.0, -2.5]), _norm([-1.0, 0.0, 0.0])),
            (np.array([18.0,-16.0, -3.5]), _norm([-0.707, 0.707, 0.0])),
            (np.array([10.0, -6.0, -2.5]), _norm([0.0, 1.0, 0.0])),       # final turn home
            (np.array([12.0,  4.0, -2.5]), _norm([1.0, 0.0, 0.0])),
        ],
        14574, 15714,
    ),
    # Tight spacing: 8m between gates, steep altitude swings, forward normals
    (
        "tight_gates_8m",
        [
            (np.array([ 8.0,  0.0, -2.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([16.0,  3.0, -4.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([24.0, -3.0, -1.5]), np.array([1.0, 0.0, 0.0])),
            (np.array([32.0,  3.0, -4.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([40.0,  0.0, -2.5]), np.array([1.0, 0.0, 0.0])),
        ],
        14575, 15715,
    ),
    # Extreme: 3D spiral + diagonal normals, tight 8m spacing
    (
        "extreme_3d_tight",
        [
            (np.array([ 8.0,  0.0, -2.0]), _norm([1.0, 0.0, 0.0])),
            (np.array([14.0,  6.0, -3.5]), _norm([0.707, 0.707, 0.0])),
            (np.array([ 8.0, 12.0, -5.0]), _norm([-0.707, 0.707, 0.0])),
            (np.array([ 2.0,  8.0, -6.0]), _norm([-1.0, 0.0, 0.0])),
            (np.array([ 6.0,  2.0, -7.0]), _norm([0.707, -0.707, 0.0])),
            (np.array([14.0,  0.0, -5.5]), _norm([1.0, 0.0, 0.0])),
        ],
        14576, 15716,
    ),
]


def main() -> int:
    print("\nHARD COURSE BATTERY")
    print("=" * 60)

    results = []
    for course_name, gate_defs, mav_port, vis_port in HARD_COURSES:
        print(f"\nRunning: {course_name} ({len(gate_defs)} gates) ...", flush=True)
        r = run_course(course_name, gate_defs, mav_port, vis_port)
        results.append(r)
        if r.get("error"):
            print(f"  ERROR: {r['error'][:300]}")
        else:
            status = "PASS ✓" if r["passed"] else "FAIL ✗"
            print(f"  {status}  {r['gates_passed']}/{r['total_gates']} gates  {r['total_time_s']:.2f}s")
        time.sleep(1.0)

    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r.get("passed"))
    print(f"TOTAL: {passed}/{len(results)} courses passed")
    print("=" * 60)

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
