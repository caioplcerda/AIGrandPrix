"""E2E test: autonomy stack vs DCL mock server (VADR-TS-002 interface).

Spins up DCLMockServer in a background thread on port 14551 with a straight
5-gate course, runs the full autonomy loop (MAVLink client + NED state estimator
+ NED path planner), and asserts all 5 gates are passed within 90 s.

Usage:
    python scripts/test_e2e_mock.py
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("e2e")

_MAVLINK_PORT = 14551   # non-default to avoid conflicts
_VISION_PORT = 15600    # unused by test, server sends here silently
_N_GATES = 5
_MAX_DURATION_S = 90.0
_LOOP_HZ = 50.0
_MAX_SPEED = 6.0
_GATE_PASS_RADIUS_M = 1.5
_REPLAN_INTERVAL_S = 1.5


def run_server(gates, result_out: dict, stop_event: threading.Event) -> None:
    from aigrandprix.mock_sim.dcl_mock_server import DCLMockServer

    server = DCLMockServer(
        gates=gates,
        mavlink_port=_MAVLINK_PORT,
        vision_port=_VISION_PORT,
        host="127.0.0.1",
    )
    result = server.run(max_duration_s=_MAX_DURATION_S)
    result_out.update(result)
    stop_event.set()  # wake client after server finishes


def main() -> int:
    from aigrandprix.mock_sim.dcl_mock_server import build_straight_course
    from aigrandprix.comms.mavlink_client import MAVLinkClient, PositionTargetNED
    from aigrandprix.state.state_estimator import NEDStateEstimator
    from aigrandprix.planning.path_planner_ned import GateNED, PathPlannerNED, WaypointNED

    gates_mock = build_straight_course(
        n_gates=_N_GATES,
        spacing_m=12.0,
        altitude_ned=-2.5,
        start_north_m=12.0,
    )
    gate_neds = [
        GateNED(
            position=g.center_ned.copy(),
            normal=g.normal_ned.copy(),
            gate_id=g.gate_id,
        )
        for g in gates_mock
    ]

    server_result: dict = {}
    stop_event = threading.Event()

    logger.info("Starting mock server on MAVLink port %d …", _MAVLINK_PORT)
    server_thread = threading.Thread(
        target=run_server,
        args=(gates_mock, server_result, stop_event),
        daemon=True,
        name="mock-server",
    )
    server_thread.start()
    time.sleep(0.15)  # let server bind sockets

    # ── MAVLink client
    client = MAVLinkClient(host="127.0.0.1", mavlink_port=_MAVLINK_PORT, heartbeat_rate_hz=5.0)
    client.connect()
    client.start()

    logger.info("Waiting for server heartbeat …")
    t_wait = time.time()
    while not client.connected and time.time() - t_wait < 15.0:
        time.sleep(0.05)

    if not client.connected:
        logger.error("FAIL: no heartbeat received in 15 s")
        client.stop()
        return 1

    logger.info("Connected! Running %d-gate course …", _N_GATES)

    estimator = NEDStateEstimator()
    planner = PathPlannerNED(
        max_speed=_MAX_SPEED,
        approach_distance=2.5,
        exit_distance=1.5,
    )

    loop_dt = 1.0 / _LOOP_HZ
    next_gate_idx = 0
    waypoints: list[WaypointNED] = []
    last_replan = -_REPLAN_INTERVAL_S  # force immediate plan
    wp = WaypointNED(pos=np.zeros(3), vel=np.zeros(3), yaw=0.0, time=0.0)
    t_start = time.time()

    while True:
        t_loop = time.perf_counter()
        elapsed = time.time() - t_start

        if elapsed > _MAX_DURATION_S:
            logger.warning("Timeout at %.1f s", elapsed)
            break

        if stop_event.is_set():
            logger.info("Server stopped (all gates passed or timeout)")
            break

        if next_gate_idx >= _N_GATES:
            # Client thinks all gates done; keep commanding past last gate until server confirms
            if stop_event.wait(timeout=0.0):
                logger.info("Server confirmed completion at %.1f s", elapsed)
                break
            # Command well past last gate so drone fully crosses plane
            last_gate = gate_neds[-1]
            exit_pos = last_gate.position + last_gate.normal * 6.0  # 6m past gate
            target = PositionTargetNED(
                x=float(exit_pos[0]),
                y=float(exit_pos[1]),
                z=float(exit_pos[2]),
                vx=float(last_gate.normal[0] * _MAX_SPEED),  # full speed through
                vy=float(last_gate.normal[1] * _MAX_SPEED),
                vz=0.0,
                yaw=0.0,
            )
            client.send_position_target(target)
            t_used = time.perf_counter() - t_loop
            sleep_t = loop_dt - t_used
            if sleep_t > 0:
                time.sleep(sleep_t)
            continue

        telem = client.telemetry
        if not telem.connected:
            time.sleep(0.05)
            continue

        state = estimator.update(telem)

        # Gate pass check (client-side by estimated distance)
        # Use gate center position: advance only if approaching AND within radius
        remaining = gate_neds[next_gate_idx:]
        if remaining:
            g = remaining[0]
            diff = g.position - state.pos_ned
            dist = float(np.linalg.norm(diff))
            # Check drone has crossed gate plane (dot product with normal flips sign)
            past_gate = float(np.dot(diff, g.normal)) < 0
            if dist < _GATE_PASS_RADIUS_M or past_gate:
                logger.info(
                    "Gate %d in range (est dist=%.2fm) — advancing planner",
                    g.gate_id, dist,
                )
                next_gate_idx += 1
                waypoints = []
                estimator.reset_position(g.position)

        # Replan
        remaining = gate_neds[next_gate_idx:]
        if remaining and (not waypoints or elapsed - last_replan > _REPLAN_INTERVAL_S):
            waypoints = planner.plan(remaining, state.pos_ned, state.vel_ned)
            last_replan = elapsed

        # Select waypoint target
        if waypoints:
            wp = planner.next_position_target(waypoints, state.pos_ned, lookahead_m=3.0)
        else:
            wp = WaypointNED(pos=state.pos_ned.copy(), vel=np.zeros(3), yaw=state.yaw, time=elapsed)

        if wp is None:
            wp = WaypointNED(pos=state.pos_ned.copy(), vel=np.zeros(3), yaw=state.yaw, time=elapsed)

        target = PositionTargetNED(
            x=float(wp.pos[0]),
            y=float(wp.pos[1]),
            z=float(wp.pos[2]),
            vx=float(wp.vel[0]),
            vy=float(wp.vel[1]),
            vz=float(wp.vel[2]),
            yaw=float(wp.yaw),
        )
        client.send_position_target(target)

        # Loop timing
        t_used = time.perf_counter() - t_loop
        sleep_t = loop_dt - t_used
        if sleep_t > 0:
            time.sleep(sleep_t)

    client.stop()

    # Wait for server to finish and report
    server_thread.join(timeout=15.0)

    gates_passed = server_result.get("gates_passed", 0)
    total_gates = server_result.get("total_gates", _N_GATES)
    total_time = server_result.get("total_time_s", 0.0)
    completed = server_result.get("completed", False)

    print()
    print("=" * 50)
    print(f"  Gates passed : {gates_passed} / {total_gates}")
    print(f"  Total time   : {total_time:.2f} s")
    print(f"  Completed    : {completed}")
    print("=" * 50)

    if gates_passed >= _N_GATES:
        print("  RESULT: PASS ✓")
        return 0
    else:
        print(f"  RESULT: FAIL ✗  ({gates_passed}/{_N_GATES} gates)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
