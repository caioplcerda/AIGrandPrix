"""AI Grand Prix VQ1 entry point — connects to DCL simulator via MAVLink2/UDP.

Usage (against mock sim):
    python run_vq1.py --host localhost --mavlink_port 14550 --vision_port 5600

Usage (against DCL sim when released):
    python run_vq1.py --host <dcl_sim_ip> --mavlink_port 14550 --vision_port 5600

The only change needed when switching from mock to DCL: --host value.

Architecture:
    MAVLinkClient ──(telemetry)──► NEDStateEstimator ──► PathPlannerNED ──► MAVLinkClient
    VisionStreamReceiver ──(frames)──► GateDetector ──► (gate position correction)

Loop: 50 Hz command rate, heartbeat 2 Hz (background), vision 30 Hz (background).
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vq1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Grand Prix VQ1 autonomy stack")
    p.add_argument("--host", default="localhost", help="DCL simulator host")
    p.add_argument("--mavlink_port", type=int, default=14550)
    p.add_argument("--vision_port", type=int, default=5600)
    p.add_argument("--max_speed", type=float, default=8.0, help="max flight speed m/s")
    p.add_argument("--loop_hz", type=float, default=50.0, help="control loop rate Hz")
    p.add_argument(
        "--gates",
        nargs="+",
        default=None,
        help="Gate positions as 'north,east,down nx,ny,nz' pairs. "
             "If not provided, uses zero gate list (no planning, just hover).",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> int:
    from aigrandprix.comms.mavlink_client import MAVLinkClient, PositionTargetNED
    from aigrandprix.comms.vision_stream import VisionStreamReceiver
    from aigrandprix.state.state_estimator import NEDStateEstimator
    from aigrandprix.planning.path_planner_ned import GateNED, PathPlannerNED, WaypointNED

    # ── MAVLink client
    client = MAVLinkClient(
        host=args.host,
        mavlink_port=args.mavlink_port,
        heartbeat_rate_hz=2.0,
    )
    client.connect()
    client.start()

    # ── Vision stream receiver
    vision = VisionStreamReceiver(host="0.0.0.0", port=args.vision_port)
    vision.start()

    # ── State estimator (NED, no GPS)
    estimator = NEDStateEstimator()

    # ── Path planner (NED)
    planner = PathPlannerNED(max_speed=args.max_speed)

    # ── Parse gates from args (or empty list)
    gates: list[GateNED] = []
    if args.gates:
        # Expect pairs: "north,east,down" "nx,ny,nz" "north,east,down" "nx,ny,nz" ...
        it = iter(args.gates)
        gid = 0
        for pos_str, norm_str in zip(it, it):
            pos = np.array([float(v) for v in pos_str.split(",")])
            norm = np.array([float(v) for v in norm_str.split(",")])
            gates.append(GateNED(position=pos, normal=norm, gate_id=gid))
            gid += 1

    logger.info(
        "Connecting to sim at %s:%d | gates: %d | max_speed: %.1f m/s",
        args.host, args.mavlink_port, len(gates), args.max_speed,
    )

    # ── Wait for initial telemetry
    logger.info("Waiting for MAVLink heartbeat...")
    t_wait = time.time()
    while not client.connected and time.time() - t_wait < 30.0:
        time.sleep(0.1)

    if not client.connected:
        logger.error("No heartbeat received in 30s — exiting")
        client.stop()
        vision.stop()
        return 1

    logger.info("Connected! Starting control loop at %.0f Hz", args.loop_hz)

    # ── Control loop
    loop_dt = 1.0 / args.loop_hz
    next_gate_idx = 0
    waypoints: list[WaypointNED] = []
    last_replan = 0.0
    REPLAN_INTERVAL_S = 2.0
    GATE_PASS_RADIUS_M = 1.5

    running = True

    def _handle_sigint(sig: int, frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _handle_sigint)

    start_time = time.time()

    while running:
        t_loop = time.perf_counter()
        elapsed = time.time() - start_time

        # VQ1 max run = 8 minutes
        if elapsed > 480.0:
            logger.info("Max run duration reached (480 s)")
            break

        # All gates passed → hold position
        if next_gate_idx >= len(gates) and gates:
            logger.info("All %d gates passed in %.1f s!", len(gates), elapsed)
            break

        # ── Update state from telemetry
        telem = client.telemetry
        if not telem.connected:
            logger.warning("Lost MAVLink connection — hovering")
            client.send_hover()
            time.sleep(0.1)
            continue

        state = estimator.update(telem)

        # ── Gate pass check
        remaining_gates = gates[next_gate_idx:]
        if remaining_gates:
            next_gate = remaining_gates[0]
            dist_to_gate = float(np.linalg.norm(next_gate.position - state.pos_ned))
            if dist_to_gate < GATE_PASS_RADIUS_M:
                logger.info(
                    "Gate %d detected as passed (dist=%.2fm) — advancing",
                    next_gate.gate_id, dist_to_gate,
                )
                next_gate_idx += 1
                waypoints = []  # force replan
                estimator.reset_position(state.pos_ned)  # anchor drift

        # ── Replan trajectory
        remaining_gates = gates[next_gate_idx:]
        if remaining_gates and (not waypoints or elapsed - last_replan > REPLAN_INTERVAL_S):
            waypoints = planner.plan(remaining_gates, state.pos_ned, state.vel_ned)
            last_replan = elapsed
            logger.debug("Replanned: %d waypoints for %d gates", len(waypoints), len(remaining_gates))

        # ── Select target waypoint
        if waypoints:
            wp = planner.next_position_target(waypoints, state.pos_ned, lookahead_m=3.0)
        else:
            # No gates / no plan: hover at current position
            wp = WaypointNED(
                pos=state.pos_ned.copy(),
                vel=np.zeros(3),
                yaw=state.yaw,
                time=elapsed,
            )

        # ── Send command
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

        # ── Loop timing
        t_used = time.perf_counter() - t_loop
        sleep_t = loop_dt - t_used
        if sleep_t > 0:
            time.sleep(sleep_t)

    logger.info("Shutting down")
    client.send_hover()
    time.sleep(0.2)
    client.stop()
    vision.stop()
    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args()))
