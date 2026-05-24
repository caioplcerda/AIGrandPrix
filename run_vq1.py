"""AI Grand Prix VQ1 entry point — connects to DCL simulator via MAVLink2/UDP.

Usage (against mock sim):
    python run_vq1.py --host localhost --mavlink_port 14550 --vision_port 5600

Usage (against DCL sim when released):
    python run_vq1.py --host <dcl_sim_ip> --mavlink_port 14550 --vision_port 5600

With trained YOLOv8n weights:
    python run_vq1.py --host localhost --cnn_model runs/gate_v1/weights/best.pt

The only change needed when switching from mock to DCL: --host value.

Architecture:
    MAVLinkClient ──(telemetry)──► NEDStateEstimator ──► PathPlannerNED ──► MAVLinkClient
    VisionStreamReceiver ──(frames)──► GateDetector ──► gate position correction

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

# Camera intrinsics (VADR-TS-002)
_FX = 320.0
_FY = 320.0
_CX = 320.0
_CY = 180.0
_IMG_W = 640
_IMG_H = 360
_CAM_TILT_RAD = math.radians(20.0)  # camera tilts 20° up from body

# Gate geometry (VADR-TS-002)
_GATE_OUTER_M = 2.7

# Vision correction parameters
_VISION_CORRECTION_GAIN = 0.3   # how aggressively to correct gate position
_VISION_MAX_CORRECTION_M = 2.0  # clamp correction magnitude
_VISION_MIN_CONFIDENCE = 0.4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Grand Prix VQ1 autonomy stack")
    p.add_argument("--host", default="localhost", help="DCL simulator host")
    p.add_argument("--mavlink_port", type=int, default=14550)
    p.add_argument("--vision_port", type=int, default=5600)
    p.add_argument("--max_speed", type=float, default=10.0, help="max flight speed m/s")
    p.add_argument("--loop_hz", type=float, default=50.0, help="control loop rate Hz")
    p.add_argument("--cnn_model", default=None, help="Path to YOLOv8n .pt weights file")
    p.add_argument("--no_vision", action="store_true", help="Disable vision correction")
    p.add_argument(
        "--gates",
        nargs="+",
        default=None,
        help="Gate positions as 'north,east,down nx,ny,nz' pairs. "
             "If not provided, uses zero gate list (no planning, just hover).",
    )
    return p.parse_args()


def _pixel_to_bearing(cx_px: float, cy_px: float) -> tuple[float, float]:
    """Convert pixel coords to (yaw_offset_rad, pitch_offset_rad) from camera boresight.

    Returns angles relative to camera boresight (not body frame).
    Positive yaw = gate to the right, positive pitch = gate above boresight.
    """
    dx = cx_px - _CX
    dy = _CY - cy_px  # image y increases downward, pitch positive = up
    yaw_off = math.atan2(dx, _FX)
    pitch_off = math.atan2(dy, _FY)
    return yaw_off, pitch_off


def _bearing_to_gate_ned_offset(
    yaw_off: float,
    pitch_off: float,
    dist_est: float,
    drone_yaw: float,
) -> np.ndarray:
    """Convert camera bearing angles + distance → NED offset from drone to gate.

    Camera is tilted +20° upward from body X axis.
    Returns [north, east, down] offset vector.
    """
    # Camera boresight in body frame (tilted up)
    # pitch_off is in camera frame; add tilt offset
    pitch_total = pitch_off + _CAM_TILT_RAD  # positive = above horizontal

    # 3D direction in body frame (X=fwd, Y=right, Z=down)
    dx_body = math.cos(pitch_total) * math.cos(yaw_off)
    dy_body = math.cos(pitch_total) * math.sin(yaw_off)
    dz_body = -math.sin(pitch_total)  # up = negative Z body

    # Rotate body→NED using drone yaw (simplified: assume level flight)
    cy, sy = math.cos(drone_yaw), math.sin(drone_yaw)
    dx_ned = cy * dx_body - sy * dy_body
    dy_ned = sy * dx_body + cy * dy_body
    dz_ned = dz_body

    # Scale to distance
    mag = math.sqrt(dx_ned**2 + dy_ned**2 + dz_ned**2)
    if mag < 1e-6:
        return np.zeros(3)
    return np.array([dx_ned, dy_ned, dz_ned]) * (dist_est / mag)


def run(args: argparse.Namespace) -> int:
    from aigrandprix.comms.mavlink_client import MAVLinkClient, PositionTargetNED
    from aigrandprix.comms.vision_stream import VisionStreamReceiver
    from aigrandprix.state.state_estimator import NEDStateEstimator
    from aigrandprix.planning.path_planner_ned import GateNED, PathPlannerNED, WaypointNED
    from aigrandprix.perception.gate_detector import GateDetector

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

    # ── Gate detector (YOLO if weights given, else HSV fallback)
    # Vision position correction only active with CNN (HSV too noisy for correction).
    # HSV still used for detection confidence but NOT for gate position correction.
    vision_enabled = not args.no_vision
    use_vision_correction = False
    if vision_enabled:
        detector_config: dict = {}
        if args.cnn_model:
            detector_config["cnn"] = {
                "backend": "yolo",
                "model_path": args.cnn_model,
                "confidence_threshold": 0.4,
            }
            detector = GateDetector(method="cnn", config=detector_config)
            use_vision_correction = True
            logger.info("Vision: YOLOv8n from %s (with position correction)", args.cnn_model)
        else:
            detector = GateDetector(method="color", config=detector_config)
            logger.info("Vision: HSV color fallback (detection only, no position correction)")
    else:
        detector = None
        logger.info("Vision: disabled")

    # ── State estimator (NED, no GPS)
    estimator = NEDStateEstimator()

    # ── Path planner (NED)
    planner = PathPlannerNED(max_speed=args.max_speed)

    # ── Parse gates from args (or empty list)
    gates: list[GateNED] = []
    if args.gates:
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

    # ── Control loop state
    loop_dt = 1.0 / args.loop_hz
    next_gate_idx = 0
    waypoints: list[WaypointNED] = []
    last_replan = 0.0
    last_vision_frame_id = -1
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

        # ── Vision: gate position correction (CNN only — HSV too noisy)
        if use_vision_correction and detector is not None and next_gate_idx < len(gates):
            frame = vision.latest_frame
            if frame is not None:
                try:
                    detections = detector.detect(frame)
                    if detections:
                        best = detections[0]  # closest detected gate
                        if best.confidence >= _VISION_MIN_CONFIDENCE:
                            # Convert pixel detection → NED bearing
                            yaw_off, pitch_off = _pixel_to_bearing(
                                best.center_px[0], best.center_px[1]
                            )
                            offset_ned = _bearing_to_gate_ned_offset(
                                yaw_off, pitch_off, best.distance, state.yaw
                            )
                            # Predicted gate position from vision
                            predicted_gate_ned = state.pos_ned + offset_ned

                            # Soft correction: blend toward vision estimate
                            current_gate = gates[next_gate_idx]
                            correction = predicted_gate_ned - current_gate.position
                            corr_mag = float(np.linalg.norm(correction))
                            if corr_mag > 0.1 and corr_mag < _VISION_MAX_CORRECTION_M:
                                correction_scaled = correction * _VISION_CORRECTION_GAIN
                                gates[next_gate_idx] = GateNED(
                                    position=current_gate.position + correction_scaled,
                                    normal=current_gate.normal,
                                    gate_id=current_gate.gate_id,
                                    inner_half=current_gate.inner_half,
                                )
                                waypoints = []  # force replan with corrected gate
                                logger.debug(
                                    "Vision correction gate %d: Δ=%.2fm conf=%.2f",
                                    next_gate_idx, corr_mag, best.confidence,
                                )
                except Exception as exc:
                    logger.debug("Vision detection error: %s", exc)

        # ── Gate pass check
        remaining_gates = gates[next_gate_idx:]
        if remaining_gates:
            next_gate = remaining_gates[0]
            diff = next_gate.position - state.pos_ned
            dist_to_gate = float(np.linalg.norm(diff))
            # Plane-crossing: dot product with normal flips sign when drone crosses gate plane.
            # Guard dist<5m prevents false trigger from drift at long range.
            past_gate = (float(np.dot(diff, next_gate.normal)) < 0) and (dist_to_gate < 5.0)
            if dist_to_gate < GATE_PASS_RADIUS_M or past_gate:
                logger.info(
                    "Gate %d detected as passed (dist=%.2fm) — advancing",
                    next_gate.gate_id, dist_to_gate,
                )
                next_gate_idx += 1
                waypoints = []  # force replan
                estimator.reset_position(next_gate.position)

        # ── Replan trajectory
        remaining_gates = gates[next_gate_idx:]
        if remaining_gates and (not waypoints or elapsed - last_replan > REPLAN_INTERVAL_S):
            waypoints = planner.plan(remaining_gates, state.pos_ned, state.vel_ned)
            last_replan = elapsed
            logger.debug("Replanned: %d waypoints for %d gates", len(waypoints), len(remaining_gates))

        # ── Select target waypoint
        if waypoints:
            wp = planner.next_position_target(waypoints, state.pos_ned, lookahead_m=5.0)
        else:
            wp = WaypointNED(
                pos=state.pos_ned.copy(),
                vel=np.zeros(3),
                yaw=state.yaw,
                time=elapsed,
            )

        if wp is None:
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
