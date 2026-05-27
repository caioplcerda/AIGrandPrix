"""Generate drone POV video — real autonomy stack flies diagonal_normals course.

Records actual physics trajectory at 120Hz, renders 640×360 POV at 30fps.
Output: ~/Downloads/drone_pov.mp4

Usage:
    python scripts/gen_pov_video.py
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Course: diagonal_normals (hardest — U-turn loop with diagonal gates) ─────
GATE_DEFS = [
    (np.array([12.0,  0.0, -2.5]), np.array([1.0, 0.0, 0.0])),
    (np.array([24.0,  6.0, -2.5]), np.array([0.707, 0.707, 0.0])),
    (np.array([30.0, 12.0, -2.5]), np.array([0.0, 1.0, 0.0])),
    (np.array([24.0, 18.0, -2.5]), np.array([-0.707, 0.707, 0.0])),
    (np.array([12.0, 18.0, -2.5]), np.array([-1.0, 0.0, 0.0])),
]

MAV_PORT = 14575
VIS_PORT = 15715
MAX_SPEED = 15.0
FPS = 30
OUT_PATH = Path.home() / "Downloads" / "drone_pov.mp4"


@dataclass
class Frame:
    pos: np.ndarray
    roll: float
    pitch: float
    yaw: float
    speed: float
    gate_idx: int
    elapsed: float


def _gate_crossed(gate_pos, gate_normal, old_pos, new_pos, inner_half=0.75):
    n, c = gate_normal, gate_pos
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


def run_sim_and_capture() -> list[Frame]:
    """Fly the diagonal_normals course and return physics frames."""
    from aigrandprix.mock_sim.dcl_mock_server import DCLMockServer, MockGate
    from aigrandprix.mock_sim.physics_6dof import Physics6DOF
    from aigrandprix.comms.mavlink_client import MAVLinkClient, PositionTargetNED
    from aigrandprix.state.state_estimator import NEDStateEstimator
    from aigrandprix.planning.path_planner_ned import GateNED, PathPlannerNED, WaypointNED

    n_gates = len(GATE_DEFS)
    gates_mock = [
        MockGate(center_ned=pos.copy(), normal_ned=norm.copy(), gate_id=i)
        for i, (pos, norm) in enumerate(GATE_DEFS)
    ]
    gate_neds = [
        GateNED(position=pos.copy(), normal=norm.copy(), gate_id=i)
        for i, (pos, norm) in enumerate(GATE_DEFS)
    ]

    frames: list[Frame] = []
    server_result: dict = {}
    stop_event = threading.Event()

    def run_server():
        server = DCLMockServer(
            gates=gates_mock,
            mavlink_port=MAV_PORT,
            vision_port=VIS_PORT,
            host="127.0.0.1",
        )
        r = server.run(max_duration_s=90.0)
        server_result.update(r)
        stop_event.set()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(0.2)

    client = MAVLinkClient(host="127.0.0.1", mavlink_port=MAV_PORT, heartbeat_rate_hz=5.0)
    client.connect()
    client.start()

    t0 = time.time()
    while not client.connected and time.time() - t0 < 10.0:
        time.sleep(0.05)

    if not client.connected:
        client.stop()
        raise RuntimeError("No heartbeat in 10s")

    estimator = NEDStateEstimator()
    planner = PathPlannerNED(max_speed=MAX_SPEED, approach_distance=6.0, exit_distance=1.0)

    loop_dt = 1.0 / 50.0
    next_idx = 0
    waypoints: list[WaypointNED] = []
    last_replan = -0.25
    prev_pos: np.ndarray | None = None
    t_start = time.time()

    print("Flying diagonal_normals course...")

    while True:
        t_loop = time.perf_counter()
        elapsed = time.time() - t_start

        if elapsed > 90.0 or stop_event.is_set() or next_idx >= n_gates:
            break

        telem = client.telemetry
        if not telem.connected:
            time.sleep(0.05)
            continue

        state = estimator.update(telem)
        curr_pos = state.pos_ned

        if prev_pos is not None and next_idx < n_gates:
            g = gate_neds[next_idx]
            if _gate_crossed(g.position, g.normal, prev_pos, curr_pos):
                estimator.reset_position(g.position)
                next_idx += 1
                waypoints = []

        prev_pos = curr_pos.copy()

        # Capture frame at ~30Hz (every ~1.67 control loops at 50Hz)
        if len(frames) == 0 or elapsed - frames[-1].elapsed >= 1.0 / FPS - 0.001:
            speed = float(np.linalg.norm(state.vel_ned))
            # Derive roll from lateral acceleration (cosmetic lean)
            lat_accel = float(state.vel_ned[1]) * 0.05
            roll = float(np.clip(lat_accel, -0.25, 0.25))
            # Derive pitch from speed and vertical velocity
            vert_vel = float(state.vel_ned[2])  # NED down positive
            pitch = float(np.clip(-vert_vel * 0.04, -0.2, 0.2))
            frames.append(Frame(
                pos=curr_pos.copy(),
                roll=roll,
                pitch=pitch,
                yaw=state.yaw,
                speed=speed,
                gate_idx=next_idx,
                elapsed=elapsed,
            ))

        remaining = gate_neds[next_idx:]
        if remaining and (not waypoints or elapsed - last_replan > 0.25):
            waypoints = planner.plan(remaining[:1], state.pos_ned, state.vel_ned)
            last_replan = elapsed

        if waypoints:
            wp = planner.next_position_target(waypoints, state.pos_ned, lookahead_m=5.0)
        else:
            wp = WaypointNED(pos=state.pos_ned.copy(), vel=np.zeros(3), yaw=state.yaw, time=elapsed)
        if wp is None:
            wp = WaypointNED(pos=state.pos_ned.copy(), vel=np.zeros(3), yaw=state.yaw, time=elapsed)

        client.send_position_target(PositionTargetNED(
            x=float(wp.pos[0]), y=float(wp.pos[1]), z=float(wp.pos[2]),
            vx=float(wp.vel[0]), vy=float(wp.vel[1]), vz=float(wp.vel[2]),
            yaw=float(wp.yaw),
        ))

        used = time.perf_counter() - t_loop
        if (s := loop_dt - used) > 0:
            time.sleep(s)

    client.stop()
    server_thread.join(timeout=15.0)

    gp = server_result.get("gates_passed", 0)
    tt = server_result.get("total_time_s", 0.0)
    print(f"Flight complete: {gp}/{n_gates} gates  {tt:.2f}s  ({len(frames)} frames captured)")
    if gp < n_gates:
        raise RuntimeError(f"Only {gp}/{n_gates} gates passed — aborting video")
    return frames


def render_video(frames: list[Frame]) -> None:
    """Render captured frames to POV video."""
    from aigrandprix.mock_sim.gate_renderer import GateVisual, render_frame

    gates_visual = [
        GateVisual(
            center_ned=pos.copy(),
            normal_ned=norm.copy(),
            gate_id=i,
            color_bgr=(180, 30, 20),  # DCL blue
        )
        for i, (pos, norm) in enumerate(GATE_DEFS)
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (640, 360))

    total = len(frames)
    print(f"Rendering {total} frames → {OUT_PATH}")

    for fi, f in enumerate(frames):
        if fi % 30 == 0:
            print(f"  {fi}/{total}  t={f.elapsed:.1f}s  spd={f.speed:.1f}m/s  gate={f.gate_idx}")

        img = render_frame(
            f.pos, f.roll, f.pitch, f.yaw,
            gates_visual,
            background_bgr=(10, 8, 6),
        )

        # ── Stadium atmosphere: vignette + subtle scan lines ─────────────
        # Vignette
        vig = np.zeros((360, 640), dtype=np.float32)
        cx_v, cy_v = 320, 180
        for r in range(360):
            for c in range(0, 640, 4):  # subsample for speed
                d = math.sqrt(((c - cx_v) / 320) ** 2 + ((r - cy_v) / 180) ** 2)
                vig[r, c:c+4] = max(0.0, 1.0 - 0.5 * d * d)
        img = (img.astype(np.float32) * vig[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)

        # ── HUD ─────────────────────────────────────────────────────────
        speed_kph = f.speed * 3.6
        altitude_m = -f.pos[2]

        # Speed bar (bottom left)
        bar_w = int(min(f.speed / MAX_SPEED, 1.0) * 150)
        cv2.rectangle(img, (10, 325), (160, 340), (40, 40, 40), -1)
        bar_color = (0, 200, 60) if f.speed < MAX_SPEED * 0.7 else (0, 120, 255)
        if bar_w > 0:
            cv2.rectangle(img, (10, 325), (10 + bar_w, 340), bar_color, -1)
        cv2.rectangle(img, (10, 325), (160, 340), (100, 100, 100), 1)

        # Text overlays
        def put(text, x, y, scale=0.55, color=(220, 220, 220), thickness=1):
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thickness, cv2.LINE_AA)

        put(f"{speed_kph:.0f} km/h", 12, 320, color=(255, 255, 255), scale=0.5)
        put(f"ALT {altitude_m:.1f}m", 12, 345, color=(160, 255, 160), scale=0.45)
        put(f"GATE {f.gate_idx}/{len(GATE_DEFS)}", 12, 22, scale=0.65,
            color=(255, 220, 80) if f.gate_idx < len(GATE_DEFS) else (80, 255, 80))
        put(f"{f.elapsed:.1f}s", 580, 22, scale=0.55, color=(200, 200, 200))

        # Gate passage flash: bright overlay at gate crossing
        gate_crossed_recently = False
        for pos, norm in GATE_DEFS:
            dist = float(np.linalg.norm(f.pos - pos))
            if dist < 1.5:
                gate_crossed_recently = True
        if gate_crossed_recently:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (640, 360), (0, 220, 255), -1)
            img = cv2.addWeighted(img, 0.85, overlay, 0.15, 0)

        # Diagonal course label
        put("DIAGONAL NORMALS COURSE", 170, 350, scale=0.42, color=(120, 120, 180))

        writer.write(img)

    writer.release()
    print(f"Done → {OUT_PATH}")


def main():
    frames = run_sim_and_capture()
    render_video(frames)


if __name__ == "__main__":
    main()
