"""DCL Mock Simulator (VADR-TS-002 compliant UDP server).

Emulates the DCL simulator interface so the autonomy stack can be developed and
tested without the official Windows binary.

Architecture:
  - Single-threaded main loop at 120 Hz
  - MAVLink2 UDP on configurable port (default 14550)
  - Vision stream UDP on port 5600, sent to client IP once connected
  - Physics: Physics6DOF inner controller

Gate pass detection:
  - A gate is "passed" when drone crosses the gate plane within inner opening bounds.
  - Gates are passed in order (next gate only after current is done).

Usage:
    gates = [
        MockGate(center_ned=np.array([10.0, 0.0, -2.0]), normal_ned=np.array([1.0, 0.0, 0.0])),
        MockGate(center_ned=np.array([20.0, 5.0, -3.0]), normal_ned=np.array([1.0, 0.0, 0.0])),
    ]
    server = DCLMockServer(gates=gates, mavlink_port=14550, vision_port=5600)
    server.run()  # blocking
"""

from __future__ import annotations

import logging
import math
import socket
import struct
import time
from dataclasses import dataclass, field

import numpy as np

from aigrandprix.mock_sim.physics_6dof import (
    ControlCommand,
    Physics6DOF,
    PhysicsState,
    _DT as PHYSICS_DT,
)
from aigrandprix.mock_sim.gate_renderer import GateVisual, render_jpeg

logger = logging.getLogger(__name__)

_VISION_CHUNK_SIZE = 60000  # bytes per UDP packet payload
_VISION_HEADER_FMT = "<IHHIIq"  # frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns
_VISION_HEADER_SIZE = struct.calcsize(_VISION_HEADER_FMT)  # 24

_HEARTBEAT_INTERVAL = 1.0   # seconds
_ATTITUDE_INTERVAL = PHYSICS_DT  # 120 Hz
_VISION_INTERVAL = 1.0 / 30.0   # 30 Hz


@dataclass
class MockGate:
    """Gate definition for mock simulator."""

    center_ned: np.ndarray                # gate center in NED (m)
    normal_ned: np.ndarray                # unit normal pointing toward approach side
    gate_id: int = 0
    inner_half: float = 0.75             # inner opening half-width = 1.5m / 2

    def __post_init__(self) -> None:
        n = np.linalg.norm(self.normal_ned)
        if n > 1e-6:
            self.normal_ned = self.normal_ned / n

    def is_passed(self, old_pos: np.ndarray, new_pos: np.ndarray) -> bool:
        """Check if drone crossed the gate plane between old and new positions."""
        c = self.center_ned
        n = self.normal_ned
        # Signed distance from gate plane
        d_old = float(np.dot(old_pos - c, n))
        d_new = float(np.dot(new_pos - c, n))
        # Crossed plane (sign change)?
        if d_old * d_new > 0:
            return False
        # Interpolate crossing point
        if abs(d_old - d_new) < 1e-9:
            return False
        t = d_old / (d_old - d_new)
        cross = old_pos + t * (new_pos - old_pos)
        # Check within inner opening bounds
        diff = cross - c
        # Gate up = NED -Z, gate right = cross(normal, up)
        up_ned = np.array([0.0, 0.0, -1.0])
        right_ned = np.cross(n, up_ned)
        rn = np.linalg.norm(right_ned)
        if rn < 1e-6:
            right_ned = np.array([0.0, 1.0, 0.0])
        else:
            right_ned /= rn
        lat = abs(float(np.dot(diff, right_ned)))
        vert = abs(float(np.dot(diff, up_ned)))
        return lat <= self.inner_half and vert <= self.inner_half


class DCLMockServer:
    """Mock DCL simulator server (VADR-TS-002 interface).

    Call run() to start the blocking event loop.
    """

    def __init__(
        self,
        gates: list[MockGate],
        mavlink_port: int = 14550,
        vision_port: int = 5600,
        host: str = "0.0.0.0",
        sysid: int = 1,
        compid: int = 1,
    ) -> None:
        self._gates = gates
        self._mavlink_port = mavlink_port
        self._vision_port = vision_port
        self._host = host
        self._sysid = sysid
        self._compid = compid

        self._physics = Physics6DOF()
        self._cmd = ControlCommand()
        self._client_addr: tuple[str, int] | None = None
        self._client_vision_port = 5600  # client receives vision on this port

        self._next_gate_idx = 0
        self._gates_passed: list[int] = []
        self._start_time = time.time()
        self._sim_time_ns: int = 0

        self._mav_sock: socket.socket | None = None
        self._vision_sock: socket.socket | None = None
        self._mav: object | None = None

        self._frame_id = 0
        self._last_heartbeat = 0.0
        self._last_vision = 0.0
        self._tick = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, max_duration_s: float = 480.0) -> dict:
        """Run the mock server. Blocks until max_duration_s or all gates passed.

        Returns summary dict: {gates_passed, total_time_s, completed}.
        """
        self._setup_sockets()

        logger.info(
            "DCL Mock Server running on MAVLink port %d, vision port %d",
            self._mavlink_port, self._vision_port,
        )
        logger.info("Waiting for client connection (send HEARTBEAT to arm)...")

        t_start = time.perf_counter()
        loop_dt = PHYSICS_DT

        while True:
            t_loop_start = time.perf_counter()
            elapsed = t_loop_start - t_start

            if elapsed > max_duration_s:
                logger.info("Max duration reached (%.1f s)", elapsed)
                break

            # Check race complete
            if self._next_gate_idx >= len(self._gates):
                logger.info(
                    "All %d gates passed! Time: %.2f s", len(self._gates), elapsed
                )
                break

            # Receive MAVLink messages (non-blocking)
            self._recv_mavlink()

            # Physics step
            old_pos = self._physics.state.pos.copy()
            state = self._physics.step()
            new_pos = state.pos

            self._sim_time_ns = int(elapsed * 1e9)
            self._tick += 1

            # Gate pass detection
            if self._next_gate_idx < len(self._gates):
                gate = self._gates[self._next_gate_idx]
                if gate.is_passed(old_pos, new_pos):
                    self._gates_passed.append(gate.gate_id)
                    self._next_gate_idx += 1
                    logger.info(
                        "Gate %d passed (%.2f s) — %d/%d done",
                        gate.gate_id, elapsed, self._next_gate_idx, len(self._gates),
                    )

            # Send MAVLink telemetry at 120 Hz
            self._send_attitude(state)
            self._send_highres_imu(state)

            # Send HEARTBEAT at 1 Hz
            if elapsed - self._last_heartbeat >= _HEARTBEAT_INTERVAL:
                self._send_heartbeat()
                self._last_heartbeat = elapsed

            # Send TIMESYNC occasionally
            if self._tick % 600 == 0:
                self._send_timesync()

            # Send vision frame at 30 Hz
            if elapsed - self._last_vision >= _VISION_INTERVAL:
                self._send_vision_frame(state)
                self._last_vision = elapsed

            # Precise 120 Hz timing
            t_used = time.perf_counter() - t_loop_start
            sleep_t = loop_dt - t_used
            if sleep_t > 0:
                time.sleep(sleep_t)

        self._cleanup()
        total_time = time.perf_counter() - t_start
        return {
            "gates_passed": len(self._gates_passed),
            "total_gates": len(self._gates),
            "total_time_s": total_time,
            "completed": self._next_gate_idx >= len(self._gates),
        }

    # ------------------------------------------------------------------
    # Setup / cleanup
    # ------------------------------------------------------------------

    def _setup_sockets(self) -> None:
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise RuntimeError("pymavlink not installed: pip install pymavlink>=2.4") from exc

        # MAVLink socket (listen for incoming, reply to sender)
        self._mav_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._mav_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._mav_sock.bind((self._host, self._mavlink_port))
        self._mav_sock.setblocking(False)

        # MAVLink dialect object for building/parsing packets
        self._mav = mavutil.mavlink.MAVLink(
            None,
            srcSystem=self._sysid,
            srcComponent=self._compid,
        )
        self._mav.signing.secret_key = None  # no signing

        # Vision stream socket (send only)
        self._vision_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._vision_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)

    def _cleanup(self) -> None:
        if self._mav_sock:
            self._mav_sock.close()
        if self._vision_sock:
            self._vision_sock.close()

    # ------------------------------------------------------------------
    # MAVLink receiving
    # ------------------------------------------------------------------

    def _recv_mavlink(self) -> None:
        if self._mav_sock is None:
            return
        try:
            while True:
                data, addr = self._mav_sock.recvfrom(512)
                if self._client_addr is None:
                    self._client_addr = addr
                    logger.info("Client connected from %s:%d", addr[0], addr[1])
                try:
                    msgs = self._mav.parse_buffer(data)  # type: ignore[attr-defined]
                    if msgs:
                        for msg in msgs:
                            self._handle_msg(msg, addr)
                except Exception:
                    pass
        except BlockingIOError:
            pass
        except Exception as exc:
            logger.debug("recv error: %s", exc)

    def _handle_msg(self, msg: object, addr: tuple) -> None:
        msg_id = msg.get_msgId()  # type: ignore[attr-defined]
        try:
            from pymavlink.dialects.v20 import common as mav_common
            SET_POS_TARGET_ID = mav_common.MAVLINK_MSG_ID_SET_POSITION_TARGET_LOCAL_NED
            SET_ATT_TARGET_ID = mav_common.MAVLINK_MSG_ID_SET_ATTITUDE_TARGET
            HEARTBEAT_ID = 0
        except ImportError:
            SET_POS_TARGET_ID = 84
            SET_ATT_TARGET_ID = 82
            HEARTBEAT_ID = 0

        if msg_id == HEARTBEAT_ID:
            logger.debug("Client heartbeat received")
        elif msg_id == SET_POS_TARGET_ID:
            self._on_position_target(msg)
        elif msg_id == SET_ATT_TARGET_ID:
            self._on_attitude_target(msg)

    def _on_position_target(self, msg: object) -> None:
        x = float(getattr(msg, "x", 0.0))
        y = float(getattr(msg, "y", 0.0))
        z = float(getattr(msg, "z", -2.0))
        vx = float(getattr(msg, "vx", 0.0))
        vy = float(getattr(msg, "vy", 0.0))
        vz = float(getattr(msg, "vz", 0.0))
        yaw = float(getattr(msg, "yaw", 0.0))
        type_mask = int(getattr(msg, "type_mask", 0))

        # If position bits are set (not ignored), use pos command
        if not (type_mask & 0x7):  # bits 0-2 clear = use position
            self._cmd = ControlCommand(
                pos=np.array([x, y, z]),
                vel=np.array([vx, vy, vz]),
                yaw=yaw,
            )
        else:
            # Velocity-only command
            cur_pos = self._physics.state.pos
            self._cmd = ControlCommand(
                pos=cur_pos.copy(),
                vel=np.array([vx, vy, vz]),
                yaw=yaw,
            )
        self._physics.set_command(self._cmd)

    def _on_attitude_target(self, msg: object) -> None:
        # Minimal support: extract thrust and convert to hover/move
        # Full attitude control would need much more. For now, just hover.
        logger.debug("SET_ATTITUDE_TARGET received (minimal support in mock)")

    # ------------------------------------------------------------------
    # MAVLink sending
    # ------------------------------------------------------------------

    def _send_raw(self, buf: bytes) -> None:
        if self._mav_sock is None or self._client_addr is None:
            return
        try:
            self._mav_sock.sendto(buf, self._client_addr)
        except Exception as exc:
            logger.debug("send error: %s", exc)

    def _send_heartbeat(self) -> None:
        try:
            from pymavlink import mavutil
            buf = self._mav.heartbeat_encode(  # type: ignore[attr-defined]
                type=mavutil.mavlink.MAV_TYPE_GENERIC,
                autopilot=mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
                base_mode=mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED,
                custom_mode=0,
                system_status=mavutil.mavlink.MAV_STATE_ACTIVE,
            ).pack(self._mav)
            self._send_raw(buf)
        except Exception as exc:
            logger.debug("heartbeat send error: %s", exc)

    def _send_attitude(self, state: PhysicsState) -> None:
        try:
            t_ms = int(state.time_s * 1000) & 0xFFFFFFFF
            buf = self._mav.attitude_encode(  # type: ignore[attr-defined]
                time_boot_ms=t_ms,
                roll=float(state.roll),
                pitch=float(state.pitch),
                yaw=float(state.yaw),
                rollspeed=float(state.roll_rate),
                pitchspeed=float(state.pitch_rate),
                yawspeed=float(state.yaw_rate),
            ).pack(self._mav)
            self._send_raw(buf)
        except Exception as exc:
            logger.debug("attitude send error: %s", exc)

    def _send_highres_imu(self, state: PhysicsState) -> None:
        try:
            accel = self._physics.body_frame_accel()
            t_us = int(state.time_s * 1e6) & 0xFFFFFFFFFFFFFFFF
            buf = self._mav.highres_imu_encode(  # type: ignore[attr-defined]
                time_usec=t_us,
                xacc=float(accel[0]),
                yacc=float(accel[1]),
                zacc=float(accel[2]),
                xgyro=float(state.roll_rate),
                ygyro=float(state.pitch_rate),
                zgyro=float(state.yaw_rate),
                xmag=0.0, ymag=0.0, zmag=0.0,
                abs_pressure=101325.0,
                diff_pressure=0.0,
                pressure_alt=float(-state.pos[2]),
                temperature=25.0,
                fields_updated=0xFFF,
                id=0,
            ).pack(self._mav)
            self._send_raw(buf)
        except Exception as exc:
            logger.debug("highres_imu send error: %s", exc)

    def _send_timesync(self) -> None:
        try:
            ts1 = int(time.time() * 1e9)
            buf = self._mav.timesync_encode(  # type: ignore[attr-defined]
                tc1=0,
                ts1=ts1,
            ).pack(self._mav)
            self._send_raw(buf)
        except Exception as exc:
            logger.debug("timesync send error: %s", exc)

    # ------------------------------------------------------------------
    # Vision stream
    # ------------------------------------------------------------------

    def _send_vision_frame(self, state: PhysicsState) -> None:
        if self._vision_sock is None or self._client_addr is None:
            return

        gate_visuals = [
            GateVisual(
                center_ned=g.center_ned.copy(),
                normal_ned=g.normal_ned.copy(),
                gate_id=g.gate_id,
            )
            for g in self._gates[self._next_gate_idx: self._next_gate_idx + 3]
        ]

        try:
            jpeg_bytes = render_jpeg(
                state.pos, state.roll, state.pitch, state.yaw, gate_visuals
            )
        except Exception as exc:
            logger.debug("render_jpeg error: %s", exc)
            return

        jpeg_size = len(jpeg_bytes)
        total_chunks = max(1, math.ceil(jpeg_size / _VISION_CHUNK_SIZE))
        frame_id = self._frame_id & 0xFFFFFFFF
        self._frame_id += 1

        client_ip = self._client_addr[0]
        dest = (client_ip, self._client_vision_port)

        for chunk_id in range(total_chunks):
            start = chunk_id * _VISION_CHUNK_SIZE
            payload = jpeg_bytes[start: start + _VISION_CHUNK_SIZE]
            header = struct.pack(
                _VISION_HEADER_FMT,
                frame_id,
                chunk_id,
                total_chunks,
                jpeg_size,
                len(payload),
                self._sim_time_ns,
            )
            try:
                self._vision_sock.sendto(header + payload, dest)
            except Exception as exc:
                logger.debug("vision chunk send error: %s", exc)
                break


# ---------------------------------------------------------------------------
# Convenience builder for test courses
# ---------------------------------------------------------------------------

def build_straight_course(
    n_gates: int = 5,
    spacing_m: float = 10.0,
    altitude_ned: float = -2.5,
    start_north_m: float = 10.0,
) -> list[MockGate]:
    """Build a straight NED course: gates spaced along north axis."""
    return [
        MockGate(
            center_ned=np.array([start_north_m + i * spacing_m, 0.0, altitude_ned]),
            normal_ned=np.array([1.0, 0.0, 0.0]),  # facing north
            gate_id=i,
        )
        for i in range(n_gates)
    ]
