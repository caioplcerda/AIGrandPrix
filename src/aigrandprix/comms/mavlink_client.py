"""MAVLink2/UDP client for DCL simulator (VADR-TS-002).

Implements:
- HEARTBEAT keepalive (≥2 Hz, runs in background thread)
- ATTITUDE reception (quat via roll/pitch/yaw Euler)
- HIGHRES_IMU reception (body-frame accel + gyro)
- TIMESYNC response
- SET_POSITION_TARGET_LOCAL_NED sending (pos + vel + yaw, NED frame)

Transport: UDP. Protocol: MAVLink2. No GPS, no position feedback from sim.

Usage:
    client = MAVLinkClient("192.168.1.100", mavlink_port=14550)
    client.connect()
    client.start()
    # control loop
    client.send_position_target(PositionTargetNED(x=5.0, y=0.0, z=-2.0, vx=2.0, vy=0.0, vz=0.0, yaw=0.0))
    state = client.telemetry  # latest MAVLink telemetry snapshot
    client.stop()
"""

from __future__ import annotations

import logging
import math
import struct
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# type_mask for SET_POSITION_TARGET_LOCAL_NED:
# use x,y,z (bits 0-2=0) + vx,vy,vz (bits 3-5=0) + yaw (bit 10=0)
# ignore ax,ay,az (bits 6-8=1) + force (bit 9=1) + yaw_rate (bit 11=1)
_POSITION_VELOCITY_YAW_MASK = (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 11)  # 0xBC0


@dataclass
class PositionTargetNED:
    """Desired drone state in NED frame for SET_POSITION_TARGET_LOCAL_NED."""

    x: float       # north (m)
    y: float       # east (m)
    z: float       # down (m) — altitude = -z, so z=-2 means 2m above ground
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0  # rad, NED convention (0=north, pi/2=east)


@dataclass
class TelemetryState:
    """Latest telemetry snapshot from simulator."""

    # From ATTITUDE message
    roll: float = 0.0      # rad
    pitch: float = 0.0     # rad
    yaw: float = 0.0       # rad
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0
    attitude_timestamp_us: int = 0

    # From HIGHRES_IMU message
    accel_x: float = 0.0   # body frame (m/s²), includes gravity reaction
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0    # body frame (rad/s)
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    imu_timestamp_us: int = 0

    # Connection quality
    last_heartbeat_time: float = field(default_factory=time.time)
    connected: bool = False


class MAVLinkClient:
    """Thread-safe MAVLink2/UDP client connecting to DCL simulator.

    The client uses pymavlink under the hood. A background thread continuously
    receives messages and updates the telemetry snapshot. Another background
    thread sends HEARTBEAT at the configured rate.

    The control loop calls send_position_target() at ≤100 Hz.
    """

    def __init__(
        self,
        host: str,
        mavlink_port: int = 14550,
        heartbeat_rate_hz: float = 2.0,
        source_system: int = 255,
        source_component: int = 0,
    ) -> None:
        self._host = host
        self._port = mavlink_port
        self._heartbeat_interval = 1.0 / max(heartbeat_rate_hz, 2.0)
        self._source_system = source_system
        self._source_component = source_component

        self._mav: object | None = None
        self._telemetry = TelemetryState()
        self._lock = threading.Lock()
        self._running = False

        self._recv_thread: threading.Thread | None = None
        self._hb_thread: threading.Thread | None = None

        self._timesync_offset_ns: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish MAVLink2 UDP connection to simulator."""
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise RuntimeError(
                "pymavlink not installed. Run: pip install pymavlink>=2.4"
            ) from exc

        address = f"udpout:{self._host}:{self._port}"
        self._mav = mavutil.mavlink_connection(
            address,
            source_system=self._source_system,
            source_component=self._source_component,
            dialect="common",
        )
        # Force MAVLink2
        self._mav.mav.dialect_version = 2  # type: ignore[attr-defined]
        logger.info("MAVLink client connecting to %s:%d", self._host, self._port)

    def start(self) -> None:
        """Start background receive + heartbeat threads."""
        if self._mav is None:
            raise RuntimeError("Call connect() before start()")
        self._running = True

        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True, name="mav-recv")
        self._recv_thread.start()

        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name="mav-hb")
        self._hb_thread.start()

        logger.info("MAVLink client started")

    def stop(self) -> None:
        """Stop background threads and close connection."""
        self._running = False
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2.0)
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=2.0)
        if self._mav is not None:
            try:
                self._mav.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        logger.info("MAVLink client stopped")

    # ------------------------------------------------------------------
    # Telemetry (thread-safe read)
    # ------------------------------------------------------------------

    @property
    def telemetry(self) -> TelemetryState:
        """Return a snapshot copy of the latest telemetry."""
        with self._lock:
            return TelemetryState(
                roll=self._telemetry.roll,
                pitch=self._telemetry.pitch,
                yaw=self._telemetry.yaw,
                roll_rate=self._telemetry.roll_rate,
                pitch_rate=self._telemetry.pitch_rate,
                yaw_rate=self._telemetry.yaw_rate,
                attitude_timestamp_us=self._telemetry.attitude_timestamp_us,
                accel_x=self._telemetry.accel_x,
                accel_y=self._telemetry.accel_y,
                accel_z=self._telemetry.accel_z,
                gyro_x=self._telemetry.gyro_x,
                gyro_y=self._telemetry.gyro_y,
                gyro_z=self._telemetry.gyro_z,
                imu_timestamp_us=self._telemetry.imu_timestamp_us,
                last_heartbeat_time=self._telemetry.last_heartbeat_time,
                connected=self._telemetry.connected,
            )

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._telemetry.connected

    # ------------------------------------------------------------------
    # Control commands
    # ------------------------------------------------------------------

    def send_position_target(self, target: PositionTargetNED) -> None:
        """Send SET_POSITION_TARGET_LOCAL_NED (pos + vel + yaw, NED frame).

        Must be called at ≤100 Hz as per VADR-TS-002 §4.4.
        """
        if self._mav is None or not self._running:
            return
        try:
            time_boot_ms = int((time.time() * 1000) % (2**32))
            self._mav.mav.set_position_target_local_ned_send(  # type: ignore[attr-defined]
                time_boot_ms=time_boot_ms,
                target_system=1,
                target_component=1,
                coordinate_frame=1,  # MAV_FRAME_LOCAL_NED
                type_mask=_POSITION_VELOCITY_YAW_MASK,
                x=float(target.x),
                y=float(target.y),
                z=float(target.z),
                vx=float(target.vx),
                vy=float(target.vy),
                vz=float(target.vz),
                afx=0.0,
                afy=0.0,
                afz=0.0,
                yaw=float(target.yaw),
                yaw_rate=0.0,
            )
        except Exception as exc:
            logger.debug("send_position_target failed: %s", exc)

    def send_hover(self) -> None:
        """Send hover command at current position (vel=0). Safety fallback."""
        telem = self.telemetry
        # Without position feedback from sim, hover = zero velocity command
        hover = PositionTargetNED(x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, yaw=telem.yaw)
        # Override type_mask to use velocity only (not position, since we don't know it)
        if self._mav is None or not self._running:
            return
        try:
            vel_only_mask = 0b111111000111  # use vx,vy,vz; ignore pos+accel+yaw_rate+yaw
            time_boot_ms = int((time.time() * 1000) % (2**32))
            self._mav.mav.set_position_target_local_ned_send(  # type: ignore[attr-defined]
                time_boot_ms=time_boot_ms,
                target_system=1,
                target_component=1,
                coordinate_frame=1,
                type_mask=vel_only_mask,
                x=0.0, y=0.0, z=0.0,
                vx=0.0, vy=0.0, vz=0.0,
                afx=0.0, afy=0.0, afz=0.0,
                yaw=0.0, yaw_rate=0.0,
            )
        except Exception as exc:
            logger.debug("send_hover failed: %s", exc)

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        """Send HEARTBEAT at configured rate (≥2 Hz)."""
        try:
            from pymavlink import mavutil
        except ImportError:
            return

        while self._running:
            try:
                self._mav.mav.heartbeat_send(  # type: ignore[attr-defined]
                    type=mavutil.mavlink.MAV_TYPE_GCS,
                    autopilot=mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    base_mode=0,
                    custom_mode=0,
                    system_status=mavutil.mavlink.MAV_STATE_ACTIVE,
                )
            except Exception as exc:
                logger.debug("heartbeat send error: %s", exc)
            time.sleep(self._heartbeat_interval)

    def _recv_loop(self) -> None:
        """Receive and dispatch MAVLink messages from simulator."""
        while self._running:
            try:
                msg = self._mav.recv_match(blocking=True, timeout=0.05)  # type: ignore[attr-defined]
                if msg is None:
                    continue
                msg_type = msg.get_type()
                if msg_type == "HEARTBEAT":
                    self._on_heartbeat(msg)
                elif msg_type == "ATTITUDE":
                    self._on_attitude(msg)
                elif msg_type == "HIGHRES_IMU":
                    self._on_highres_imu(msg)
                elif msg_type == "TIMESYNC":
                    self._on_timesync(msg)
            except Exception as exc:
                logger.debug("recv_loop error: %s", exc)

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def _on_heartbeat(self, msg: object) -> None:
        with self._lock:
            self._telemetry.last_heartbeat_time = time.time()
            self._telemetry.connected = True

    def _on_attitude(self, msg: object) -> None:
        with self._lock:
            self._telemetry.roll = float(msg.roll)  # type: ignore[attr-defined]
            self._telemetry.pitch = float(msg.pitch)  # type: ignore[attr-defined]
            self._telemetry.yaw = float(msg.yaw)  # type: ignore[attr-defined]
            self._telemetry.roll_rate = float(msg.rollspeed)  # type: ignore[attr-defined]
            self._telemetry.pitch_rate = float(msg.pitchspeed)  # type: ignore[attr-defined]
            self._telemetry.yaw_rate = float(msg.yawspeed)  # type: ignore[attr-defined]
            self._telemetry.attitude_timestamp_us = int(msg.time_boot_ms) * 1000  # type: ignore[attr-defined]

    def _on_highres_imu(self, msg: object) -> None:
        with self._lock:
            self._telemetry.accel_x = float(msg.xacc)   # type: ignore[attr-defined]
            self._telemetry.accel_y = float(msg.yacc)   # type: ignore[attr-defined]
            self._telemetry.accel_z = float(msg.zacc)   # type: ignore[attr-defined]
            self._telemetry.gyro_x = float(msg.xgyro)   # type: ignore[attr-defined]
            self._telemetry.gyro_y = float(msg.ygyro)   # type: ignore[attr-defined]
            self._telemetry.gyro_z = float(msg.zgyro)   # type: ignore[attr-defined]
            self._telemetry.imu_timestamp_us = int(msg.time_usec)  # type: ignore[attr-defined]

    def _on_timesync(self, msg: object) -> None:
        """Respond to TIMESYNC to maintain clock sync with simulator."""
        if self._mav is None:
            return
        try:
            tc1 = int(time.time() * 1e9)  # current time in nanoseconds
            self._mav.mav.timesync_send(tc1=tc1, ts1=int(msg.ts1))  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("timesync response error: %s", exc)
