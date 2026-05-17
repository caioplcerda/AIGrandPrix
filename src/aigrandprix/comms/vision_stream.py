"""UDP vision stream receiver for DCL simulator (VADR-TS-002 §4.6).

Protocol:
  - UDP port 5600 (default)
  - Each UDP packet: 24-byte little-endian header + JPEG data slice
  - Header fields: frame_id(4B), chunk_id(2B), total_chunks(2B),
                   jpeg_size(4B), payload_size(4B), sim_time_ns(8B)
  - Reassemble all chunks with same frame_id → decode JPEG → numpy BGR

Usage:
    recv = VisionStreamReceiver(port=5600)
    recv.start()
    frame = recv.latest_frame  # latest numpy BGR image or None
    recv.stop()
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_HEADER_FORMAT = "<IHHIIq"  # little-endian: u32 u16 u16 u32 u32 i64
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)  # must be 24


@dataclass
class _FrameBuffer:
    frame_id: int
    total_chunks: int
    jpeg_size: int
    sim_time_ns: int
    chunks: dict[int, bytes] = field(default_factory=dict)

    def is_complete(self) -> bool:
        return len(self.chunks) == self.total_chunks

    def assemble(self) -> bytes:
        return b"".join(self.chunks[i] for i in range(self.total_chunks))


class VisionStreamReceiver:
    """Receives and reassembles JPEG frames from the DCL vision stream.

    Runs a background thread that continuously reads UDP packets,
    reassembles JPEG frames from chunks, and decodes them to numpy arrays.
    The latest complete frame is available via `latest_frame`.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5600,
        recv_buf_size: int = 65536,
        max_incomplete_frames: int = 10,
    ) -> None:
        assert _HEADER_SIZE == 24, f"Header size mismatch: {_HEADER_SIZE}"
        self._host = host
        self._port = port
        self._recv_buf_size = recv_buf_size
        self._max_incomplete = max_incomplete_frames

        self._sock: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        self._incomplete: dict[int, _FrameBuffer] = {}
        self._latest_frame: np.ndarray | None = None
        self._latest_sim_time_ns: int = 0
        self._frame_count: int = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._host, self._port))
        self._sock.settimeout(0.1)
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True, name="vision-recv")
        self._thread.start()
        logger.info("Vision stream receiver started on %s:%d", self._host, self._port)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sock is not None:
            self._sock.close()
        logger.info("Vision stream receiver stopped")

    @property
    def latest_frame(self) -> np.ndarray | None:
        """Latest decoded BGR frame or None if no frame received yet."""
        with self._lock:
            return self._latest_frame

    @property
    def latest_sim_time_ns(self) -> int:
        """Simulator timestamp (nanoseconds) of the latest frame."""
        with self._lock:
            return self._latest_sim_time_ns

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(self._recv_buf_size)  # type: ignore[union-attr]
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < _HEADER_SIZE:
                continue

            try:
                frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = (
                    struct.unpack_from(_HEADER_FORMAT, data, 0)
                )
            except struct.error:
                continue

            payload = data[_HEADER_SIZE: _HEADER_SIZE + payload_size]

            if frame_id not in self._incomplete:
                if len(self._incomplete) >= self._max_incomplete:
                    # Evict oldest frame_id
                    oldest = min(self._incomplete)
                    del self._incomplete[oldest]
                self._incomplete[frame_id] = _FrameBuffer(
                    frame_id=frame_id,
                    total_chunks=total_chunks,
                    jpeg_size=jpeg_size,
                    sim_time_ns=sim_time_ns,
                )

            buf = self._incomplete[frame_id]
            buf.chunks[chunk_id] = payload

            if buf.is_complete():
                del self._incomplete[frame_id]
                self._decode_and_store(buf)

    def _decode_and_store(self, buf: _FrameBuffer) -> None:
        try:
            jpeg_bytes = buf.assemble()[: buf.jpeg_size]
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                logger.debug("Failed to decode JPEG for frame_id %d", buf.frame_id)
                return
            with self._lock:
                self._latest_frame = frame
                self._latest_sim_time_ns = buf.sim_time_ns
                self._frame_count += 1
        except Exception as exc:
            logger.debug("Frame decode error frame_id=%d: %s", buf.frame_id, exc)
