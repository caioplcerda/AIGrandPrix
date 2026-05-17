"""Gate renderer for DCL mock simulator vision stream (VADR-TS-002 §3.7, §3.8).

Renders gate frames into 640×360 JPEG images using OpenCV.

Gate geometry (VADR-TS-002):
  - Outer boundary: 2700×2700 mm (2.7×2.7 m)
  - Inner opening: 1500×1500 mm (1.5×1.5 m)
  - Frame border width: (2.7-1.5)/2 = 0.6 m
  - Gate color: dark blue (DCL visual style)

Camera (VADR-TS-002 §3.8):
  - Resolution: 640×360 px
  - Pinhole model: fx=fy=320, cx=320, cy=180
  - VFoV: 90° (matches fx=fy=320 on 640×360)
  - Camera tilts +20° UP from body X axis
  - Body frame: X forward, Y right, Z down (NED)
  - Camera frame: X right, Y down, Z forward (OpenCV convention)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# Camera intrinsics (VADR-TS-002)
IMG_W, IMG_H = 640, 360
FX = FY = 320.0
CX, CY = 320.0, 180.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]], dtype=np.float64)

# Gate dimensions (m)
GATE_OUTER = 2.7
GATE_INNER = 1.5
GATE_BORDER = (GATE_OUTER - GATE_INNER) / 2  # 0.6 m

# Camera tilt: +20° upward from body X axis
_CAM_TILT_RAD = math.radians(20.0)

# Body-to-camera rotation:
#   Camera is mounted such that:
#     - Z_cam (forward) ≈ body X (forward), tilted +20° up (= -Y_cam tilt)
#     - X_cam (right) = body Y (right)
#     - Y_cam (down) ≈ body Z (down), modified by tilt
#   R_body_to_cam: rotate body frame to OpenCV camera frame
#   Step 1: body → intermediate (X fwd, Y right, Z down) → (X right, Y down, Z fwd)
#     = permute: x_cam=y_body, y_cam=z_body, z_cam=x_body
#   Step 2: apply tilt: rotate around cam X by +tilt (gate moves up in image)
_R_BODY_TO_OPENCAM_NO_TILT = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
], dtype=np.float64)

_ct = math.cos(_CAM_TILT_RAD)
_st = math.sin(_CAM_TILT_RAD)
_R_TILT = np.array([
    [1, 0,    0   ],
    [0, _ct,  _st ],
    [0, -_st, _ct ],
], dtype=np.float64)

R_BODY_TO_CAM = _R_TILT @ _R_BODY_TO_OPENCAM_NO_TILT  # full body → camera


def _rpy_to_R_ned_to_body(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """NED-to-body rotation matrix (transpose of body-to-NED)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    R_body_ned = np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                  cp * cr],
    ])
    return R_body_ned.T  # ned_to_body = body_to_ned.T


def world_to_camera(
    point_ned: np.ndarray,
    drone_pos_ned: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """Transform a NED world point into OpenCV camera frame (X right, Y down, Z fwd)."""
    # World → body
    R_ned_to_body = _rpy_to_R_ned_to_body(roll, pitch, yaw)
    pt_body = R_ned_to_body @ (point_ned - drone_pos_ned)
    # Body → camera
    return R_BODY_TO_CAM @ pt_body


def project_point(cam_pt: np.ndarray) -> tuple[int, int] | None:
    """Project camera-frame point to pixel. Returns None if behind camera."""
    if cam_pt[2] <= 0.05:
        return None
    u = int(K[0, 0] * cam_pt[0] / cam_pt[2] + K[0, 2])
    v = int(K[1, 1] * cam_pt[1] / cam_pt[2] + K[1, 2])
    return u, v


@dataclass
class GateVisual:
    """A gate with NED position and facing direction for rendering."""

    center_ned: np.ndarray        # gate center in NED (m)
    normal_ned: np.ndarray        # unit vector perpendicular to gate plane (facing drone)
    gate_id: int = 0
    color_bgr: tuple[int, int, int] = (180, 30, 20)  # DCL blue in BGR


def render_frame(
    drone_pos_ned: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
    gates: list[GateVisual],
    background_bgr: tuple[int, int, int] = (60, 50, 40),
) -> np.ndarray:
    """Render a 640×360 BGR image of the gates visible from the drone.

    Only gates within ~40m and in front hemisphere are rendered.

    Returns:
        BGR numpy array shape (360, 640, 3) uint8.
    """
    img = np.full((IMG_H, IMG_W, 3), background_bgr, dtype=np.uint8)
    _render_ground_line(img, drone_pos_ned, roll, pitch, yaw)

    for gate in gates:
        _render_gate(img, drone_pos_ned, roll, pitch, yaw, gate)

    return img


def render_jpeg(
    drone_pos_ned: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
    gates: list[GateVisual],
    jpeg_quality: int = 80,
) -> bytes:
    """Render frame and encode as JPEG bytes."""
    img = render_frame(drone_pos_ned, roll, pitch, yaw, gates)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_gate(
    img: np.ndarray,
    drone_pos_ned: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
    gate: GateVisual,
) -> None:
    """Draw gate outer frame + inner opening on image."""
    dist = float(np.linalg.norm(gate.center_ned - drone_pos_ned))
    if dist > 40.0 or dist < 0.3:
        return

    # Gate axes in NED: up = -Z_ned, right = cross(normal, up)
    up_ned = np.array([0.0, 0.0, -1.0])  # NED up = -Z
    right_ned = np.cross(gate.normal_ned, up_ned)
    norm = np.linalg.norm(right_ned)
    if norm < 1e-6:
        # Gate facing straight up/down — use Y as right
        right_ned = np.array([0.0, 1.0, 0.0])
    else:
        right_ned /= norm

    ho = GATE_OUTER / 2.0
    hi = GATE_INNER / 2.0
    c = gate.center_ned

    # Outer corners (clockwise from top-left when facing gate)
    outer_pts_ned = [
        c - right_ned * ho + up_ned * ho,
        c + right_ned * ho + up_ned * ho,
        c + right_ned * ho - up_ned * ho,
        c - right_ned * ho - up_ned * ho,
    ]
    # Inner corners (hole in gate)
    inner_pts_ned = [
        c - right_ned * hi + up_ned * hi,
        c + right_ned * hi + up_ned * hi,
        c + right_ned * hi - up_ned * hi,
        c - right_ned * hi - up_ned * hi,
    ]

    # Project to pixels
    outer_px = [project_point(world_to_camera(p, drone_pos_ned, roll, pitch, yaw)) for p in outer_pts_ned]
    inner_px = [project_point(world_to_camera(p, drone_pos_ned, roll, pitch, yaw)) for p in inner_pts_ned]

    if any(p is None for p in outer_px) or any(p is None for p in inner_px):
        # Gate partially behind camera — skip (simplification)
        return

    outer_pts = np.array(outer_px, dtype=np.int32)
    inner_pts = np.array(inner_px, dtype=np.int32)

    # Distance-based thickness
    thickness = max(3, int(20.0 / dist))

    # Draw outer border as solid filled polygon (gate frame material)
    cv2.fillPoly(img, [outer_pts], gate.color_bgr)

    # Draw inner opening as dark hole
    cv2.fillPoly(img, [inner_pts], (20, 20, 20))

    # Bright border lines on outer edges
    for i in range(4):
        p1 = outer_pts[i]
        p2 = outer_pts[(i + 1) % 4]
        cv2.line(img, tuple(p1), tuple(p2), (220, 60, 40), thickness)


def _render_ground_line(
    img: np.ndarray,
    drone_pos_ned: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
) -> None:
    """Render a simple horizon/ground reference line."""
    altitude = -drone_pos_ned[2]  # altitude = -Z_ned
    if altitude < 0.1:
        return
    # Horizon: ground is below, place a gradient at bottom of image
    horizon_y = int(IMG_H * (0.5 + pitch / math.radians(90) * 0.5))
    horizon_y = max(0, min(IMG_H - 1, horizon_y))
    if horizon_y < IMG_H:
        img[horizon_y:, :] = (35, 60, 35)  # dark green = ground
