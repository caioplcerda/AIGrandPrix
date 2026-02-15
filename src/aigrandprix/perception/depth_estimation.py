"""PnP-based depth and pose estimation for racing gates.

Uses OpenCV's solvePnP with IPPE (coplanar) to recover gate pose
from 2D corner detections and known gate geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aigrandprix.perception.gate_detector import GateDetection


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        """3x3 camera intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)

    @classmethod
    def from_config(cls, config: dict) -> CameraIntrinsics:
        """Create from a config dict with fx, fy, cx, cy keys."""
        return cls(
            fx=config.get("fx", 320.0),
            fy=config.get("fy", 320.0),
            cx=config.get("cx", 320.0),
            cy=config.get("cy", 240.0),
        )

    @classmethod
    def from_fov(cls, fov_deg: float, width: int, height: int) -> CameraIntrinsics:
        """Derive intrinsics from horizontal FOV and image dimensions."""
        fx = (width / 2.0) / np.tan(np.radians(fov_deg / 2.0))
        fy = fx  # square pixels
        return cls(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0)


@dataclass
class GatePose:
    """Estimated 3D pose of a gate relative to the camera."""

    position: np.ndarray  # [x, y, z] translation in camera frame (meters)
    rotation: np.ndarray  # 3x3 rotation matrix (gate -> camera)
    distance: float  # Euclidean distance to gate center
    normal: np.ndarray  # gate normal in camera frame (3D unit vector)
    reprojection_error: float  # RMS pixel error


def _order_corners(corners: np.ndarray) -> np.ndarray:
    """Sort 4 corners to match IPPE_SQUARE model convention.

    Returns corners in order: BL_img, BR_img, TR_img, TL_img
    which corresponds to model points (+Y bottom, -Y top in image).
    """
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    ys = corners[:, 1]
    # Bottom pair (larger y in image) → model +Y
    bot_idx = np.argsort(ys)[2:]
    # Top pair (smaller y in image) → model -Y
    top_idx = np.argsort(ys)[:2]
    bot = corners[bot_idx]
    top = corners[top_idx]
    # Bottom: left first (BL), then right (BR)
    bl, br = (bot[0], bot[1]) if bot[0, 0] <= bot[1, 0] else (bot[1], bot[0])
    # Top: right first (TR), then left (TL)
    tr, tl = (top[0], top[1]) if top[0, 0] >= top[1, 0] else (top[1], top[0])
    return np.array([bl, br, tr, tl], dtype=np.float64)


class DepthEstimator:
    """Estimates gate depth and 3D pose from 2D corner detections via PnP."""

    def __init__(
        self,
        gate_width: float = 1.0,
        gate_height: float = 1.0,
        config: dict | None = None,
    ) -> None:
        hw = gate_width / 2.0
        hh = gate_height / 2.0
        # Gate model points for SOLVEPNP_IPPE_SQUARE convention:
        # P0(-hw,+hh), P1(+hw,+hh), P2(+hw,-hh), P3(-hw,-hh)
        # In image: +Y_model → large pixel y (bottom), -Y_model → small pixel y (top)
        # So: P0=BL_img, P1=BR_img, P2=TR_img, P3=TL_img
        self._gate_model_points = np.array([
            [-hw,  hh, 0],  # P0: BL in image
            [ hw,  hh, 0],  # P1: BR in image
            [ hw, -hh, 0],  # P2: TR in image
            [-hw, -hh, 0],  # P3: TL in image
        ], dtype=np.float64)

        if config and "camera" in config:
            self.intrinsics = CameraIntrinsics.from_config(config["camera"])
        elif config and "camera_fov" in config:
            self.intrinsics = CameraIntrinsics.from_fov(
                config["camera_fov"],
                config.get("image_width", 640),
                config.get("image_height", 480),
            )
        else:
            self.intrinsics = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0)

        self._dist_coeffs = np.zeros(4)

    def estimate_pose(self, corners_px: np.ndarray) -> GatePose | None:
        """Estimate gate pose from 4 corner pixel coordinates.

        Args:
            corners_px: 4x2 array of corner pixel coordinates

        Returns:
            GatePose or None if PnP fails or corners are degenerate.
        """
        corners_px = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)

        # Validate: check for degenerate (collinear) corners via area
        # Shoelace formula for polygon area
        x = corners_px[:, 0]
        y = corners_px[:, 1]
        area = 0.5 * abs(
            np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        )
        if area < 100:  # too small / degenerate
            return None

        ordered = _order_corners(corners_px)
        image_points = ordered.reshape(4, 1, 2)

        success, rvec, tvec = cv2.solvePnP(
            self._gate_model_points,
            image_points,
            self.intrinsics.matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if not success:
            return None

        R, _ = cv2.Rodrigues(rvec)
        position = tvec.flatten()
        distance = float(np.linalg.norm(position))
        normal = R[:, 2]  # 3rd column = gate Z-axis in camera frame
        normal = normal / (np.linalg.norm(normal) + 1e-12)

        # Reprojection error
        projected, _ = cv2.projectPoints(
            self._gate_model_points, rvec, tvec,
            self.intrinsics.matrix, self._dist_coeffs,
        )
        projected = projected.reshape(4, 2)
        reproj_err = float(np.sqrt(np.mean((projected - ordered) ** 2)))

        return GatePose(
            position=position,
            rotation=R,
            distance=distance,
            normal=normal,
            reprojection_error=reproj_err,
        )

    def estimate_pose_from_detection(self, detection: GateDetection) -> GatePose | None:
        """Convenience wrapper for GateDetection objects."""
        return self.estimate_pose(detection.corners_px)
