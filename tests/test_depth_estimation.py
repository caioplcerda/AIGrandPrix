"""Tests for PnP depth and pose estimation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from aigrandprix.perception.depth_estimation import (
    CameraIntrinsics,
    DepthEstimator,
    GatePose,
    _order_corners,
)


def _project_gate(estimator: DepthEstimator, rvec, tvec):
    """Project gate model points to 2D using known pose."""
    projected, _ = cv2.projectPoints(
        estimator._gate_model_points,
        rvec, tvec,
        estimator.intrinsics.matrix,
        estimator._dist_coeffs,
    )
    return projected.reshape(4, 2)


class TestCameraIntrinsics:
    def test_camera_matrix_shape(self):
        cam = CameraIntrinsics(fx=320, fy=320, cx=320, cy=240)
        K = cam.matrix
        assert K.shape == (3, 3)
        assert K[0, 0] == 320
        assert K[1, 1] == 320
        assert K[2, 2] == 1

    def test_from_fov(self):
        """90° FOV, 640×480 → fx ≈ 320."""
        cam = CameraIntrinsics.from_fov(90.0, 640, 480)
        assert abs(cam.fx - 320.0) < 1.0
        assert abs(cam.fy - 320.0) < 1.0
        assert cam.cx == 320.0
        assert cam.cy == 240.0


class TestDepthEstimator:
    def test_frontal_gate_distance(self):
        """Gate 5m directly ahead → distance ≈ 5m (±10%)."""
        est = DepthEstimator(gate_width=1.0, gate_height=1.0)
        rvec = np.zeros(3)  # no rotation
        tvec = np.array([0.0, 0.0, 5.0])  # 5m ahead
        corners = _project_gate(est, rvec, tvec)
        pose = est.estimate_pose(corners)
        assert pose is not None
        assert abs(pose.distance - 5.0) < 0.5  # within 10%

    def test_angled_gate(self):
        """Gate rotated 30° around Y → normal should have significant X component."""
        est = DepthEstimator(gate_width=1.0, gate_height=1.0)
        angle = np.radians(30)
        rvec = np.array([0.0, angle, 0.0])
        tvec = np.array([0.0, 0.0, 5.0])
        corners = _project_gate(est, rvec, tvec)
        pose = est.estimate_pose(corners)
        assert pose is not None
        # Normal Z-component should be reduced from 1.0 due to rotation
        assert abs(pose.normal[2]) < 0.97  # not perfectly frontal

    def test_various_distances(self):
        """Test accuracy at 2m, 5m, 10m."""
        est = DepthEstimator(gate_width=1.0, gate_height=1.0)
        for true_dist in [2.0, 5.0, 10.0]:
            rvec = np.zeros(3)
            tvec = np.array([0.0, 0.0, true_dist])
            corners = _project_gate(est, rvec, tvec)
            pose = est.estimate_pose(corners)
            assert pose is not None, f"PnP failed at {true_dist}m"
            assert abs(pose.distance - true_dist) < true_dist * 0.1

    def test_noisy_corners(self):
        """2px noise → distance within 15%."""
        rng = np.random.default_rng(42)
        est = DepthEstimator(gate_width=1.0, gate_height=1.0)
        rvec = np.zeros(3)
        tvec = np.array([0.0, 0.0, 5.0])
        corners = _project_gate(est, rvec, tvec)
        noisy = corners + rng.normal(0, 2.0, corners.shape)
        pose = est.estimate_pose(noisy)
        assert pose is not None
        assert abs(pose.distance - 5.0) < 0.75  # within 15%

    def test_degenerate_returns_none(self):
        """Collinear corners → None."""
        est = DepthEstimator()
        # Four nearly collinear points
        corners = np.array([
            [100, 200], [200, 200], [300, 200], [400, 200]
        ], dtype=np.float64)
        assert est.estimate_pose(corners) is None

    def test_corner_ordering(self):
        """Shuffled corners → correctly reordered to BL, BR, TR, TL (IPPE convention)."""
        # Rectangle: TL=(10,10), TR=(90,10), BR=(90,90), BL=(10,90)
        # Expected IPPE order: BL, BR, TR, TL
        expected = np.array([[10, 90], [90, 90], [90, 10], [10, 10]], dtype=np.float64)
        shuffled = np.array([[90, 90], [10, 10], [10, 90], [90, 10]], dtype=np.float64)
        ordered = _order_corners(shuffled)
        np.testing.assert_allclose(ordered, expected, atol=1.0)

    def test_reprojection_error_small(self):
        """Clean synthetic corners → low reprojection error."""
        est = DepthEstimator(gate_width=1.0, gate_height=1.0)
        rvec = np.zeros(3)
        tvec = np.array([0.0, 0.0, 5.0])
        corners = _project_gate(est, rvec, tvec)
        pose = est.estimate_pose(corners)
        assert pose is not None
        assert pose.reprojection_error < 2.0

    def test_estimate_pose_from_detection(self):
        """Convenience wrapper should produce same result."""
        from aigrandprix.perception.gate_detector import GateDetection

        est = DepthEstimator(gate_width=1.0, gate_height=1.0)
        rvec = np.zeros(3)
        tvec = np.array([0.0, 0.0, 5.0])
        corners = _project_gate(est, rvec, tvec)
        det = GateDetection(
            center_px=(int(corners[:, 0].mean()), int(corners[:, 1].mean())),
            corners_px=corners,
            distance=5.0,
            normal=np.array([0, 0, 1]),
            confidence=0.9,
        )
        pose = est.estimate_pose_from_detection(det)
        assert pose is not None
        assert abs(pose.distance - 5.0) < 0.5
