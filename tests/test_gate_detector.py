"""Tests for gate detection (color-based pipeline), CNN, and tracking."""

from __future__ import annotations

import numpy as np
import pytest

from aigrandprix.perception.gate_detector import (
    GateDetection,
    GateDetector,
    GateTrack,
    GateTracker,
)


def _make_blank_image(h: int = 360, w: int = 640) -> np.ndarray:
    """Create a blank dark BGR image (VADR-TS-002 resolution)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _draw_blue_rect(
    image: np.ndarray,
    cx: int,
    cy: int,
    half_w: int = 40,
    half_h: int = 40,
) -> np.ndarray:
    """Draw DCL-style dark blue gate rectangle (BGR ~(180,35,20))."""
    img = image.copy()
    y1, y2 = max(cy - half_h, 0), min(cy + half_h, img.shape[0])
    x1, x2 = max(cx - half_w, 0), min(cx + half_w, img.shape[1])
    img[y1:y2, x1:x2] = [180, 35, 20]  # DCL gate color BGR
    return img


# Keep backward-compat alias for other tests that may use red
_draw_red_rect = _draw_blue_rect


def _make_detection(pos: np.ndarray, gate_id: int = 0) -> GateDetection:
    """Helper to create a GateDetection with given 3D hint."""
    return GateDetection(
        center_px=(320, 240),
        corners_px=np.zeros((4, 2)),
        distance=float(np.linalg.norm(pos)),
        normal=np.array([0, 0, 1]),
        confidence=0.9,
        gate_id=gate_id,
    )


# ---------------------------------------------------------------------------
# Color detection tests
# ---------------------------------------------------------------------------

class TestColorDetection:
    def test_detects_dcl_blue_gate(self):
        """DCL dark blue gate rectangle should be detected."""
        img = _make_blank_image()
        img = _draw_blue_rect(img, cx=320, cy=180, half_w=50, half_h=50)
        detector = GateDetector(method="color")
        detections = detector.detect(img)
        assert len(detections) >= 1, "Should detect at least one gate"
        det = detections[0]
        assert abs(det.center_px[0] - 320) < 25
        assert abs(det.center_px[1] - 180) < 25

    def test_no_detection_on_blank_image(self):
        """No gates should be detected on a blank image."""
        img = _make_blank_image()
        detector = GateDetector(method="color")
        detections = detector.detect(img)
        assert len(detections) == 0

    def test_no_detection_on_red_image(self):
        """Red rectangles should NOT be detected (HSV filter targets blue)."""
        img = _make_blank_image()
        img[140:220, 280:360] = [0, 0, 255]  # pure red BGR
        detector = GateDetector(method="color")
        detections = detector.detect(img)
        assert len(detections) == 0

    def test_small_contour_filtered(self):
        """Very small blue areas should be filtered out (< area_filter threshold)."""
        img = _make_blank_image()
        img = _draw_blue_rect(img, cx=320, cy=180, half_w=5, half_h=5)
        detector = GateDetector(method="color")
        detections = detector.detect(img)
        assert len(detections) == 0, "Small area should be filtered"

    def test_multiple_gates_detected(self):
        """Multiple separated blue rectangles should yield multiple detections."""
        img = _make_blank_image()
        img = _draw_blue_rect(img, cx=150, cy=180, half_w=40, half_h=40)
        img = _draw_blue_rect(img, cx=490, cy=180, half_w=40, half_h=40)
        detector = GateDetector(method="color")
        detections = detector.detect(img)
        assert len(detections) >= 2, f"Expected 2 detections, got {len(detections)}"

    def test_distance_decreases_with_larger_gate(self):
        """A larger gate image should report a shorter distance."""
        detector = GateDetector(method="color")

        img_far = _make_blank_image()
        img_far = _draw_blue_rect(img_far, 320, 180, half_w=20, half_h=20)
        det_far = detector.detect(img_far)

        img_close = _make_blank_image()
        img_close = _draw_blue_rect(img_close, 320, 180, half_w=80, half_h=80)
        det_close = detector.detect(img_close)

        assert len(det_far) >= 1 and len(det_close) >= 1
        assert det_close[0].distance < det_far[0].distance, (
            "Larger gate should report shorter distance"
        )

    def test_confidence_bounded(self):
        """Confidence should be in [0, 1]."""
        img = _make_blank_image()
        img = _draw_blue_rect(img, 320, 180, half_w=50, half_h=50)
        detector = GateDetector(method="color")
        for det in detector.detect(img):
            assert 0.0 <= det.confidence <= 1.0


# ---------------------------------------------------------------------------
# CNN / Hybrid
# ---------------------------------------------------------------------------

class TestCNN:
    def test_cnn_no_model_returns_empty(self):
        """CNN without weights file should return empty list."""
        img = _make_blank_image()
        detector = GateDetector(method="cnn")
        assert detector.detect(img) == []

    def test_hybrid_falls_back_to_color(self):
        """Hybrid should fall back to color when CNN returns nothing."""
        img = _make_blank_image()
        img = _draw_red_rect(img, 320, 240, half_w=50, half_h=50)
        detector = GateDetector(method="hybrid")
        detections = detector.detect(img)
        assert len(detections) >= 1, "Hybrid should fallback to color detection"

    def test_centernet_forward_shape(self):
        """CenterNet model forward pass should produce correct output shapes."""
        import torch
        from aigrandprix.perception.cnn_model import GateCenterNet

        model = GateCenterNet(input_size=(320, 320))
        x = torch.randn(1, 3, 320, 320)
        out = model(x)
        assert out["heatmap"].shape == (1, 1, 40, 40)
        assert out["size"].shape == (1, 2, 40, 40)
        assert out["corners"].shape == (1, 8, 40, 40)

    def test_cnn_config_loading(self):
        """Verify CNN config params are stored correctly."""
        cfg = {
            "cnn": {
                "backend": "custom",
                "model_path": "/tmp/test.pt",
                "confidence_threshold": 0.7,
                "nms_threshold": 0.3,
                "input_size": [256, 256],
            }
        }
        detector = GateDetector(method="cnn", config=cfg)
        assert detector._cnn_backend == "custom"
        assert detector._cnn_model_path == "/tmp/test.pt"
        assert detector._cnn_confidence == 0.7
        assert detector._cnn_nms == 0.3
        assert detector._cnn_input_size == (256, 256)

    def test_hybrid_still_falls_back(self):
        """Hybrid without trained model → color detection."""
        img = _make_blank_image()
        img = _draw_red_rect(img, 320, 240, half_w=50, half_h=50)
        detector = GateDetector(method="hybrid", config={"cnn": {"backend": "custom"}})
        detections = detector.detect(img)
        assert len(detections) >= 1


# ---------------------------------------------------------------------------
# Temporal tracking
# ---------------------------------------------------------------------------

class TestGateTracker:
    def test_tracker_creates_tracks(self):
        """Two detections → two tracks (after min_hits)."""
        tracker = GateTracker({"min_hits": 1})
        det1 = _make_detection(np.array([1.0, 0, 5]))
        det2 = _make_detection(np.array([-1.0, 0, 5]))
        tracks = tracker.update(
            [(det1, np.array([1.0, 0, 5])), (det2, np.array([-1.0, 0, 5]))],
            timestamp=0.0,
        )
        assert len(tracks) == 2

    def test_tracker_matches_across_frames(self):
        """Same gate seen twice → 1 track with 2 hits."""
        tracker = GateTracker({"min_hits": 1})
        pos = np.array([0.0, 0.0, 5.0])
        det = _make_detection(pos)
        tracker.update([(det, pos.copy())], timestamp=0.0)
        tracks = tracker.update([(det, pos.copy())], timestamp=0.01)
        assert len(tracks) == 1
        assert tracks[0].hits == 2

    def test_tracker_prunes_stale(self):
        """Track should be removed after max_age without detection."""
        tracker = GateTracker({"max_age": 0.1, "min_hits": 1})
        pos = np.array([0.0, 0.0, 5.0])
        det = _make_detection(pos)
        tracker.update([(det, pos.copy())], timestamp=0.0)
        # No detection, time advances beyond max_age
        tracks = tracker.update([], timestamp=0.5)
        assert len(tracks) == 0

    def test_tracker_min_hits(self):
        """Track not active until min_hits reached."""
        tracker = GateTracker({"min_hits": 3})
        pos = np.array([0.0, 0.0, 5.0])
        det = _make_detection(pos)
        # First frame: 1 hit < 3
        tracks = tracker.update([(det, pos.copy())], timestamp=0.0)
        assert len(tracks) == 0
        # Second: 2 hits < 3
        tracks = tracker.update([(det, pos.copy())], timestamp=0.01)
        assert len(tracks) == 0
        # Third: 3 hits >= 3
        tracks = tracker.update([(det, pos.copy())], timestamp=0.02)
        assert len(tracks) == 1

    def test_tracker_reset(self):
        """Reset should clear all tracks."""
        tracker = GateTracker({"min_hits": 1})
        pos = np.array([0.0, 0.0, 5.0])
        det = _make_detection(pos)
        tracker.update([(det, pos.copy())], timestamp=0.0)
        tracker.reset()
        assert tracker.get_predicted_positions(0.0) == []

    def test_detect_and_track_without_tracker(self):
        """detect_and_track without tracker returns empty tracks."""
        img = _make_blank_image()
        img = _draw_red_rect(img, 320, 240, half_w=50, half_h=50)
        detector = GateDetector(method="color")
        dets, tracks = detector.detect_and_track(img, timestamp=0.0)
        assert len(dets) >= 1
        assert tracks == []


# ---------------------------------------------------------------------------
# GateDetection dataclass
# ---------------------------------------------------------------------------

class TestGateDetectionData:
    def test_gate_detection_fields(self):
        det = GateDetection(
            center_px=(100, 200),
            corners_px=np.zeros((4, 2)),
            distance=5.0,
            normal=np.array([0, 0, 1]),
            confidence=0.9,
            gate_id=1,
        )
        assert det.center_px == (100, 200)
        assert det.distance == 5.0
        assert det.gate_id == 1
