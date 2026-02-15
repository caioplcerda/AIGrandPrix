"""Tests for gate detection (color-based pipeline)."""

from __future__ import annotations

import numpy as np
import pytest

from aigrandprix.perception.gate_detector import GateDetection, GateDetector


def _make_blank_image(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a blank dark BGR image."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _draw_red_rect(
    image: np.ndarray,
    cx: int,
    cy: int,
    half_w: int = 40,
    half_h: int = 40,
) -> np.ndarray:
    """Draw a filled red rectangle on the image (BGR: red = [0,0,255])."""
    img = image.copy()
    y1, y2 = max(cy - half_h, 0), min(cy + half_h, img.shape[0])
    x1, x2 = max(cx - half_w, 0), min(cx + half_w, img.shape[1])
    img[y1:y2, x1:x2] = [0, 0, 255]  # pure red in BGR
    return img


# ---------------------------------------------------------------------------
# Color detection tests
# ---------------------------------------------------------------------------

class TestColorDetection:
    def test_detects_red_gate(self):
        """A large red rectangle should be detected as a gate."""
        img = _make_blank_image()
        img = _draw_red_rect(img, cx=320, cy=240, half_w=50, half_h=50)
        detector = GateDetector(method="color")
        detections = detector.detect(img)
        assert len(detections) >= 1, "Should detect at least one gate"
        det = detections[0]
        # Center should be near the drawn rectangle
        assert abs(det.center_px[0] - 320) < 20
        assert abs(det.center_px[1] - 240) < 20

    def test_no_detection_on_blank_image(self):
        """No gates should be detected on a blank image."""
        img = _make_blank_image()
        detector = GateDetector(method="color")
        detections = detector.detect(img)
        assert len(detections) == 0

    def test_no_detection_on_blue_image(self):
        """Blue rectangles should NOT be detected as gates (HSV filter is for red)."""
        img = _make_blank_image()
        # Blue in BGR
        img[200:280, 280:360] = [255, 0, 0]
        detector = GateDetector(method="color")
        detections = detector.detect(img)
        assert len(detections) == 0

    def test_small_contour_filtered(self):
        """Very small red areas should be filtered out (< 500px area)."""
        img = _make_blank_image()
        # 10x10 = 100 pixels, below the 500 threshold
        img = _draw_red_rect(img, cx=320, cy=240, half_w=5, half_h=5)
        detector = GateDetector(method="color")
        detections = detector.detect(img)
        assert len(detections) == 0, "Small red area should be filtered"

    def test_multiple_gates_detected(self):
        """Multiple separated red rectangles should yield multiple detections."""
        img = _make_blank_image()
        img = _draw_red_rect(img, cx=150, cy=240, half_w=40, half_h=40)
        img = _draw_red_rect(img, cx=490, cy=240, half_w=40, half_h=40)
        detector = GateDetector(method="color")
        detections = detector.detect(img)
        assert len(detections) >= 2, f"Expected 2 detections, got {len(detections)}"

    def test_distance_decreases_with_larger_gate(self):
        """A larger gate image should report a shorter distance."""
        detector = GateDetector(method="color")

        img_far = _make_blank_image()
        img_far = _draw_red_rect(img_far, 320, 240, half_w=20, half_h=20)
        det_far = detector.detect(img_far)

        img_close = _make_blank_image()
        img_close = _draw_red_rect(img_close, 320, 240, half_w=80, half_h=80)
        det_close = detector.detect(img_close)

        assert len(det_far) >= 1 and len(det_close) >= 1
        assert det_close[0].distance < det_far[0].distance, (
            "Larger gate should report shorter distance"
        )

    def test_confidence_bounded(self):
        """Confidence should be in [0, 1]."""
        img = _make_blank_image()
        img = _draw_red_rect(img, 320, 240, half_w=50, half_h=50)
        detector = GateDetector(method="color")
        for det in detector.detect(img):
            assert 0.0 <= det.confidence <= 1.0


# ---------------------------------------------------------------------------
# CNN / Hybrid stubs
# ---------------------------------------------------------------------------

class TestCNNStub:
    def test_cnn_returns_empty(self):
        """CNN stub should return empty list (not yet implemented)."""
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
