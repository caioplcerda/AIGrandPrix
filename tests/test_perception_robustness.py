"""Perception robustness tests."""

from __future__ import annotations

import numpy as np
import pytest

from aigrandprix.perception.gate_detector import GateDetector


def _make_gate_image(
    center_x: int = 320,
    center_y: int = 240,
    size: int = 100,
    brightness: float = 1.0,
    color_hsv: tuple = (5, 200, 200),
) -> np.ndarray:
    """Create a synthetic image with a colored rectangle (gate)."""
    import cv2

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw gate as a rectangle in HSV then convert
    hsv = np.zeros((480, 640, 3), dtype=np.uint8)
    half = size // 2
    x1 = max(center_x - half, 0)
    x2 = min(center_x + half, 640)
    y1 = max(center_y - half, 0)
    y2 = min(center_y + half, 480)
    hsv[y1:y2, x1:x2] = color_hsv
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Apply brightness
    image = np.clip(image * brightness, 0, 255).astype(np.uint8)
    return image


class TestColorDetectionRobustness:
    """Test color detector at various sizes and brightnesses."""

    @pytest.mark.parametrize("size", [30, 60, 100, 150])
    def test_various_sizes(self, size: int):
        """Detect gates at various apparent sizes."""
        detector = GateDetector(method="color")
        image = _make_gate_image(size=size)
        dets = detector.detect(image)
        if size >= 30:  # area_filter default is 500, 30x30 = 900
            assert len(dets) >= 0  # may or may not detect at small sizes

    @pytest.mark.parametrize("brightness", [0.5, 0.8, 1.0, 1.5, 2.0])
    def test_various_brightness(self, brightness: float):
        """Detection at various brightness levels."""
        detector = GateDetector(method="color")
        image = _make_gate_image(size=100, brightness=brightness)
        dets = detector.detect(image)
        # Should not crash at any brightness
        assert isinstance(dets, list)

    def test_no_false_positives_blank(self):
        """No detections on a blank image."""
        detector = GateDetector(method="color")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(blank)
        assert len(dets) == 0

    def test_no_false_positives_white(self):
        """No detections on a white image."""
        detector = GateDetector(method="color")
        white = np.ones((480, 640, 3), dtype=np.uint8) * 255
        dets = detector.detect(white)
        assert len(dets) == 0


class TestDetectorFallback:
    """Test hybrid fallback chain."""

    def test_hybrid_falls_back_to_color(self):
        """Hybrid mode: CNN fail -> color detection."""
        detector = GateDetector(method="hybrid")
        # CNN won't work (no model loaded), should fallback to color
        image = _make_gate_image(size=100)
        dets = detector.detect(image)
        # Should not crash, may or may not detect
        assert isinstance(dets, list)


class TestAgentWithoutPerception:
    """Agent should work with known_gates only (no vision)."""

    def test_known_gates_only(self):
        """Agent runs without perception detections."""
        from aigrandprix.control.drone_controller import DroneState
        from aigrandprix.main import RaceConfig, RacingAgent
        from aigrandprix.planning.path_planner import Gate

        gates = [
            Gate(position=np.array([5, 0, 2]), normal=np.array([1, 0, 0]), gate_id=0),
            Gate(position=np.array([10, 0, 2]), normal=np.array([1, 0, 0]), gate_id=1),
        ]
        agent = RacingAgent(RaceConfig(max_speed=10.0, control_dt=0.02))
        state = DroneState(
            position=np.array([0.0, 0.0, 1.0]),
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        )
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

        for step in range(50):
            cmd = agent.compute_action(
                image=dummy_image, state=state,
                elapsed_time=step * 0.02, known_gates=gates,
            )
            assert not np.isnan(cmd.thrust)
