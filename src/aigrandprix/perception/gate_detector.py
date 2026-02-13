"""Gate detection using computer vision and neural networks.

Detects racing gates from camera feed, estimating:
- Gate center position (pixel coords)
- Gate corners (for pose estimation)
- Distance and orientation relative to drone
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GateDetection:
    """Detected gate with pose information."""

    center_px: tuple[int, int]  # pixel coordinates of gate center
    corners_px: np.ndarray  # 4x2 array of corner pixel coordinates
    distance: float  # estimated distance in meters
    normal: np.ndarray  # 3D normal vector of gate plane
    confidence: float  # detection confidence [0, 1]
    gate_id: int | None = None  # gate identifier if known


class GateDetector:
    """Detects racing gates from camera images.

    Supports multiple detection strategies:
    - Color-based segmentation (fast, less robust)
    - CNN-based detection (accurate, needs GPU)
    - Hybrid approach
    """

    def __init__(self, method: str = "hybrid") -> None:
        self.method = method
        self._model = None

    def detect(self, image: np.ndarray) -> list[GateDetection]:
        """Detect all visible gates in the image.

        Args:
            image: BGR image from drone camera (H, W, 3)

        Returns:
            List of detected gates sorted by distance (closest first)
        """
        if self.method == "color":
            return self._detect_color(image)
        elif self.method == "cnn":
            return self._detect_cnn(image)
        else:
            return self._detect_hybrid(image)

    def _detect_color(self, image: np.ndarray) -> list[GateDetection]:
        """Fast color-based gate detection using HSV thresholding."""
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Gate color range - will be tuned to actual gate colors
        lower = np.array([0, 100, 100])
        upper = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # filter noise
                continue

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            center = (int(rect[0][0]), int(rect[0][1]))

            # Rough distance estimation from apparent size
            apparent_size = max(rect[1])
            gate_real_size = 1.0  # meters, TBD
            focal_length = 500  # pixels, TBD from camera calibration
            distance = (gate_real_size * focal_length) / apparent_size if apparent_size > 0 else 999

            detections.append(GateDetection(
                center_px=center,
                corners_px=box,
                distance=distance,
                normal=np.array([0, 0, 1]),  # placeholder
                confidence=min(area / 5000, 1.0),
            ))

        return sorted(detections, key=lambda d: d.distance)

    def _detect_cnn(self, image: np.ndarray) -> list[GateDetection]:
        """CNN-based gate detection. To be implemented with trained model."""
        # TODO: Implement CNN detection (YOLO/custom architecture)
        # Reference: open-airlab/GateNet
        return []

    def _detect_hybrid(self, image: np.ndarray) -> list[GateDetection]:
        """Hybrid detection: CNN for detection, color for refinement."""
        cnn_results = self._detect_cnn(image)
        if cnn_results:
            return cnn_results
        return self._detect_color(image)
