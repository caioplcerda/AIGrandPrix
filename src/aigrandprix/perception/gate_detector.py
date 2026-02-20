"""Gate detection using computer vision and neural networks.

Detects racing gates from camera feed, estimating:
- Gate center position (pixel coords)
- Gate corners (for pose estimation)
- Distance and orientation relative to drone

Also provides temporal tracking of gates across frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


# ---------------------------------------------------------------------------
# Temporal gate tracking
# ---------------------------------------------------------------------------

@dataclass
class GateTrack:
    """A tracked gate across multiple frames."""

    track_id: int
    position_3d: np.ndarray  # Kalman-filtered 3D position
    position_cov: np.ndarray  # 3x3 covariance
    last_detection: GateDetection
    last_seen_time: float
    age: int = 0  # total frames since creation
    hits: int = 0  # number of successful matches
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))


class GateTracker:
    """Tracks gates across frames using Kalman filtering + Hungarian matching."""

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.max_age = config.get("max_age", 0.5)
        self.min_hits = config.get("min_hits", 3)
        self.match_threshold = config.get("match_threshold", 2.0)
        self._process_noise = config.get("process_noise", 0.01)
        self._measurement_noise = config.get("measurement_noise", 0.5)
        self._tracks: list[GateTrack] = []
        self._next_id = 0

    def update(
        self,
        detections_3d: list[tuple[GateDetection, np.ndarray]],
        timestamp: float,
    ) -> list[GateTrack]:
        """Update tracks with new detections.

        Args:
            detections_3d: list of (GateDetection, 3D_position) tuples
            timestamp: current time in seconds

        Returns:
            List of active tracks (hits >= min_hits)
        """
        from scipy.optimize import linear_sum_assignment

        # Predict: grow covariance (gates are static, position unchanged)
        for trk in self._tracks:
            dt = max(timestamp - trk.last_seen_time, 1e-6)
            trk.position_cov += np.eye(3) * self._process_noise * dt
            trk.age += 1

        if not self._tracks or not detections_3d:
            # Create new tracks for unmatched detections
            for det, pos in detections_3d:
                self._create_track(det, pos, timestamp)
            # Prune stale tracks
            self._prune(timestamp)
            return [t for t in self._tracks if t.hits >= self.min_hits]

        # Cost matrix: Euclidean distance
        n_tracks = len(self._tracks)
        n_dets = len(detections_3d)
        cost = np.full((n_tracks, n_dets), 1e6)
        for i, trk in enumerate(self._tracks):
            for j, (_, pos) in enumerate(detections_3d):
                cost[i, j] = float(np.linalg.norm(trk.position_3d - pos))

        # Hungarian matching
        row_idx, col_idx = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_dets = set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < self.match_threshold:
                det, pos = detections_3d[c]
                self._kalman_update(self._tracks[r], pos, det, timestamp)
                matched_tracks.add(r)
                matched_dets.add(c)

        # New tracks from unmatched detections
        for j in range(n_dets):
            if j not in matched_dets:
                det, pos = detections_3d[j]
                self._create_track(det, pos, timestamp)

        self._prune(timestamp)
        return [t for t in self._tracks if t.hits >= self.min_hits]

    def get_predicted_positions(self, timestamp: float) -> list[GateTrack]:
        """Return all tracks still within max_age."""
        return [
            t for t in self._tracks
            if (timestamp - t.last_seen_time) <= self.max_age
        ]

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 0

    def _create_track(
        self, det: GateDetection, pos: np.ndarray, timestamp: float
    ) -> None:
        self._tracks.append(GateTrack(
            track_id=self._next_id,
            position_3d=pos.copy(),
            position_cov=np.eye(3) * self._measurement_noise ** 2,
            last_detection=det,
            last_seen_time=timestamp,
            age=1,
            hits=1,
            normal=det.normal.copy(),
        ))
        self._next_id += 1

    def _kalman_update(
        self,
        track: GateTrack,
        measurement: np.ndarray,
        det: GateDetection,
        timestamp: float,
    ) -> None:
        """Kalman measurement update (H=I, direct position observation)."""
        R = np.eye(3) * self._measurement_noise ** 2
        P = track.position_cov
        S = P + R
        K = P @ np.linalg.inv(S)
        innovation = measurement - track.position_3d
        track.position_3d = track.position_3d + K @ innovation
        track.position_cov = (np.eye(3) - K) @ P
        track.last_detection = det
        track.last_seen_time = timestamp
        track.hits += 1
        track.normal = det.normal.copy()

    def _prune(self, timestamp: float) -> None:
        self._tracks = [
            t for t in self._tracks
            if (timestamp - t.last_seen_time) <= self.max_age
        ]


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class GateDetector:
    """Detects racing gates from camera images.

    Supports multiple detection strategies:
    - Color-based segmentation (fast, less robust)
    - CNN-based detection (accurate, needs GPU)
    - Hybrid approach
    """

    def __init__(self, method: str = "hybrid", config: dict | None = None) -> None:
        self.method = method
        self._model = None
        self._model_loaded = False
        self._detection_count = 0
        self._fallback_count = 0
        self._last_latency_ms = 0.0

        if config is not None:
            self._gate_color_lower = np.array(config.get("gate_color_lower", [0, 100, 100]))
            self._gate_color_upper = np.array(config.get("gate_color_upper", [10, 255, 255]))
            self._area_filter = config.get("area_filter", 500)
            self._focal_length = config.get("focal_length", 500)
            self._gate_real_size = config.get("gate_real_size", 1.0)
            self._confidence_scale = config.get("confidence_scale", 5000)
            # CNN config
            cnn_cfg = config.get("cnn", {})
            self._cnn_backend = cnn_cfg.get("backend", "custom")
            self._cnn_model_path = cnn_cfg.get("model_path", None)
            self._cnn_confidence = cnn_cfg.get("confidence_threshold", 0.5)
            self._cnn_nms = cnn_cfg.get("nms_threshold", 0.4)
            self._cnn_input_size = tuple(cnn_cfg.get("input_size", [320, 320]))
            # Tracking config
            tracking_cfg = config.get("tracking", {})
            if tracking_cfg.get("enabled", False):
                self._tracker = GateTracker(tracking_cfg)
            else:
                self._tracker = None
        else:
            self._gate_color_lower = np.array([0, 100, 100])
            self._gate_color_upper = np.array([10, 255, 255])
            self._area_filter = 500
            self._focal_length = 500
            self._gate_real_size = 1.0
            self._confidence_scale = 5000
            self._cnn_backend = "custom"
            self._cnn_model_path = None
            self._cnn_confidence = 0.5
            self._cnn_nms = 0.4
            self._cnn_input_size = (320, 320)
            self._tracker = None

    def detect(self, image: np.ndarray) -> list[GateDetection]:
        """Detect all visible gates with fallback chain.

        If primary method fails or returns empty, tries color detection as fallback.
        """
        import time
        t0 = time.perf_counter()
        self._detection_count += 1

        detections = []
        try:
            if self.method == "color":
                detections = self._detect_color(image)
            elif self.method == "cnn":
                detections = self._detect_cnn(image)
                if not detections:
                    detections = self._detect_color(image)
                    if detections:
                        self._fallback_count += 1
            else:  # hybrid
                detections = self._detect_hybrid(image)
        except Exception:
            # Last resort: try color detection
            try:
                detections = self._detect_color(image)
                self._fallback_count += 1
            except Exception:
                detections = []

        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return detections

    @property
    def detection_latency_ms(self) -> float:
        """Latency of last detect() call in milliseconds."""
        return self._last_latency_ms

    @property
    def fallback_rate(self) -> float:
        """Fraction of detections that used fallback chain."""
        if self._detection_count == 0:
            return 0.0
        return self._fallback_count / self._detection_count

    def detect_and_track(
        self,
        image: np.ndarray,
        timestamp: float,
        depth_estimator: object | None = None,
    ) -> tuple[list[GateDetection], list[GateTrack]]:
        """Detect gates and update temporal tracking.

        Args:
            image: BGR camera image
            timestamp: current time in seconds
            depth_estimator: optional DepthEstimator for 3D positioning

        Returns:
            (detections, active_tracks)
        """
        detections = self.detect(image)

        if self._tracker is None:
            return (detections, [])

        # Convert to 3D positions
        detections_3d: list[tuple[GateDetection, np.ndarray]] = []
        for det in detections:
            pos_3d = None
            if depth_estimator is not None:
                pose = depth_estimator.estimate_pose_from_detection(det)  # type: ignore[attr-defined]
                if pose is not None:
                    pos_3d = pose.position
            if pos_3d is None:
                # Heuristic: place on Z axis using distance estimate
                pos_3d = np.array([0.0, 0.0, det.distance])
            detections_3d.append((det, pos_3d))

        tracks = self._tracker.update(detections_3d, timestamp)
        return (detections, tracks)

    def _detect_color(self, image: np.ndarray) -> list[GateDetection]:
        """Fast color-based gate detection using HSV thresholding."""
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._gate_color_lower, self._gate_color_upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._area_filter:
                continue

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            center = (int(rect[0][0]), int(rect[0][1]))

            # Rough distance estimation from apparent size
            apparent_size = max(rect[1])
            distance = (
                (self._gate_real_size * self._focal_length) / apparent_size
                if apparent_size > 0
                else 999
            )

            detections.append(GateDetection(
                center_px=center,
                corners_px=box,
                distance=distance,
                normal=np.array([0, 0, 1]),  # placeholder
                confidence=min(area / self._confidence_scale, 1.0),
            ))

        return sorted(detections, key=lambda d: d.distance)

    def _detect_cnn(self, image: np.ndarray) -> list[GateDetection]:
        """CNN-based gate detection using CenterNet or YOLO backend."""
        if not self._model_loaded:
            self._model = self._load_model()
            self._model_loaded = True

        if self._model is None:
            return []

        return self._run_centernet(image)

    def _load_model(self) -> object | None:
        """Load CNN model weights. Returns None if unavailable."""
        if self._cnn_backend == "yolo":
            try:
                from ultralytics import YOLO
                if self._cnn_model_path:
                    return YOLO(self._cnn_model_path)
            except (ImportError, Exception):
                pass
            return None

        # Default: custom CenterNet
        try:
            import torch
            from aigrandprix.perception.cnn_model import GateCenterNet

            model = GateCenterNet(input_size=self._cnn_input_size)
            if self._cnn_model_path:
                state = torch.load(self._cnn_model_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
            else:
                return None  # No weights → can't run
            model.eval()
            return model
        except (ImportError, Exception):
            return None

    def _run_centernet(self, image: np.ndarray) -> list[GateDetection]:
        """Run CenterNet inference and decode detections."""
        import cv2
        import torch

        h, w = image.shape[:2]
        inp_h, inp_w = self._cnn_input_size

        # Preprocess
        resized = cv2.resize(image, (inp_w, inp_h))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            out = self._model(tensor)

        heatmap = out["heatmap"][0, 0].numpy()  # (H/8, W/8)
        size_map = out["size"][0].numpy()  # (2, H/8, W/8)
        corner_map = out["corners"][0].numpy()  # (8, H/8, W/8)

        # Find peaks above threshold
        from scipy.ndimage import maximum_filter
        hm_max = maximum_filter(heatmap, size=3)
        peaks = (heatmap == hm_max) & (heatmap >= self._cnn_confidence)
        ys, xs = np.where(peaks)

        if len(xs) == 0:
            return []

        stride = 8
        scale_x = w / inp_w
        scale_y = h / inp_h

        detections = []
        for yi, xi in zip(ys, xs):
            conf = float(heatmap[yi, xi])
            cx = (xi * stride + stride // 2) * scale_x
            cy = (yi * stride + stride // 2) * scale_y
            bw = float(size_map[0, yi, xi]) * w
            bh = float(size_map[1, yi, xi]) * h

            # Decode corner offsets
            corners = np.zeros((4, 2))
            for k in range(4):
                corners[k, 0] = cx + float(corner_map[k * 2, yi, xi]) * w
                corners[k, 1] = cy + float(corner_map[k * 2 + 1, yi, xi]) * h

            # Rough distance from bbox size
            apparent = max(bw, bh) if max(bw, bh) > 0 else 1.0
            dist = (self._gate_real_size * self._focal_length) / apparent

            detections.append(GateDetection(
                center_px=(int(cx), int(cy)),
                corners_px=corners,
                distance=dist,
                normal=np.array([0, 0, 1]),
                confidence=conf,
            ))

        # NMS by distance (keep closest non-overlapping)
        detections.sort(key=lambda d: -d.confidence)
        kept: list[GateDetection] = []
        for det in detections:
            overlap = False
            for k in kept:
                if abs(det.center_px[0] - k.center_px[0]) < 30 and abs(
                    det.center_px[1] - k.center_px[1]
                ) < 30:
                    overlap = True
                    break
            if not overlap:
                kept.append(det)

        return sorted(kept, key=lambda d: d.distance)

    def _detect_hybrid(self, image: np.ndarray) -> list[GateDetection]:
        """Hybrid detection: CNN for detection, color for refinement."""
        cnn_results = self._detect_cnn(image)
        if cnn_results:
            return cnn_results
        return self._detect_color(image)
