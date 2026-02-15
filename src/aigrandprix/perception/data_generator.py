"""Synthetic training data generation for gate detection.

Generates labeled images with rendered gate frames for training
CNN-based gate detectors. Supports COCO-format annotation output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from aigrandprix.perception.depth_estimation import CameraIntrinsics


@dataclass
class GateAnnotation:
    """Annotation for a single gate in an image."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h) in pixels
    corners: np.ndarray  # 4x2 pixel coordinates
    center: tuple[int, int]
    distance: float  # true distance in meters


@dataclass
class SyntheticSample:
    """A generated training sample."""

    image: np.ndarray  # BGR (H, W, 3) uint8
    annotations: list[GateAnnotation]
    metadata: dict = field(default_factory=dict)


class GateDataGenerator:
    """Generates synthetic training images with gate annotations."""

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.width = config.get("image_width", 640)
        self.height = config.get("image_height", 480)
        self.gate_width = config.get("gate_real_width", 1.0)
        self.gate_height = config.get("gate_real_height", 1.0)

        cam_cfg = config.get("camera", {})
        if cam_cfg:
            self.intrinsics = CameraIntrinsics.from_config(cam_cfg)
        else:
            fov = config.get("camera_fov", 90)
            self.intrinsics = CameraIntrinsics.from_fov(fov, self.width, self.height)

        # Gate model corners in 3D (same convention as DepthEstimator)
        hw, hh = self.gate_width / 2.0, self.gate_height / 2.0
        self._gate_3d = np.array([
            [-hw,  hh, 0],
            [ hw,  hh, 0],
            [ hw, -hh, 0],
            [-hw, -hh, 0],
        ], dtype=np.float64)

        # Randomization ranges
        self._dist_range = (1.0, 15.0)
        self._offset_range = (-2.0, 2.0)
        self._rot_range = np.radians(45)

    def generate_sample(self, rng: np.random.Generator | None = None) -> SyntheticSample:
        """Generate one synthetic training sample."""
        rng = rng or np.random.default_rng()
        image = self._random_background(rng)
        annotations: list[GateAnnotation] = []

        # Decide number of gates (0, 1, or 2)
        r = rng.random()
        num_gates = 0 if r < 0.1 else (2 if r > 0.9 else 1)

        for _ in range(num_gates):
            ann = self._render_gate(image, rng)
            if ann is not None:
                annotations.append(ann)

        image = self._augment(image, rng)

        return SyntheticSample(
            image=image,
            annotations=annotations,
            metadata={"num_gates": num_gates, "width": self.width, "height": self.height},
        )

    def generate_dataset(
        self, num_images: int, output_dir: str, seed: int = 42
    ) -> dict:
        """Generate a dataset of images with COCO-format annotations."""
        rng = np.random.default_rng(seed)
        out = Path(output_dir)
        img_dir = out / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        coco: dict = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "gate"}],
        }
        ann_id = 1

        for i in range(num_images):
            sample = self.generate_sample(rng)
            fname = f"{i:06d}.png"
            cv2.imwrite(str(img_dir / fname), sample.image)

            coco["images"].append({
                "id": i,
                "file_name": fname,
                "width": self.width,
                "height": self.height,
            })

            for ann in sample.annotations:
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [int(v) for v in ann.bbox],
                    "corners": ann.corners.tolist(),
                    "distance": float(ann.distance),
                })
                ann_id += 1

        ann_path = out / "annotations.json"
        with open(ann_path, "w") as f:
            json.dump(coco, f)

        return coco

    def _random_background(self, rng: np.random.Generator) -> np.ndarray:
        """Create a random background image."""
        choice = rng.integers(0, 3)
        if choice == 0:
            # Solid color
            color = rng.integers(0, 200, size=3).tolist()
            img = np.full((self.height, self.width, 3), color, dtype=np.uint8)
        elif choice == 1:
            # Vertical gradient
            top = rng.integers(0, 200, size=3)
            bot = rng.integers(0, 200, size=3)
            rows = np.linspace(0, 1, self.height).reshape(-1, 1, 1)
            img = ((1 - rows) * top + rows * bot).astype(np.uint8)
            img = np.broadcast_to(img, (self.height, self.width, 3)).copy()
        else:
            # Random noise
            img = rng.integers(0, 150, size=(self.height, self.width, 3), dtype=np.uint8)
        return img

    def _render_gate(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> GateAnnotation | None:
        """Render a single gate onto the image and return annotation."""
        # Random pose
        distance = rng.uniform(*self._dist_range)
        offset_x = rng.uniform(*self._offset_range)
        offset_y = rng.uniform(*self._offset_range)
        rot_y = rng.uniform(-self._rot_range, self._rot_range)
        rot_x = rng.uniform(-self._rot_range * 0.3, self._rot_range * 0.3)

        # Build rotation matrix
        rvec = np.array([rot_x, rot_y, 0.0])
        R, _ = cv2.Rodrigues(rvec)

        # Translation
        tvec = np.array([offset_x, offset_y, distance])

        # Project corners
        K = self.intrinsics.matrix
        pts_3d = (R @ self._gate_3d.T).T + tvec
        # Check all points in front of camera
        if np.any(pts_3d[:, 2] <= 0.1):
            return None

        pts_2d = np.zeros((4, 2))
        for i in range(4):
            p = K @ pts_3d[i]
            pts_2d[i] = p[:2] / p[2]

        # Check all corners in frame
        margin = -20
        if (np.any(pts_2d[:, 0] < margin) or np.any(pts_2d[:, 0] >= self.width - margin)
                or np.any(pts_2d[:, 1] < margin) or np.any(pts_2d[:, 1] >= self.height - margin)):
            return None

        # Draw gate frame (thick colored lines)
        color = (
            int(rng.integers(0, 50)),
            int(rng.integers(0, 50)),
            int(rng.integers(180, 255)),
        )  # reddish
        pts_int = pts_2d.astype(np.int32)
        thickness = max(2, int(30 / distance))
        for j in range(4):
            cv2.line(image, tuple(pts_int[j]), tuple(pts_int[(j + 1) % 4]), color, thickness)

        # Build annotation
        x_min, y_min = pts_2d.min(axis=0).astype(int)
        x_max, y_max = pts_2d.max(axis=0).astype(int)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        center = (int(pts_2d[:, 0].mean()), int(pts_2d[:, 1].mean()))

        return GateAnnotation(
            bbox=bbox,
            corners=pts_2d,
            center=center,
            distance=distance,
        )

    def _augment(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply random augmentations."""
        # Brightness/contrast
        alpha = rng.uniform(0.7, 1.3)
        beta = rng.uniform(-30, 30)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Gaussian noise
        if rng.random() < 0.5:
            noise = rng.normal(0, 5, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return image
