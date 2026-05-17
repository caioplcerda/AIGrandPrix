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
    """Generates synthetic training images with gate annotations.

    Configured by default for VADR-TS-002 DCL competition:
      - Image: 640×360 px
      - Gate outer: 2.7×2.7 m (frame border 0.6 m, inner opening 1.5×1.5 m)
      - Camera: fx=fy=320, cx=320, cy=180, VFoV=90°
      - Camera tilt: +20° up from body (camera looks slightly upward)
      - Gate color: dark blue (DCL visual style, BGR ~(180, 30, 20))
    """

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        # VADR-TS-002: 640×360
        self.width = config.get("image_width", 640)
        self.height = config.get("image_height", 360)
        # VADR-TS-002: outer gate 2.7m, inner 1.5m
        self.gate_outer = config.get("gate_outer_width", 2.7)
        self.gate_inner = config.get("gate_inner_width", 1.5)
        # Backward-compat alias
        self.gate_width = self.gate_inner
        self.gate_height = self.gate_inner

        # VADR-TS-002 camera intrinsics: fx=fy=320, cx=320, cy=180
        cam_cfg = config.get("camera", {})
        if cam_cfg:
            self.intrinsics = CameraIntrinsics.from_config(cam_cfg)
        else:
            self.intrinsics = CameraIntrinsics(
                fx=config.get("fx", 320.0),
                fy=config.get("fy", 320.0),
                cx=config.get("cx", float(self.width) / 2),
                cy=config.get("cy", float(self.height) / 2),
            )

        # Camera tilt: +20° upward (camera looks 20° above body X axis)
        tilt_deg = config.get("camera_tilt_deg", 20.0)
        self._cam_tilt_rad = np.radians(tilt_deg)

        # Outer gate corners (4 corners of 2.7×2.7 m square, front face)
        ho = self.gate_outer / 2.0
        self._outer_3d = np.array([
            [-ho,  ho, 0],
            [ ho,  ho, 0],
            [ ho, -ho, 0],
            [-ho, -ho, 0],
        ], dtype=np.float64)

        # Inner gate corners (1.5×1.5 m opening)
        hi = self.gate_inner / 2.0
        self._inner_3d = np.array([
            [-hi,  hi, 0],
            [ hi,  hi, 0],
            [ hi, -hi, 0],
            [-hi, -hi, 0],
        ], dtype=np.float64)

        # Backward-compat
        self._gate_3d = self._outer_3d

        # Randomization ranges
        self._dist_range = (2.0, 15.0)
        self._offset_range = (-2.5, 2.5)
        self._rot_range = np.radians(45)

        # Tilt rotation matrix: rotate camera frame down by tilt angle
        # (equivalent to camera looking up by tilt_deg)
        ct = np.cos(self._cam_tilt_rad)
        st = np.sin(self._cam_tilt_rad)
        # Rotate around X axis by -tilt (gate moves up in image)
        self._R_tilt = np.array([
            [1,   0,   0],
            [0,  ct, -st],
            [0,  st,  ct],
        ], dtype=np.float64)

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
        """Render a single gate (outer frame + inner opening) onto the image.

        Uses VADR-TS-002 gate geometry (2.7m outer, 1.5m inner) and
        applies camera tilt of +20° upward.
        """
        # Random pose in front of camera
        distance = rng.uniform(*self._dist_range)
        offset_x = rng.uniform(*self._offset_range)
        offset_y = rng.uniform(*self._offset_range)
        rot_y = rng.uniform(-self._rot_range, self._rot_range)
        rot_x = rng.uniform(-self._rot_range * 0.3, self._rot_range * 0.3)

        rvec = np.array([rot_x, rot_y, 0.0])
        R, _ = cv2.Rodrigues(rvec)

        tvec = np.array([offset_x, offset_y, distance])

        K = self.intrinsics.matrix

        def _project(pts_3d_local: np.ndarray) -> np.ndarray | None:
            pts = (R @ pts_3d_local.T).T + tvec
            # Apply camera tilt: rotate 3D points by tilt matrix
            pts = (self._R_tilt @ pts.T).T
            if np.any(pts[:, 2] <= 0.1):
                return None
            px = np.zeros((len(pts), 2))
            for i in range(len(pts)):
                p = K @ pts[i]
                px[i] = p[:2] / p[2]
            return px

        outer_2d = _project(self._outer_3d)
        inner_2d = _project(self._inner_3d)
        if outer_2d is None or inner_2d is None:
            return None

        # Check outer corners are visible (with margin)
        margin = -30
        all_pts = outer_2d
        if (np.any(all_pts[:, 0] < margin) or np.any(all_pts[:, 0] >= self.width - margin)
                or np.any(all_pts[:, 1] < margin) or np.any(all_pts[:, 1] >= self.height - margin)):
            return None

        # Gate color: dark blue (DCL style) with slight variation
        blue_b = int(rng.integers(150, 210))
        blue_g = int(rng.integers(20, 50))
        blue_r = int(rng.integers(10, 40))
        gate_color = (blue_b, blue_g, blue_r)  # BGR

        outer_int = outer_2d.astype(np.int32)
        inner_int = inner_2d.astype(np.int32)
        thickness = max(3, int(25.0 / distance))

        # Fill outer polygon (gate frame material)
        cv2.fillPoly(image, [outer_int], gate_color)
        # Fill inner polygon dark (the hole you fly through)
        cv2.fillPoly(image, [inner_int], (15, 15, 15))
        # Bright highlight edges on outer frame
        bright = (min(255, blue_b + 60), min(255, blue_g + 30), min(255, blue_r + 20))
        for j in range(4):
            cv2.line(image, tuple(outer_int[j]), tuple(outer_int[(j + 1) % 4]), bright, thickness)

        # Annotation: bbox covers outer gate
        x_min = int(max(0, outer_2d[:, 0].min()))
        y_min = int(max(0, outer_2d[:, 1].min()))
        x_max = int(min(self.width, outer_2d[:, 0].max()))
        y_max = int(min(self.height, outer_2d[:, 1].max()))
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        center = (int(inner_2d[:, 0].mean()), int(inner_2d[:, 1].mean()))

        return GateAnnotation(
            bbox=bbox,
            corners=outer_2d,
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
