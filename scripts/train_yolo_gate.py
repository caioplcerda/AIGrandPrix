"""Generate synthetic gate dataset and train YOLOv8n for DCL gate detection.

Usage:
    python scripts/train_yolo_gate.py --n_train 5000 --n_val 500 --epochs 50
    python scripts/train_yolo_gate.py --n_train 5000 --n_val 500 --epochs 50 --output runs/gate_v1

After training, weights at: <output>/weights/best.pt
Set in config: cnn: {backend: yolo, model_path: runs/gate_v1/weights/best.pt}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def generate_yolo_dataset(
    n_images: int,
    output_dir: Path,
    seed: int,
    split: str = "train",
) -> None:
    """Generate images + YOLO-format label files."""
    from aigrandprix.perception.data_generator import GateDataGenerator

    gen = GateDataGenerator()
    rng = np.random.default_rng(seed)

    img_dir = output_dir / "images" / split
    lbl_dir = output_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    W, H = gen.width, gen.height

    for i in range(n_images):
        sample = gen.generate_sample(rng=rng)
        import cv2
        fname = f"{split}_{i:06d}.jpg"
        cv2.imwrite(str(img_dir / fname), sample.image, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # YOLO format: one line per bbox → class cx cy w h (all normalized 0-1)
        lines = []
        for ann in sample.annotations:
            x, y, w, h = ann.bbox
            if w <= 0 or h <= 0:
                continue
            cx_n = (x + w / 2) / W
            cy_n = (y + h / 2) / H
            w_n = w / W
            h_n = h / H
            # clamp
            cx_n = max(0.0, min(1.0, cx_n))
            cy_n = max(0.0, min(1.0, cy_n))
            w_n = max(0.001, min(1.0, w_n))
            h_n = max(0.001, min(1.0, h_n))
            lines.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

        lbl_file = lbl_dir / fname.replace(".jpg", ".txt")
        lbl_file.write_text("\n".join(lines))

        if (i + 1) % 500 == 0:
            print(f"  [{split}] {i + 1}/{n_images} generated")


def write_data_yaml(output_dir: Path) -> Path:
    import yaml
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["gate"],
    }
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml.dump(data, default_flow_style=False))
    return yaml_path


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)

    print(f"Generating {args.n_train} train images…")
    generate_yolo_dataset(args.n_train, output_dir, seed=42, split="train")

    print(f"Generating {args.n_val} val images…")
    generate_yolo_dataset(args.n_val, output_dir, seed=999, split="val")

    try:
        import yaml
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
        import yaml

    yaml_path = write_data_yaml(output_dir)
    print(f"data.yaml → {yaml_path}")

    print(f"\nTraining YOLOv8n for {args.epochs} epochs…")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics>=8.0")
        sys.exit(1)

    model = YOLO("yolov8n.pt")  # downloads pretrained weights on first run
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        project=str(output_dir / "runs"),
        name="gate_detector",
        device=args.device,
        workers=args.workers,
        patience=15,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        save=True,
        verbose=True,
    )

    best_weights = output_dir / "runs" / "gate_detector" / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights: {best_weights}")
    print(f"\nTo use in run_vq1.py, add --cnn_model {best_weights}")
    print(f"Or set in config: cnn: {{backend: yolo, model_path: {best_weights}}}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8n gate detector")
    p.add_argument("--n_train", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="", help="cuda device or '' for auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", default="datasets/gate_yolo")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
