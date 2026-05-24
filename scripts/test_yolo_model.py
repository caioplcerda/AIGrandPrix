"""Validate trained YOLOv8n gate detector on held-out synthetic images.

Usage:
    python scripts/test_yolo_model.py --weights datasets/gate_yolo_mps/runs/gate_detector/weights/best.pt
    python scripts/test_yolo_model.py  # auto-finds best.pt in datasets/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def find_weights() -> Path | None:
    for p in sorted(Path("datasets").rglob("best.pt")):
        return p
    for p in sorted(Path("datasets").rglob("last.pt")):
        return p
    return None


def generate_test_images(n: int = 50, seed: int = 12345) -> list[tuple[np.ndarray, bool]]:
    """Generate held-out test frames (image, has_gate)."""
    from aigrandprix.perception.data_generator import GateDataGenerator

    gen = GateDataGenerator()
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        s = gen.generate_sample(rng=rng)
        has_gate = len(s.annotations) > 0
        samples.append((s.image, has_gate, s.annotations))
    return samples


def run_yolo_detection(model_path: Path, image: np.ndarray) -> list[dict]:
    """Run YOLO inference, return list of {bbox, conf, cx_px, cy_px, dist}."""
    from ultralytics import YOLO
    import torch

    if not hasattr(run_yolo_detection, "_model"):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        run_yolo_detection._model = YOLO(str(model_path))
        run_yolo_detection._device = device

    model = run_yolo_detection._model
    results = model(image, verbose=False, conf=0.3, device=run_yolo_detection._device)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            bw = x2 - x1
            bh = y2 - y1
            # Distance estimate: gate outer 2.7m, fx=320
            dist = (2.7 * 320) / max(bw, bh) if max(bw, bh) > 1 else 999.0
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "cx_px": (x1 + x2) / 2,
                "cy_px": (y1 + y2) / 2,
                "dist": dist,
            })
    return detections


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="Path to best.pt")
    ap.add_argument("--n_images", type=int, default=100, help="Number of test images")
    ap.add_argument("--conf_thresh", type=float, default=0.4)
    ap.add_argument("--save_failures", action="store_true", help="Save FP/FN images to /tmp/")
    args = ap.parse_args()

    weights = Path(args.weights) if args.weights else find_weights()
    if weights is None or not weights.exists():
        print(f"ERROR: No weights found. Train first with scripts/train_yolo_gate.py")
        return 1

    print(f"Weights: {weights}")
    print(f"Generating {args.n_images} test images...")
    samples = generate_test_images(n=args.n_images)

    tp = fp = fn = tn = 0
    dist_errors = []
    inference_times = []

    for i, (image, has_gate, annotations) in enumerate(samples):
        t0 = time.perf_counter()
        dets = run_yolo_detection(weights, image)
        dt = time.perf_counter() - t0
        inference_times.append(dt * 1000)

        # Keep only dets above conf threshold
        dets = [d for d in dets if d["conf"] >= args.conf_thresh]

        detected = len(dets) > 0

        if has_gate and detected:
            tp += 1
            # Distance error (first annotation vs best detection)
            true_dist = annotations[0].distance
            pred_dist = dets[0]["dist"]
            dist_errors.append(abs(pred_dist - true_dist))
        elif has_gate and not detected:
            fn += 1
            if args.save_failures:
                cv2.imwrite(f"/tmp/fn_{i:04d}.jpg", image)
        elif not has_gate and detected:
            fp += 1
            if args.save_failures:
                # Draw detections on false positive
                vis = image.copy()
                for d in dets:
                    x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imwrite(f"/tmp/fp_{i:04d}.jpg", vis)
        else:
            tn += 1

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    avg_dist_err = np.mean(dist_errors) if dist_errors else 0.0
    avg_infer_ms = np.mean(inference_times[1:]) if len(inference_times) > 1 else inference_times[0]  # skip first (model warmup)

    print()
    print("=" * 55)
    print(f"  YOLOv8n Gate Detector — Validation Results")
    print("=" * 55)
    print(f"  Weights       : {weights.name}")
    print(f"  Test images   : {total}")
    print(f"  Conf threshold: {args.conf_thresh}")
    print()
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision     : {precision:.3f}  ({tp}/{tp+fp})")
    print(f"  Recall        : {recall:.3f}  ({tp}/{tp+fn})")
    print(f"  F1 score      : {f1:.3f}")
    print(f"  Accuracy      : {accuracy:.3f}")
    print(f"  Dist error    : {avg_dist_err:.2f} m (avg, on TP)")
    print(f"  Inference     : {avg_infer_ms:.1f} ms avg (excl. warmup)")
    print("=" * 55)

    # Pass criteria
    ok = precision >= 0.85 and recall >= 0.80 and f1 >= 0.82
    if ok:
        print(f"  RESULT: PASS ✓  (P≥0.85, R≥0.80, F1≥0.82)")
        return 0
    else:
        print(f"  RESULT: FAIL ✗  (need P≥0.85, R≥0.80, F1≥0.82)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
