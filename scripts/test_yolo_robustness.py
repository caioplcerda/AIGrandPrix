"""YOLO gate-detector robustness stress eval — failure-envelope mapping.

The baseline validation (test_yolo_model.py) tests the SAME synthetic distribution
the model trained on (P=R=1.0). This pushes BEYOND training augmentation to find where
the detector breaks: heavy motion blur (48 m/s flight), aggressive JPEG (UDP stream),
extreme lighting, occlusion. Reports recall vs degradation so we know the safe envelope.

Usage:
    python scripts/test_yolo_robustness.py --weights <best.pt> --n 80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def gen_clean_gate_samples(n, seed=777):
    """Generate n images each containing exactly one gate, WITHOUT augmentation."""
    from aigrandprix.perception.data_generator import GateDataGenerator
    gen = GateDataGenerator()
    rng = np.random.default_rng(seed)
    out = []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        img = gen._random_background(rng)
        ann = gen._render_gate(img, rng)
        if ann is not None:
            out.append((img.copy(), ann))
    return out


def _motion_blur(img, ksize, angle=30.0):
    if ksize < 3:
        return img
    k = np.zeros((ksize, ksize)); k[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle, 1)
    k = cv2.warpAffine(k, M, (ksize, ksize)); k /= k.sum() + 1e-8
    return cv2.filter2D(img, -1, k)


def _jpeg(img, quality):
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _brightness(img, alpha, beta=0.0):
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _occlude(img, ann, frac):
    """Cover `frac` of the gate bbox (top-down) with black (simulates partial occlusion)."""
    x, y, w, h = [int(v) for v in ann.bbox]  # (x, y, w, h)
    if w <= 0 or h <= 0:
        return img
    out = img.copy()
    hh = max(1, int(h * frac))
    out[y:y + hh, x:x + w] = 0
    return out


def run_detector(model, image, conf=0.3):
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    res = model(image, verbose=False, conf=conf, device=device)
    best = None
    for r in res:
        for b in r.boxes:
            c = float(b.conf[0])
            if best is None or c > best:
                best = c
    return best  # None = no detection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="datasets/gate_yolo_mps/runs/gate_detector2/weights/best.pt")
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    print(f"Generating {args.n} clean single-gate samples...")
    samples = gen_clean_gate_samples(args.n)
    print(f"Got {len(samples)} gate samples\n")

    def recall(transform):
        hits = 0
        for img, ann in samples:
            d = run_detector(model, transform(img, ann))
            if d is not None:
                hits += 1
        return hits / len(samples) if samples else 0.0

    print("=" * 60)
    print("  YOLO ROBUSTNESS — recall vs degradation")
    print("=" * 60)

    base = recall(lambda im, a: im)
    print(f"  clean (no degradation)        recall = {base:.3f}")

    print("  -- motion blur (px kernel) — 48 m/s flight --")
    for k in [5, 11, 17, 23, 31]:
        r = recall(lambda im, a, k=k: _motion_blur(im, k))
        print(f"    blur k={k:2d}px                  recall = {r:.3f}")

    print("  -- JPEG quality — UDP stream --")
    for q in [40, 25, 15, 8]:
        r = recall(lambda im, a, q=q: _jpeg(im, q))
        print(f"    jpeg q={q:2d}                     recall = {r:.3f}")

    print("  -- brightness (alpha) — lighting extremes --")
    for al in [0.3, 0.2, 2.0, 2.8]:
        r = recall(lambda im, a, al=al: _brightness(im, al))
        print(f"    brightness x{al:.1f}              recall = {r:.3f}")

    print("  -- partial occlusion (bbox fraction covered) --")
    for f in [0.25, 0.5, 0.75]:
        r = recall(lambda im, a, f=f: _occlude(im, a, f))
        print(f"    occlude {int(f*100):2d}%                  recall = {r:.3f}")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
