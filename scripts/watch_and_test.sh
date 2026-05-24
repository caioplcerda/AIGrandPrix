#!/bin/bash
# Wait for YOLO training to finish, then auto-run test_yolo_model.py
# Usage: bash scripts/watch_and_test.sh
WEIGHTS="datasets/gate_yolo_mps/runs/gate_detector/weights/best.pt"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"

echo "Watching for $WEIGHTS ..."
until [ -f "$WEIGHTS" ]; do sleep 30; done

echo "Training complete. Running model validation..."
$PYTHON scripts/test_yolo_model.py --weights "$WEIGHTS" --n_images 200 --save_failures
echo "Validation done. Exit: $?"
