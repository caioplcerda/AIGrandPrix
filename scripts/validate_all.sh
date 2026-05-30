#!/usr/bin/env bash
# Consolidated validation gate — runs the full verified suite in one command.
# Covers all 3 competition pillars (perception, control+planning, integration).
#
# Usage:  bash scripts/validate_all.sh
# Exit 0 = all green.  Uses the system Python 3.13 (has pymavlink + cv2 + torch).

set -uo pipefail
PY="${PY:-/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13}"
WEIGHTS="datasets/gate_yolo_mps/runs/gate_detector2/weights/best.pt"
cd "$(dirname "$0")/.."

fail=0
run() {  # run <label> <cmd...>
  local label="$1"; shift
  printf '\n=== %s ===\n' "$label"
  if "$@"; then printf '  [PASS] %s\n' "$label"
  else printf '  [FAIL] %s\n' "$label"; fail=1; fi
}

run "unit tests (243)"      "$PY" -m pytest tests/ -q
run "VQ1 e2e (5 gates)"     "$PY" scripts/test_e2e_mock.py
run "godtier battery (8)"   "$PY" scripts/test_godtier_courses.py
run "perception robustness" "$PY" scripts/test_yolo_robustness.py --weights "$WEIGHTS" --n 60

printf '\n========================================\n'
if [ "$fail" -eq 0 ]; then printf 'ALL VALIDATIONS GREEN\n'; else printf 'SOME VALIDATIONS FAILED\n'; fi
printf '========================================\n'
exit "$fail"
