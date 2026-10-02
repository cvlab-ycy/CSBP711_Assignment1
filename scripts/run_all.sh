#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

DEVICE="${1:-auto}"
case "$DEVICE" in
  auto|cuda|cpu) ;;
  *) echo "Usage: bash scripts/run_all.sh [auto|cuda|cpu]" >&2; exit 2 ;;
esac

PYTHON_COMMAND="${PYTHON_COMMAND:-python}"

"$PYTHON_COMMAND" -m pytest -q
"$PYTHON_COMMAND" src/study.py all --device "$DEVICE"
"$PYTHON_COMMAND" scripts/verify_results.py
"$PYTHON_COMMAND" src/training_review.py

echo "End-to-end run and independent result verification completed."

