#!/usr/bin/env bash
# Continuous-archive scan with the corrected seed-42 SpecUNet at its
# validation-locked operating point (0.25, 430 s), then the same Nakamura
# scoring and operating-point sweep as the historical scan. Outputs go to
# results/archive_scan/grouped_v1_seed42/; the archive itself is only read.
#
#   WAIT_FOR=<file containing "runner exit"> PY=... bash scripts/run_archive_grouped.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
TAG=grouped_v1_seed42
MODEL=runs/unet_lunar_grouped_v1/best.pt
LOGS=runs/logs_rerun
mkdir -p "$LOGS"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd -W 2>/dev/null || pwd)"
export PLANETSEIS_ARCHIVE=${PLANETSEIS_ARCHIVE:-'C:\planetseis_archive'}
"$PY" -c "import os, planetseis.config as c; \
assert os.path.samefile(c.PROJECT_ROOT, os.getcwd()), c.PROJECT_ROOT"

# One CUDA process at a time: wait for the GPU job before starting.
if [ -n "${WAIT_FOR:-}" ]; then
  echo "waiting for GPU job ($WAIT_FOR)"
  until grep -q "runner exit" "$WAIT_FOR" 2>/dev/null; do sleep 60; done
fi

echo "== scan -> results/archive_scan/$TAG"
"$PY" scripts/scan_archive.py --model "$MODEL" --stations S12 S15 S16 \
  --threshold 0.25 --min-dur 430 --tag "$TAG" --save-curves >> "$LOGS/archive_scan.log" 2>&1
echo "== score"
"$PY" scripts/score_archive_vs_nakamura.py --stations S12 S15 S16 --tag "$TAG" \
  >> "$LOGS/archive_score.log" 2>&1
echo "== sweep"
"$PY" scripts/sweep_archive.py --tag "$TAG" --stations S12 S15 S16 >> "$LOGS/archive_sweep.log" 2>&1
echo "archive rerun complete"
