#!/usr/bin/env bash
# Positive-unlabeled ablation on lunar_grouped_v1: retrain SpecUNet from
# scratch with the injection noise pool screened against the full Nakamura
# catalog, five seeds, then evaluate each at its own validation-locked
# operating point. The unscreened five seeds already exist; only the screen
# differs. Safe to rerun: a run with last.pt resumes (finished runs exit at
# once) and an existing evaluation is skipped, never overwritten.
#
#   PY=/path/to/python bash scripts/run_screened_grouped.sh
#   SEEDS="3 4" PY=... bash scripts/run_screened_grouped.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
DATA=data/cache/lunar_grouped_v1
SCREEN=data/cache/lunar_grouped_v1_nakamura_screen.json
SEEDS=${SEEDS:-"42 1 2 3 4"}
LOGS=runs/logs_screened
mkdir -p "$LOGS"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd -W 2>/dev/null || pwd)"

# The interpreter may carry an editable install of another checkout; refuse
# to run unless this checkout's package is the one imported.
"$PY" -c "import os, planetseis.config as c; \
assert os.path.samefile(c.PROJECT_ROOT, os.getcwd()), c.PROJECT_ROOT"

suffix() { [ "$1" = 42 ] && echo "" || echo "_s$1"; }
resume_flag() { [ -f "$1/last.pt" ] && echo "--resume" || echo ""; }

for seed in $SEEDS; do
  out=runs/unet_lunar_grouped_v1_screened$(suffix "$seed")
  echo "== screened SpecUNet seed $seed -> $out"
  "$PY" scripts/train_unet.py --body lunar --epochs 30 --epoch-len 6400 \
    --batch-size 32 --patience 8 --seed "$seed" \
    --screen-nakamura --screen-file "$SCREEN" \
    --data-dir "$DATA" --out-dir "$out" --device cuda $(resume_flag "$out") \
    >> "$LOGS/train_s$seed.log" 2>&1
done
for seed in $SEEDS; do
  out=runs/unet_lunar_grouped_v1_screened$(suffix "$seed")
  result=results/lunar_grouped_v1_screened_seed${seed}_unet.json
  if [ -f "$result" ]; then echo "== $result exists, skipped"; continue; fi
  echo "== evaluate seed $seed -> $result"
  "$PY" scripts/evaluate_grouped.py --data-dir "$DATA" --unet "$out/best.pt" \
    --output "$result" --skip-baseline --device cuda >> "$LOGS/eval_s$seed.log" 2>&1
done
echo "screened ablation complete"
