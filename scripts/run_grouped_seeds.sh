#!/usr/bin/env bash
# Train SeisCNN and SpecUNet from scratch on lunar_grouped_v1 for 5 seeds,
# one GPU job at a time. Safe to rerun: a run directory that already holds
# last.pt is resumed (finished runs exit immediately), never overwritten.
#
#   bash scripts/run_grouped_seeds.sh            # all seeds, both models
#   SEEDS="3 4" bash scripts/run_grouped_seeds.sh
#
# Recipe matches the historical runs: SeisCNN 60 epochs; SpecUNet 30 epochs,
# 6400 samples/epoch, batch 32, patience 8, no Nakamura screen, no warm start.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
DATA=data/cache/lunar_grouped_v1
SEEDS=${SEEDS:-"42 1 2 3 4"}
LOGS=runs/logs_grouped
mkdir -p "$LOGS"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1

suffix() { [ "$1" = 42 ] && echo "" || echo "_s$1"; }
resume_flag() { [ -f "$1/last.pt" ] && echo "--resume" || echo ""; }

for seed in $SEEDS; do
  out=runs/lunar_grouped_v1$(suffix "$seed")
  echo "== SeisCNN seed $seed -> $out"
  "$PY" -m planetseis.train --body lunar --epochs 60 --seed "$seed" \
    --data-dir "$DATA" --out-dir "$out" --device cuda $(resume_flag "$out") \
    >> "$LOGS/cnn_s$seed.log" 2>&1
done
for seed in $SEEDS; do
  out=runs/unet_lunar_grouped_v1$(suffix "$seed")
  echo "== SpecUNet seed $seed -> $out"
  "$PY" scripts/train_unet.py --body lunar --epochs 30 --epoch-len 6400 \
    --batch-size 32 --patience 8 --seed "$seed" \
    --data-dir "$DATA" --out-dir "$out" --device cuda $(resume_flag "$out") \
    >> "$LOGS/unet_s$seed.log" 2>&1
done
echo "all grouped runs complete"
