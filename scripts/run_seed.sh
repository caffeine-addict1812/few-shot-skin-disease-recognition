#!/usr/bin/env bash
# One seed: train (resumable, skipped if finished) then test-eval on the fixed test episodes.
#   PYTHON=.venv/bin/python bash scripts/run_seed.sh 42 [WORKERS]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
SEED=$1
WORKERS=${2:-16}

mkdir -p results
if [ ! -f "runs/seed${SEED}/done.json" ]; then
  $PY src/train.py --out "runs/seed${SEED}" --seed "$SEED" --workers "$WORKERS" --resume
fi
$PY src/eval.py --ckpt "runs/seed${SEED}/best.pt" --out "results/seed${SEED}_test.json"
