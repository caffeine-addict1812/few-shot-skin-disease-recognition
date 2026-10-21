#!/usr/bin/env bash
# Short recipe sweep, selected on VALIDATION classes only (the test classes are never touched).
# Motivation: the planned 20k-episode / lr 1e-4 schedule overfits the 101 base classes (see runs/_evidence/).
#   PYTHON=.venv/bin/python bash scripts/sweep.sh [WORKERS]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
W=${1:-16}
COMMON="--episodes 4000 --val-every 100 --workers $W"      # 1,000 steps, 10 val sweeps of 600 episodes

run() {  # name, then extra train.py args
  local name=$1; shift
  [ -f "runs/sweep/$name/done.json" ] && { echo "skip $name"; return; }
  echo "=== $name: $*"
  rm -rf "runs/sweep/$name"
  $PY src/train.py --out "runs/sweep/$name" --seed 42 $COMMON "$@" 2>&1 | grep --line-buffered -E "\[val\]|done:|Error|error"
}

run lr1e-5        --lr-backbone 1e-5
run lr3e-5        --lr-backbone 3e-5
run lr1e-5_fbn    --lr-backbone 1e-5 --freeze-bn
run w15_lr1e-5    --lr-backbone 1e-5 --n-way 15 --n-query 5
run w15_lr3e-5    --lr-backbone 3e-5 --n-way 15 --n-query 5

$PY scripts/summarize_runs.py runs/sweep/* runs/_evidence/*
