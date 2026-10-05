#!/usr/bin/env bash
# Follow-up GPU experiments (same recipe as the main runs; nothing is tuned on test).
#   A. split robustness: two more train/val partitions of the base classes (seeds 7, 123), 3 training seeds each.
#      The test classes (< 20 images) are the same under every partition.
#   B. ResNet-50: backbone lr {1e-5, 3e-5} chosen on validation (seed 42), then seeds 43/44 at the chosen lr;
#      only the chosen lr is ever evaluated on test.
#   PYTHON=.venv/bin/python bash scripts/run_extra.sh [A|B|all] [WORKERS]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
PART=${1:-all}
W=${2:-12}

train_eval() {  # out_dir result_json [train args...]
  local out=$1 res=$2; shift 2
  [ -f "$out/done.json" ] || $PY src/train.py --out "$out" --workers "$W" --resume "$@"
  mkdir -p "$(dirname "$res")"
  [ -f "$res" ] || $PY src/eval.py --ckpt "$out/best.pt" --out "$res"
}

if [ "$PART" = A ] || [ "$PART" = all ]; then
  for SP in 7 123; do
    [ -f "results/split$SP/frozen_val.json" ] || $PY scripts/frozen_val.py --splits "splits_s$SP.json" --out "results/split$SP/frozen_val.json"
    for S in 42 43 44; do
      train_eval "runs/split$SP/seed$S" "results/split$SP/seed${S}_test.json" --splits "splits_s$SP.json" --seed "$S"
    done
  done
fi

if [ "$PART" = B ] || [ "$PART" = all ]; then
  mkdir -p results/r50
  [ -f results/r50/frozen_val.json ] || $PY scripts/frozen_val.py --backbone resnet50 --out results/r50/frozen_val.json
  [ -f results/r50/frozen_test.json ] || $PY src/eval.py --frozen --backbone resnet50 --out results/r50/frozen_test.json
  for LR in 1e-5 3e-5; do   # selection runs: train only, no test eval
    [ -f "runs/r50/lr${LR}_seed42/done.json" ] || \
      $PY src/train.py --out "runs/r50/lr${LR}_seed42" --workers "$W" --resume --backbone resnet50 --seed 42 --lr-backbone "$LR"
  done
  BEST=$($PY -c "import json; r={lr: json.load(open(f'runs/r50/lr{lr}_seed42/done.json'))['best_val'] for lr in ['1e-5','3e-5']}; print(max(r, key=r.get))")
  echo "ResNet-50: chosen backbone lr on validation = $BEST" | tee results/r50/chosen_lr.txt
  train_eval "runs/r50/lr${BEST}_seed42" "results/r50/seed42_test.json" --backbone resnet50 --seed 42 --lr-backbone "$BEST"
  for S in 43 44; do
    train_eval "runs/r50/lr${BEST}_seed$S" "results/r50/seed${S}_test.json" --backbone resnet50 --seed "$S" --lr-backbone "$BEST"
  done
fi
echo "run_extra.sh $PART done"
