#!/usr/bin/env bash
# Phase 2 on the GPU VM: data -> verify -> smoke -> frozen baseline -> 3 seeds -> eval -> aggregate.
# Idempotent: finished seeds (runs/seedN/done.json) are skipped, interrupted ones resume from last.pt.
#   tmux new -s run 'bash scripts/run_all.sh 12 2>&1 | tee run_all.log'
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
WORKERS=${1:-12}

$PY scripts/download_data.py          # pinned revision; asserts sha256 == HF LFS hash
$PY scripts/prep_thumbs.py
$PY src/data.py --verify              # must match the committed data_manifest.json + splits.json

mkdir -p results
$PY - <<'EOF'
import torch; print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
                   "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
assert torch.cuda.is_available(), "no CUDA device: refusing to run the full schedule on CPU"
EOF

# smoke: 500 episodes; also prints img/s so a dataloader bottleneck shows up before the long runs
$PY src/train.py --out runs/smoke --episodes 500 --val-every 50 --val-episodes 100 --workers "$WORKERS"

$PY src/eval.py --frozen --out results/frozen_test.json

for SEED in 42 43 44; do
  PYTHON=$PY bash scripts/run_seed.sh "$SEED" "$WORKERS"
done

$PY src/eval.py --aggregate results/seed42_test.json results/seed43_test.json results/seed44_test.json \
    --frozen-file results/frozen_test.json --out results/summary.json
echo "ALL DONE. Copy back: runs/ results/ (best.pt x3 are ~45 MB each; last.pt can be skipped)"
