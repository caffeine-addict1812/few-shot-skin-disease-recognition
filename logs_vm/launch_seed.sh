#!/usr/bin/env bash
# usage: launch_seed.sh SEED [SEED...]   (VM-specific CUDA lib override, see launch_sweep.sh)
export LD_LIBRARY_PATH=/root/nvlibs:$LD_LIBRARY_PATH
export PYTHON=/root/DL/.venv/bin/python
cd /root/DL
for SEED in "$@"; do
  bash scripts/run_seed.sh "$SEED" 16 2>&1 | tee -a logs/seed${SEED}.log
  echo "EXIT=${PIPESTATUS[0]}" >> logs/seed${SEED}.log
done
