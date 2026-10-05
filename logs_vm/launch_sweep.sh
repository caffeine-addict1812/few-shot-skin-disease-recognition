#!/usr/bin/env bash
# VM-specific: the container image ships CUDA libs (610) that mismatch the host kernel module (565).
export LD_LIBRARY_PATH=/root/nvlibs:$LD_LIBRARY_PATH
export PYTHON=/root/DL/.venv/bin/python
cd /root/DL && bash scripts/sweep.sh 16 2>&1 | tee logs/sweep.log
echo "EXIT=${PIPESTATUS[0]}" >> logs/sweep.log
