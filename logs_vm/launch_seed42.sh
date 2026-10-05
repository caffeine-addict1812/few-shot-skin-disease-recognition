#!/usr/bin/env bash
cd /root/DL
export PYTHON=.venv/bin/python
( $PYTHON src/eval.py --frozen --out results/frozen_test.json \
  && bash scripts/run_seed.sh 42 16 ) 2>&1 | tee logs/seed42.log
echo "EXIT=${PIPESTATUS[0]}" >> logs/seed42.log
