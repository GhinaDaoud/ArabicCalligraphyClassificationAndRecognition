#!/bin/bash
# Orchestration: run all 3 zero-shot inferences + score.
# Usage:  bash calligraphy_ocr/scripts/run_all.sh
# Logs to /tmp/cal_ocr_run.log via tee.

set -u
PYBIN=/home/ahmad/anaconda3/envs/calligraphy_ocr/bin/python
ROOT=/home/ahmad/Desktop/beebot/693

echo "=== START $(date '+%F %T') ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

for model in sherif-handwritten-v3 qari-v0.3 qari-v0.2.2.1; do
  echo "=== INFER $model $(date '+%T') ==="
  $PYBIN $ROOT/calligraphy_ocr/scripts/run_zero_shot.py --model $model
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== $model FAILED rc=$rc ==="
  else
    echo "=== $model DONE ==="
  fi
done

echo "=== SCORE $(date '+%T') ==="
$PYBIN $ROOT/calligraphy_ocr/scripts/score.py
echo "=== ALL_DONE $(date '+%T') ==="
