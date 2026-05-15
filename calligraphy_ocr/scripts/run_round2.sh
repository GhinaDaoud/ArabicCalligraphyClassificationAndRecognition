#!/bin/bash
# Round 2 — full 49-image sweep for the 3 candidates discovered in
# the post-sherif1313 search. Run after smoke tests pass.
set -u
PYBIN=/home/ahmad/anaconda3/envs/calligraphy_ocr/bin/python
ROOT=/home/ahmad/Desktop/beebot/693

echo "=== START $(date '+%F %T') ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

for model in baseer-nakba ain-7b ketaba-lora; do
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
