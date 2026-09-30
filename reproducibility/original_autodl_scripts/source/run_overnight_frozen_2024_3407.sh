#!/bin/bash
export PYTHONPATH=/root/autodl-tmp/ssr_yolorace_exp/code/YOLO-RACE:${PYTHONPATH:-}
set -euo pipefail
PY=/root/autodl-tmp/ssr_yolorace_exp/venv/bin/python
ROOT=/root/autodl-tmp/ssr_yolorace_exp/code/SSR_release_clean/SSR-VQA

echo "========== PREFLIGHT =========="
df -h /root/autodl-tmp
FREE_KB=$(df --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')
if [ "$FREE_KB" -lt 4194304 ]; then
  echo "ERROR: less than 4 GiB free on /root/autodl-tmp. Free space before overnight run."
  exit 2
fi

for S in 2024 3407
do
  echo "========== FROZEN SEED $S =========="
  "$PY" /root/autodl-tmp/ssr_frozen_multiseed_20ep.py --seed "$S" \
    2>&1 | tee "/root/autodl-tmp/frozen_seed${S}.log"
done

"$PY" /root/autodl-tmp/aggregate_frozen_three_seed.py \
  2>&1 | tee /root/autodl-tmp/frozen_three_seed_summary.log

echo "========== FINAL =========="
cat "$ROOT/outputs/frozen_multiseed/THREE_SEED_SUMMARY.json"
