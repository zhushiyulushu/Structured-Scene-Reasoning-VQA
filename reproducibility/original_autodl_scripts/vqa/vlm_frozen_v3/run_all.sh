#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
OUT=/root/autodl-tmp/gate2_vlm_baselines_final
mkdir -p "$OUT"

ENVFILE="$OUT/model_paths.env"
if [ ! -f "$ENVFILE" ]; then
  echo "ERROR: run 'bash preflight.sh' first."
  exit 2
fi
source "$ENVFILE"

PY=/root/autodl-tmp/ssr_yolorace_exp/venv/bin/python
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "PY=$PY"
echo "BLIP_MODEL=$BLIP_MODEL"
echo "QWEN2_MODEL=$QWEN2_MODEL"
echo "LLAVA_MODEL=$LLAVA_MODEL"

echo "===== 1/3 BLIP ====="
"$PY" -u "$D/eval_baseline.py" --model blip --batch-size "${BLIP_BS:-16}" \
  2>&1 | tee -a "$OUT/blip.log"

echo "===== 2/3 QWEN2-VL-2B ====="
"$PY" -u "$D/eval_baseline.py" --model qwen2 --batch-size "${QWEN2_BS:-1}" \
  2>&1 | tee -a "$OUT/qwen2.log"

echo "===== 3/3 LLAVA-1.5-7B ====="
"$PY" -u "$D/eval_baseline.py" --model llava --batch-size "${LLAVA_BS:-1}" \
  2>&1 | tee -a "$OUT/llava.log"

echo "===== AGGREGATE ====="
"$PY" "$D/aggregate_tables.py" | tee "$OUT/final_tables.log"
echo "ALL_BASELINES_COMPLETE"
