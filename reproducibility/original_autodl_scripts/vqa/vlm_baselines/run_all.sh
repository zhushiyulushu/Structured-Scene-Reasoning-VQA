#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
OUT=/root/autodl-tmp/gate2_vqa_region_final/vlm_baselines_final
mkdir -p "$OUT"
find_model(){ for p in "$@"; do [ -f "$p/config.json" ] && { echo "$p"; return 0; }; done; return 1; }
find_py(){ for p in /root/autodl-tmp/envs/qwen3vl/bin/python /root/autodl-tmp/ssr_yolorace_exp/venv/bin/python /root/autodl-tmp/envs/pred-eqa/bin/python; do [ -x "$p" ] || continue; if "$p" - <<'PY' >/dev/null 2>&1
import torch,transformers
assert torch.cuda.is_available()
PY
then echo "$p"; return 0; fi; done; return 1; }
BLIP=$(find_model /root/autodl-tmp/ssr_yolorace_exp/models/blip-vqa-base /root/autodl-tmp/models/blip-vqa-base)
Q25=$(find_model /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct)
Q3=$(find_model /root/autodl-tmp/models/Qwen3-VL-8B-Thinking)
PY=$(find_py)
echo "PY=$PY";echo "BLIP=$BLIP";echo "Q25=$Q25";echo "Q3=$Q3"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo '=== 1/3 BLIP ==='; "$PY" -u "$D/run_blip.py" --model "$BLIP" 2>&1 | tee -a "$OUT/blip.log"
echo '=== 2/3 QWEN2.5 ==='; "$PY" -u "$D/run_qwen.py" --model "$Q25" --name 'Qwen2.5-VL-7B' 2>&1 | tee -a "$OUT/qwen25.log"
echo '=== 3/3 QWEN3 ==='; "$PY" -u "$D/run_qwen.py" --model "$Q3" --name 'Qwen3-VL-8B' 2>&1 | tee -a "$OUT/qwen3.log"
echo '=== MERGE ==='; "$PY" -u "$D/merge_tables.py" | tee "$OUT/final_tables.log"
echo ALL_BASELINES_COMPLETE
