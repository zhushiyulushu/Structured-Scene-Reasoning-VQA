#!/bin/bash
set -euo pipefail
INPUT=/root/autodl-tmp/gate2_vqa_region_final/GATE2_FROZEN_VLM_INPUTS.jsonl
test -f "$INPUT" || { echo "MISSING $INPUT"; exit 2; }
echo "INPUT_LINES=$(wc -l < "$INPUT")"
for p in /root/autodl-tmp/ssr_yolorace_exp/models/blip-vqa-base /root/autodl-tmp/models/blip-vqa-base /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct /root/autodl-tmp/models/Qwen3-VL-8B-Thinking; do [ -f "$p/config.json" ] && echo "MODEL_OK $p" || true; done
for p in /root/autodl-tmp/envs/qwen3vl/bin/python /root/autodl-tmp/ssr_yolorace_exp/venv/bin/python /root/autodl-tmp/envs/pred-eqa/bin/python; do [ -x "$p" ] || continue; echo "PY $p"; "$p" - <<'PY' || true
import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.is_available())
PY
done
echo PREFLIGHT_DONE
