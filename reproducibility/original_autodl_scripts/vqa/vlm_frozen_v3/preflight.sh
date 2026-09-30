#!/bin/bash
set -euo pipefail
FROZEN=/root/autodl-tmp/gate2_vqa_region_final/GATE2_FROZEN_VLM_INPUTS.jsonl

find_model_dir () {
  pattern="$1"
  find /root/autodl-tmp/ssr_yolorace_exp /root/autodl-tmp \
    -type f -path "$pattern" -print 2>/dev/null \
    | head -n 1 | xargs -r dirname
}

echo "========== FROZEN =========="
test -f "$FROZEN"
N=$(wc -l < "$FROZEN")
echo "QUESTIONS=$N"
[ "$N" -eq 11448 ] || { echo "ERROR expected 11448"; exit 2; }

BLIP_MODEL="$(find_model_dir '*/blip-vqa-base/config.json')"
QWEN2_MODEL="$(find_model_dir '*/Qwen2-VL-2B-Instruct/config.json')"
LLAVA_MODEL="$(find_model_dir '*/llava-1.5-7b-hf/config.json')"

echo "========== AUTO-DETECTED MODELS =========="
echo "BLIP_MODEL=$BLIP_MODEL"
echo "QWEN2_MODEL=$QWEN2_MODEL"
echo "LLAVA_MODEL=$LLAVA_MODEL"

[ -n "$BLIP_MODEL" ] && [ -f "$BLIP_MODEL/config.json" ] || { echo "ERROR BLIP missing"; exit 3; }
[ -n "$QWEN2_MODEL" ] && [ -f "$QWEN2_MODEL/config.json" ] || { echo "ERROR Qwen2-VL missing"; exit 4; }
[ -n "$LLAVA_MODEL" ] && [ -f "$LLAVA_MODEL/config.json" ] || { echo "ERROR LLaVA missing"; exit 5; }

PY=/root/autodl-tmp/ssr_yolorace_exp/venv/bin/python
[ -x "$PY" ] || { echo "ERROR venv python missing"; exit 6; }

echo "========== PYTHON STACK =========="
"$PY" - <<'PY'
import torch, transformers
from transformers import BlipForQuestionAnswering, Qwen2VLForConditionalGeneration, LlavaForConditionalGeneration
print("torch",torch.__version__)
print("transformers",transformers.__version__)
print("cuda",torch.cuda.is_available())
print("BLIP_IMPORT_OK")
print("QWEN2VL_IMPORT_OK")
print("LLAVA_IMPORT_OK")
PY

cat > /root/autodl-tmp/gate2_vlm_baselines_final/model_paths.env <<EOF
export BLIP_MODEL="$BLIP_MODEL"
export QWEN2_MODEL="$QWEN2_MODEL"
export LLAVA_MODEL="$LLAVA_MODEL"
EOF

echo "PREFLIGHT_PASS"
