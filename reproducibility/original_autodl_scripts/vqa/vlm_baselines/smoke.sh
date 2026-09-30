#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
find_model(){ for p in "$@"; do [ -f "$p/config.json" ] && { echo "$p"; return 0; }; done; return 1; }
PY=/root/autodl-tmp/envs/qwen3vl/bin/python
[ -x "$PY" ] || PY=/root/autodl-tmp/ssr_yolorace_exp/venv/bin/python
BLIP=$(find_model /root/autodl-tmp/ssr_yolorace_exp/models/blip-vqa-base /root/autodl-tmp/models/blip-vqa-base)
Q25=$(find_model /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct)
Q3=$(find_model /root/autodl-tmp/models/Qwen3-VL-8B-Thinking)
"$PY" -u "$D/run_blip.py" --model "$BLIP" --limit 8
"$PY" -u "$D/run_qwen.py" --model "$Q25" --name 'Qwen2.5-VL-7B' --limit 8
"$PY" -u "$D/run_qwen.py" --model "$Q3" --name 'Qwen3-VL-8B' --limit 8
echo SMOKE_PASS
