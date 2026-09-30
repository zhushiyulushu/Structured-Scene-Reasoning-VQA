#!/bin/bash
set -euo pipefail
PY=/root/autodl-tmp/ssr_yolorace_exp/venv/bin/python
D="$(cd "$(dirname "$0")" && pwd)"
OUT=/root/autodl-tmp/gate2_vqa_region_final
mkdir -p "$OUT"; export PYTHONUNBUFFERED=1
"$PY" -u "$D/build_val_protocol.py" 2>&1 | tee "$OUT/val_protocol.log"
