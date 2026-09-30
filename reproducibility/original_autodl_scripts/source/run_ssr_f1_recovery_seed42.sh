#!/bin/bash
set -euo pipefail
PY=/root/autodl-tmp/ssr_yolorace_exp/venv/bin/python
RACE=/root/autodl-tmp/ssr_yolorace_exp/code/YOLO-RACE
export PYTHONPATH="$RACE:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd /root/autodl-tmp
"$PY" /root/autodl-tmp/ssr_f1_recovery_seed42.py 2>&1 | tee /root/autodl-tmp/ssr_f1_recovery_seed42.log
