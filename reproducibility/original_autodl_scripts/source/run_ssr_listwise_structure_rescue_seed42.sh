#!/bin/bash
set -euo pipefail
PY=/root/autodl-tmp/ssr_yolorace_exp/venv/bin/python
cd /root/autodl-tmp
"$PY" /root/autodl-tmp/ssr_listwise_structure_rescue_seed42.py 2>&1 | tee /root/autodl-tmp/ssr_listwise_structure_rescue_seed42.log
