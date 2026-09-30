#!/bin/bash
set -euo pipefail
PY=/root/autodl-tmp/ssr_yolorace_exp/venv/bin/python
"$PY" /root/autodl-tmp/gate0_runtime_no_leak_audit.py 2>&1 | tee /root/autodl-tmp/gate0_runtime_no_leak_audit.log
