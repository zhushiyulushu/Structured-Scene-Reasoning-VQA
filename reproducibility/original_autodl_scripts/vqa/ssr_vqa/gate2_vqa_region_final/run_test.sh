#!/bin/bash
set -euo pipefail
PY=/root/autodl-tmp/ssr_yolorace_exp/venv/bin/python
D="$(cd "$(dirname "$0")" && pwd)"
OUT=/root/autodl-tmp/gate2_vqa_region_final
"$PY" -u "$D/run_frozen_test.py" 2>&1 | tee "$OUT/test_ssr.log"
echo; cat "$OUT/GATE2_VQA_TEST_SSR_SUMMARY.csv"
