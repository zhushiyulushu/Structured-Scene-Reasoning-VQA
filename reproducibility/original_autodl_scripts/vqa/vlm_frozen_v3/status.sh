#!/bin/bash
OUT=/root/autodl-tmp/gate2_vlm_baselines_final
echo "=== PROCESS ==="
ps -eo pid,etime,stat,%cpu,%mem,cmd | grep -E 'eval_baseline.py|run_all.sh' | grep -v grep || true
echo "=== GPU ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
for x in blip qwen2 llava; do
  echo "=== $x ==="
  [ -f "$OUT/${x}_predictions.jsonl" ] && wc -l "$OUT/${x}_predictions.jsonl" || echo "0 predictions"
  tail -n 10 "$OUT/${x}.log" 2>/dev/null || true
done
echo "=== TABLE4 ==="
cat "$OUT/TABLE4_MAIN.csv" 2>/dev/null || echo "NOT READY"
