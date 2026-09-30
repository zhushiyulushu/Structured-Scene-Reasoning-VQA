#!/bin/bash
OUT=/root/autodl-tmp/gate2_vqa_region_final/vlm_baselines_final
ps -eo pid,etime,%cpu,%mem,cmd | grep -E 'run_blip.py|run_qwen.py|run_all.sh' | grep -v grep || true
for f in "$OUT/blip/predictions.jsonl" "$OUT/qwen25_vl_7b/predictions.jsonl" "$OUT/qwen3_vl_8b/predictions.jsonl"; do [ -f "$f" ] && echo "$(wc -l < "$f") $f" || echo "0 $f"; done
for f in "$OUT/blip.log" "$OUT/qwen25.log" "$OUT/qwen3.log" "$OUT/final_tables.log"; do echo "--- $f"; tail -n 10 "$f" 2>/dev/null || true; done
