Frozen three-baseline evaluation.

Important:
- Baselines DO NOT use product_conf, vertical_overlap, or distance_factor.
- All three consume exactly the same GATE2_FROZEN_VLM_INPUTS.jsonl.
- Same semantic prompt, same answer space, same deterministic parser, greedy decoding.
- Output JSONL supports resume after interruption.
- Final output:
  /root/autodl-tmp/gate2_vlm_baselines_final/TABLE4_MAIN.csv
  /root/autodl-tmp/gate2_vlm_baselines_final/TABLE4_PER_TYPE_EM.csv

Run:
  bash preflight.sh
  nohup bash run_all.sh > /root/autodl-tmp/gate2_vlm_baselines_final/launcher.out 2>&1 &
  bash status.sh

V3 Qwen-safe patch:
- Caps Qwen2-VL native dynamic-resolution visual budget.
- Qwen2 batch size defaults to 1.
- Keeps completed BLIP predictions; resume logic skips them.
