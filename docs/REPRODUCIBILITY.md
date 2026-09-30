# Reproducibility

## Included in this release

The repository contains:

- SKU-110K preprocessing utilities;
- SSE and NGI source modules;
- the final 16-D structural-ranker definition;
- the final paper result tables;
- source-domain no-leak evidence;
- EgoCart aggregate evidence;
- structured-VQA aggregate evidence;
- ablation evidence;
- Figure 3 data and plotting code;
- exact package snapshots;
- exact original AutoDL experiment scripts for provenance.

## Not included in the public repository

The public repository intentionally excludes:

- raw SKU-110K and EgoCart image data;
- controlled masked-image copies;
- detector prediction caches;
- per-image intermediate tensors;
- large training outputs;
- BLIP, Qwen2-VL and LLaVA checkpoints;
- Python virtual environments.

These materials are archived privately for experimental provenance.

## Historical AutoDL scripts

`reproducibility/original_autodl_scripts/` contains exact copies of
historical experiment scripts.

Absolute `/root/autodl-tmp/...` paths are intentionally preserved inside
that directory because those files document the actually executed
environment. They should not be interpreted as portable public entry points.

## Frozen source parameters

- image size: 640
- vacancy confidence threshold: 0.03
- internal detector confidence: 0.001
- product confidence threshold: 0.30
- ambiguity margin: 0.12
- structural fusion weight: 0.50
- structural ranker input dimension: 16
- structural ranker hidden dimensions: 64 and 32

The same parameters are recorded in `configs/paper.yaml`.
