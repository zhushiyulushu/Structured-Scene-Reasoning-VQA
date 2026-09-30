# Learning Structured Scene Representations for Visual Question Answering

Official reproducibility repository for the manuscript:

**Learning Structured Scene Representations for Visual Question Answering**

This repository implements **Structured Scene Reasoning (SSR)** for
absence-aware reasoning in dense structured shelf scenes.

## Method overview

SSR follows the pipeline:

1. visual product and vacancy perception;
2. sinusoidal spatial encoding and Neural Grid Inference (NGI);
3. row-column structural representation;
4. 16-dimensional structural candidate description;
5. learned structural candidate ranking;
6. ambiguity-aware visual/structural decision;
7. grounded relational representation and structured answering.

The final structural ranker uses a 16 -> 64 -> 32 -> 1 MLP.
Visual confidence is kept separate from the structural score.
Structural evidence is used when competing visual candidates are ambiguous.

## Repository structure

- `src/`: reusable SSR source modules and data-preparation utilities.
- `configs/`: paper parameters and path templates.
- `paper_results/`: final numerical evidence reported in the manuscript.
- `environment/`: exact Python package snapshots.
- `scripts/`: portable result-verification utilities.
- `reproducibility/original_autodl_scripts/`: exact scripts used on the
  original AutoDL machine for experimental provenance.

Raw datasets, generated intermediate data, prediction caches, downloaded
VLM checkpoints and training environments are intentionally not included.

## Datasets

The paper uses:

- SKU-110K for controlled source-domain localization and structured VQA.
- EgoCart for natural frame-level out-of-stock evaluation.

Obtain the datasets from their original distributions and configure local
paths using `configs/paths.example.yaml`.

## Installation

Core SSR dependencies are recorded in:

    environment/requirements_ssr_core_exact.txt

Install them with:

    pip install -r requirements.txt

The VLM-baseline environment is recorded separately in:

    environment/requirements_vlm_exact.txt

## Final paper results

### SKU-110K localization

SSR:
- Precision: 98.36 +/- 0.12 %
- Recall: 92.64 +/- 0.04 %
- F1: 95.41 +/- 0.05 %

### EgoCart natural OOS evaluation

SSR:
- F1: 78.64 %
- Specificity: 60.70 %
- Balanced Accuracy: 70.30 %
- AUROC: 68.97 %
- AUPRC: 81.20 %

### Structured VQA

SSR:
- Locate EM: 92.42 %
- Overall EM: 94.65 %

### Multi-candidate ablation

Relative to Full SSR:
- without learned candidate ranking: -8.49 F1 points;
- without topology context: -19.29 F1 points.

The complete frozen tables are stored under `paper_results/`.

## Verify released results

Run:

    python scripts/verify_paper_results.py

This does not retrain the model. It verifies that the released paper tables
contain the values frozen for the manuscript.

## Reproducibility and provenance

Exact historical launch scripts are retained under:

    reproducibility/original_autodl_scripts/

They intentionally retain the absolute paths of the original AutoDL
environment. They are preserved for auditability and are not intended as
portable entry points.

Portable source modules are provided under `src/`.

See `docs/REPRODUCIBILITY.md` for details.

## Data-leakage control

Source Test target annotations are used only by the evaluator.
The final runtime no-leak audit evidence is included in
`paper_results/source/`.

## Third-party resources

Datasets, pretrained vision-language models and third-party baseline
implementations retain their respective original licenses and should be
obtained from their original sources.
