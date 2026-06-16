\# Learning Structured Scene Representations for Visual Question Answering



This repository provides the implementation of \*\*Structured Scene Reasoning (SSR)\*\* for dense structured visual question answering. SSR converts unordered visual detections into a learnable row-column topological representation and performs missing-region reasoning for structured shelf scenes.



The project targets structured VQA for missing-region analysis and replenishment, rather than open-ended caption generation.



\## Overview



SSR follows a structured pipeline:



1\. Open-vocabulary object detection and local semantic-spatial aggregation.

2\. Sinusoidal Spatial Embedding (SSE) for spatial representation.

3\. Neural Grid Inference (NGI) for row-column topology learning.

4\. Directional Structure Induction (DSI) for missing-region reasoning.

5\. Slot-based structured answer synthesis for controlled VQA queries.



The repository contains the source code used for preprocessing, training, localization evaluation, cross-domain evaluation, ablation study, structured answer-quality evaluation, and result visualization.



\## Repository Structure



```text

configs/          Configuration files.

docs/             Notes, file manifests, and reproducibility documents.

paper\_results/    Small CSV result files used in the paper.

scripts/          Helper scripts for running experiments.

src/              Source code for preprocessing, training, evaluation, and visualization.

README.md         Repository documentation and usage instructions.

requirements.txt  Python dependencies.

```



Large datasets, processed images, model checkpoints, detector weights, and full output folders are not included in this repository.



\## Environment Setup



Create a Python environment:



```bash

conda create -n ssr-vqa python=3.10 -y

conda activate ssr-vqa

pip install -r requirements.txt

```



The experiments in the paper were run on an AutoDL node with an NVIDIA RTX 3090 GPU.



\## Data Preparation



This repository does not include raw datasets due to size and license restrictions.



The expected data structure is:



```text

data/

├── SKU-110K/

├── GapDetection/

└── processed/

```



The main experiments use SKU-110K with a controlled topological masking protocol. Existing product annotations are organized into approximate row-column layouts. Objects with sufficient neighboring context are masked, and their original boxes are retained as ground-truth missing regions. This creates controlled missing-object samples without extra manual vacancy annotations.



External Gap Detection subsets are used for cross-domain evaluation.



\## Main Experimental Pipeline



The complete experimental pipeline includes:



1\. Preparing controlled missing-object samples from SKU-110K.

2\. Training the learnable grid-completion branch.

3\. Evaluating localization baselines and the full SSR model.

4\. Running cross-domain evaluation on Gap Detection subsets.

5\. Running ablation studies.

6\. Evaluating structured answer quality.

7\. Generating paper tables and figures.



Please check the scripts under `scripts/` and modules under `src/` for the corresponding implementation.



\## Structured VQA Answer Quality Evaluation



SKU-110K does not provide human-written VQA annotations. Therefore, structured VQA references are automatically constructed from the controlled topological masking protocol. Each reference answer is derived from the ground-truth missing region, row-column position, and neighboring structural context.



Run structured answer-quality evaluation for three independent runs:



```bash

for s in 42 2024 3407

do

&#x20; python -m src.eval.eval\_vqa\_answer\_quality \\

&#x20;   --qa-jsonl data/processed/vqa/test\_qa.jsonl \\

&#x20;   --ours-json outputs/results/ours\_ssr\_grid\_test\_seed\_${s}.json \\

&#x20;   --out-summary outputs/tables/table5\_vqa\_answer\_quality\_seed\_${s}.csv \\

&#x20;   --out-detail outputs/results/table5\_vqa\_answer\_quality\_detail\_seed\_${s}.csv

done

```



The reported metrics include normalized Answer Exact Match, Token-F1, and BLEU-1. BLEU-4 and CIDEr are not used because the generated answers are short structured responses rather than long free-form captions with multiple human references.



\## Paper Results



Small paper-level CSV results are stored in:



```text

paper\_results/

```



These files summarize the main localization, cross-domain, ablation, and structured answer-quality results used in the paper.



Full intermediate outputs are excluded from GitHub and should be stored separately in the experiment backup archive.



\## Reproducibility Notes



To reproduce the experiments from scratch, users need to prepare the required datasets and follow the same preprocessing and topological masking protocol.



This repository is intended for code release and paper-level reproducibility. It does not include raw datasets, processed images, detector weights, large model checkpoints, or complete AutoDL output folders.



\## Citation



If you use this code, please cite the corresponding paper:



```bibtex

@article{ssr\_vqa\_2026,

&#x20; title={Learning Structured Scene Representations for Visual Question Answering},

&#x20; author={Fan, Miao and Zhu, Shiyu and Ma, Chen and Xiong, Haoyi},

&#x20; year={2026}

}

```



