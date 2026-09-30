EgoCart experiment freeze
Date: 2026-09-27

MAIN ORIGINAL SSR
- Dataset: EgoCart
- Test frames: 6171
- SSR Full F1: 0.686401
- Balanced Accuracy: 0.563244
- Specificity: 0.424729
- AUROC: 0.589727
- AUPRC: 0.713230

SSR + UNLABELED PRIOR-SHIFT ADAPTATION
- Test labels used for adaptation: NO
- Unlabeled target score distribution used: YES
- Frozen threshold: 0.319
- TP: 3838
- FP: 2261
- FN: 28
- TN: 44
- Accuracy: 0.629071
- Precision: 0.629283
- Recall: 0.992757
- Specificity: 0.019089
- Balanced Accuracy: 0.505923
- F1: 0.770296
- AUROC: 0.589727
- AUPRC: 0.713230

Important:
This adapted result is transductive / unlabeled-target adaptation.
It must not be described as ordinary Train-only threshold calibration.
The original SSR result remains the proper source for claims about
balanced discrimination, specificity, AUROC and AUPRC.

Train-only source-aware feature audit selected CURRENT features,
so the source-aware feature variant was rejected and not used.
