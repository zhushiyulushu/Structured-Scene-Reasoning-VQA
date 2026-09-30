# Figure 3 Detector Data Notes

This figure uses the three detector seed logs from `SSR_GATE0_CURVES_EXPORT`.

- Source seed files: `seed42_detector_epoch_metrics.csv`, `seed2024_detector_epoch_metrics.csv`, `seed3407_detector_epoch_metrics.csv`.
- Epochs are displayed as 1-20; the original YOLO-style logs store them as 0-19.
- Left panel: `train_total_loss`. Right panel: validation `mAP50`.
- The figure shows only the mean curve and a light +/-1 sample-std band.
- Styling parameters, including font family, font sizes, colors, linewidth, and band alpha, are grouped near the top of `plot_fig3_detector_final.py`.
