from pathlib import Path
import argparse
import yaml
import pandas as pd


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_row(df, method):
    return df[df["Method"] == method].iloc[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = Path(cfg["output_root"]) / "tables"

    baseline = pd.read_csv(table_dir / "baseline_comparison_iou_only_test.csv")
    learnable = pd.read_csv(table_dir / "learnable_grid_mean_std.csv")

    retail = pd.read_csv(table_dir / "retaildet_style_yolo_test_val_tuned.csv").iloc[0]
    ssr_full = pd.read_csv(table_dir / "ssr_full_fusion_test_no_pred_fallback.csv").iloc[0]

    rows = []

    h = get_row(baseline, "Heuristic Rule-based tuned")
    rows.append({
        "Method": "Heuristic Rule-based",
        "Setting": "geometry-only",
        "Precision": f"{h['Precision']:.4f}",
        "Recall": f"{h['Recall']:.4f}",
        "F1": f"{h['F1']:.4f}",
        "MeanIoU": "-"
    })

    rows.append({
        "Method": "RetailDet-style YOLO",
        "Setting": "visual vacancy detector, val-tuned conf",
        "Precision": f"{float(retail['Precision']):.4f}",
        "Recall": f"{float(retail['Recall']):.4f}",
        "F1": f"{float(retail['F1']):.4f}",
        "MeanIoU": "-"
    })

    p = get_row(baseline, "GT-Planogram Alignment")
    rows.append({
        "Method": "GT-Planogram Alignment",
        "Setting": "GT-grid assisted",
        "Precision": f"{p['Precision']:.4f}",
        "Recall": f"{p['Recall']:.4f}",
        "F1": f"{p['F1']:.4f}",
        "MeanIoU": "-"
    })

    lg = learnable[learnable["Split"] == "test"].iloc[0]
    rows.append({
        "Method": "Learnable Grid Completion",
        "Setting": "learned structural completion",
        "Precision": lg["Precision_mean_std"],
        "Recall": lg["Recall_mean_std"],
        "F1": lg["F1_mean_std"],
        "MeanIoU": lg["MeanIoU_mean_std"]
    })

    rows.append({
        "Method": "SSR-Full Fusion",
        "Setting": "visual vacancy + structural fallback",
        "Precision": f"{float(ssr_full['Precision']):.4f}",
        "Recall": f"{float(ssr_full['Recall']):.4f}",
        "F1": f"{float(ssr_full['F1']):.4f}",
        "MeanIoU": "-"
    })

    o = get_row(baseline, "Oracle Grid DSI")
    rows.append({
        "Method": "Oracle Grid DSI",
        "Setting": "GT-grid upper bound",
        "Precision": f"{o['Precision']:.4f}",
        "Recall": f"{o['Recall']:.4f}",
        "F1": f"{o['F1']:.4f}",
        "MeanIoU": "-"
    })

    out = pd.DataFrame(rows)
    out_path = table_dir / "table2_missing_localization_final.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(out)
    print("=" * 80)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()