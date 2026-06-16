from pathlib import Path
import argparse
import yaml
import pandas as pd


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_row(df, method):
    r = df[df["Method"] == method].iloc[0]
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = Path(cfg["output_root"]) / "tables"

    baseline_path = table_dir / "baseline_comparison_iou_only_test.csv"
    learnable_path = table_dir / "learnable_grid_mean_std.csv"

    baseline = pd.read_csv(baseline_path)
    learnable = pd.read_csv(learnable_path)

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

    o = get_row(baseline, "Oracle Grid DSI")
    rows.append({
        "Method": "Oracle Grid DSI",
        "Setting": "upper bound",
        "Precision": f"{o['Precision']:.4f}",
        "Recall": f"{o['Recall']:.4f}",
        "F1": f"{o['F1']:.4f}",
        "MeanIoU": "-"
    })

    out = pd.DataFrame(rows)
    out_path = table_dir / "table2_missing_localization.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(out)
    print("=" * 80)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()