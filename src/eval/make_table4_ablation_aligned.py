from pathlib import Path
import argparse
import yaml
import pandas as pd
import numpy as np


SEEDS = [42, 2024, 3407]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mean_std(vals):
    vals = np.array(vals, dtype=float)
    return vals.mean(), vals.std(ddof=0), f"{vals.mean():.4f} ± {vals.std(ddof=0):.4f}"


def read_f1(path):
    df = pd.read_csv(path)
    return float(df.iloc[0]["F1"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = Path(cfg["output_root"]) / "tables"

    rows = []

    # w/o Semantic: ordinary YOLOv8 visual detector + structural fallback
    wo_semantic_path = table_dir / "ablation_wo_semantic.csv"
    if wo_semantic_path.exists():
        wo_semantic_f1 = read_f1(wo_semantic_path)
    else:
        wo_semantic_f1 = np.nan

    rows.append({
        "Variant": "w/o Semantic",
        "Description": "Replace YOLOWorld semantic visual front-end with ordinary YOLO visual detector",
        "Metric": "Localization F1",
        "Score": f"{wo_semantic_f1:.4f}" if not np.isnan(wo_semantic_f1) else "N/A"
    })

    # w/o Structural Learning: YOLOWorld visual detector only
    visual_vals = [
        read_f1(table_dir / f"yoloworld_visual_test_seed_{s}.csv")
        for s in SEEDS
    ]
    _, _, visual_ms = mean_std(visual_vals)
    rows.append({
        "Variant": "w/o Structural Learning",
        "Description": "YOLOWorld visual detector only",
        "Metric": "Localization F1",
        "Score": visual_ms
    })

    # w/o Visual Vacancy Detector: Learnable Grid Completion only
    learnable = pd.read_csv(table_dir / "learnable_grid_mean_std.csv")
    lg = learnable[learnable["Split"] == "test"].iloc[0]
    rows.append({
        "Variant": "w/o Visual Vacancy Detector",
        "Description": "Learnable Grid Completion only",
        "Metric": "Localization F1",
        "Score": lg["F1_mean_std"]
    })

    # w/o Deep Reasoning: naive union
    union_vals = [
        read_f1(table_dir / f"ablation_wo_deep_reasoning_seed_{s}.csv")
        for s in SEEDS
    ]
    _, _, union_ms = mean_std(union_vals)
    rows.append({
        "Variant": "w/o Deep Reasoning",
        "Description": "Naive union of visual vacancy and structural predictions",
        "Metric": "Localization F1",
        "Score": union_ms
    })

    # Ours
    ours_vals = [
        read_f1(table_dir / f"ours_ssr_grid_test_seed_{s}.csv")
        for s in SEEDS
    ]
    _, _, ours_ms = mean_std(ours_vals)
    rows.append({
        "Variant": "Ours / Our SSR Grid",
        "Description": "YOLOWorld + Learnable Grid Completion fallback + symbolic answer",
        "Metric": "Localization F1",
        "Score": ours_ms
    })

    out = pd.DataFrame(rows)
    out_path = table_dir / "table4_ablation.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(out)
    print("=" * 80)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()