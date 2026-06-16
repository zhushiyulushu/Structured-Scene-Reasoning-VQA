from pathlib import Path
import argparse
import yaml
import pandas as pd
import numpy as np


SEEDS = [42, 2024, 3407]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mean_std(values):
    values = np.array(values, dtype=float)
    return values.mean(), values.std(ddof=0), f"{values.mean():.4f} ± {values.std(ddof=0):.4f}"


def read_f1(path):
    df = pd.read_csv(path)
    return float(df.iloc[0]["F1"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = Path(cfg["output_root"]) / "tables"

    baseline = pd.read_csv(table_dir / "baseline_comparison_iou_only_test.csv")

    heuristic_f1 = float(
        baseline[baseline["Method"] == "Heuristic Rule-based tuned"].iloc[0]["F1"]
    )
    planogram_f1 = float(
        baseline[baseline["Method"] == "GT-Planogram Alignment"].iloc[0]["F1"]
    )

    blip = pd.read_csv(table_dir / "blip_baseline_summary_test_-1.csv")
    blip_avg = float(blip["accuracy"].mean())

    seed_rows = []

    for s in SEEDS:
        seed_rows.append({
            "Method": "Heuristic (Rule-based)",
            "Seed": s,
            "Metric": "Localization F1",
            "Score": heuristic_f1
        })

        seed_rows.append({
            "Method": "BLIP (Vision-Language)",
            "Seed": s,
            "Metric": "Avg. VQA Accuracy",
            "Score": blip_avg
        })

        yolo_f1 = read_f1(table_dir / f"yoloworld_visual_test_seed_{s}.csv")
        seed_rows.append({
            "Method": "Baseline RetailDet (Visual)",
            "Seed": s,
            "Metric": "Localization F1",
            "Score": yolo_f1
        })

        seed_rows.append({
            "Method": "Baseline GT-Planogram Alignment",
            "Seed": s,
            "Metric": "Localization F1",
            "Score": planogram_f1
        })

        ours_f1 = read_f1(table_dir / f"ours_ssr_grid_test_seed_{s}.csv")
        seed_rows.append({
            "Method": "Ours / Our SSR Grid",
            "Seed": s,
            "Metric": "Localization F1",
            "Score": ours_f1
        })

    seed_df = pd.DataFrame(seed_rows)
    seed_path = table_dir / "table2_comparison_seed_results.csv"
    seed_df.to_csv(seed_path, index=False, encoding="utf-8-sig")

    summary_rows = []
    for method, g in seed_df.groupby("Method", sort=False):
        vals = g["Score"].tolist()
        m, sd, ms = mean_std(vals)
        summary_rows.append({
            "Method": method,
            "Metric": g["Metric"].iloc[0],
            "Seed42": f"{vals[0]:.4f}",
            "Seed2024": f"{vals[1]:.4f}",
            "Seed3407": f"{vals[2]:.4f}",
            "Mean": f"{m:.4f}",
            "Std": f"{sd:.4f}",
            "Mean±Std": ms
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = table_dir / "table2_comparison_final.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(summary_df)
    print("Saved:", seed_path)
    print("Saved:", summary_path)
    print("=" * 80)


if __name__ == "__main__":
    main()