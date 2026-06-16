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
    return f"{vals.mean():.4f} ± {vals.std(ddof=0):.4f}"


def read_single(path):
    return pd.read_csv(path).iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = Path(cfg["output_root"]) / "tables"

    baseline = pd.read_csv(table_dir / "baseline_comparison_iou_only_test.csv")

    def get_baseline(method):
        return baseline[baseline["Method"] == method].iloc[0].to_dict()

    heuristic = get_baseline("Heuristic Rule-based tuned")
    planogram = get_baseline("GT-Planogram Alignment")

    blip = pd.read_csv(table_dir / "blip_baseline_summary_test_-1.csv")
    blip_score = float(blip["accuracy"].mean())

    rows = []

    # Heuristic deterministic
    rows.append({
        "Method": "Heuristic (Rule-based)",
        "Recall": mean_std([heuristic["Recall"]] * 3),
        "Precision": mean_std([heuristic["Precision"]] * 3),
        "F1-score": mean_std([heuristic["F1"]] * 3),
        "Note": "Localization"
    })

    # BLIP VQA baseline
    rows.append({
        "Method": "BLIP (Vision-Language)",
        "Recall": mean_std([blip_score] * 3),
        "Precision": mean_std([blip_score] * 3),
        "F1-score": mean_std([blip_score] * 3),
        "Note": "Avg. structured VQA accuracy"
    })

    # RetailDet / YOLOWorld visual
    r_list, p_list, f_list = [], [], []
    for s in SEEDS:
        row = read_single(table_dir / f"yoloworld_visual_test_seed_{s}.csv")
        r_list.append(float(row["Recall"]))
        p_list.append(float(row["Precision"]))
        f_list.append(float(row["F1"]))

    rows.append({
        "Method": "Baseline RetailDet (Visual)",
        "Recall": mean_std(r_list),
        "Precision": mean_std(p_list),
        "F1-score": mean_std(f_list),
        "Note": "Localization"
    })

    # GT-Planogram deterministic
    rows.append({
        "Method": "Baseline GT-Planogram Alignment",
        "Recall": mean_std([planogram["Recall"]] * 3),
        "Precision": mean_std([planogram["Precision"]] * 3),
        "F1-score": mean_std([planogram["F1"]] * 3),
        "Note": "Localization"
    })

    # Ours
    r_list, p_list, f_list = [], [], []
    for s in SEEDS:
        row = read_single(table_dir / f"ours_ssr_grid_test_seed_{s}.csv")
        r_list.append(float(row["Recall"]))
        p_list.append(float(row["Precision"]))
        f_list.append(float(row["F1"]))

    rows.append({
        "Method": "Ours / Our SSR Grid",
        "Recall": mean_std(r_list),
        "Precision": mean_std(p_list),
        "F1-score": mean_std(f_list),
        "Note": "Localization"
    })

    out = pd.DataFrame(rows)
    out_path = table_dir / "table2_prf_final.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(out)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()