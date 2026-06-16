from pathlib import Path
import argparse
import yaml
import pandas as pd
from ultralytics import YOLO

from src.eval.eval_gap_seg_dataset_table import (
    build_prediction_cache,
    eval_cache,
)


DATASETS = ["GroZi-120", "Grocery Products", "WebMarket"]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tune_one_dataset(val_cache, dataset):
    sub_cache = [x for x in val_cache if x["dataset"] == dataset]

    if len(sub_cache) == 0:
        return None

    # 更小、更快，但对 Grocery 更友好的搜索范围
    confs = [0.03, 0.05, 0.08, 0.10, 0.15, 0.18, 0.20, 0.25]
    closes = [0, 9, 15]
    dilates = [0, 11, 21, 31]
    tolerances = [25, 35, 45, 55]
    min_areas = [0, 50, 100]

    records = []

    for conf in confs:
        for close_k in closes:
            for dilate_k in dilates:
                for tol in tolerances:
                    for ma in min_areas:
                        df = eval_cache(
                            sub_cache,
                            conf=conf,
                            close_k=close_k,
                            dilate_k=dilate_k,
                            min_area=ma,
                            tolerance=tol,
                        )

                        records.append({
                            "Dataset": dataset,
                            "conf": conf,
                            "close_k": close_k,
                            "dilate_k": dilate_k,
                            "min_area": ma,
                            "tolerance": tol,
                            "val_precision": float(df["Precision"].mean()),
                            "val_recall": float(df["Recall"].mean()),
                            "val_f1": float(df["F1"].mean()),
                            "val_iou": float(df["IoU"].mean()),
                        })

    out = pd.DataFrame(records).sort_values("val_f1", ascending=False)
    return out.iloc[0].to_dict(), out


def eval_one_dataset(test_cache, dataset, best):
    sub_cache = [x for x in test_cache if x["dataset"] == dataset]

    if len(sub_cache) == 0:
        return None

    df = eval_cache(
        sub_cache,
        conf=float(best["conf"]),
        close_k=int(best["close_k"]),
        dilate_k=int(best["dilate_k"]),
        min_area=int(best["min_area"]),
        tolerance=int(best["tolerance"]),
    )

    return {
        "Dataset": dataset,
        "Samples": len(df),
        "Recall": float(df["Recall"].mean()),
        "Precision": float(df["Precision"].mean()),
        "F1-score": float(df["F1"].mean()),
        "Best conf": float(best["conf"]),
        "Best close_k": int(best["close_k"]),
        "Best dilate_k": int(best["dilate_k"]),
        "Best min_area": int(best["min_area"]),
        "Best tolerance": int(best["tolerance"]),
        "Val F1": float(best["val_f1"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf-min", type=float, default=0.001)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    root = processed_root / "yolo_gap_seg"
    table_dir = output_root / "tables"
    result_dir = output_root / "results"
    table_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    print("[1] Build val prediction cache")
    val_cache = build_prediction_cache(model, root, "val", args.imgsz, args.conf_min)

    print("[2] Build test prediction cache")
    test_cache = build_prediction_cache(model, root, "test", args.imgsz, args.conf_min)

    all_sweeps = []
    table_rows = []

    source_f1 = 0.9671

    table_rows.append({
        "Dataset": "SKU-110K (source test set)",
        "Samples": 2862,
        "Recall": 0.9726,
        "Precision": 0.9617,
        "F1-score": source_f1,
        "Domain Shift": 0.0000,
        "Best conf": "-",
        "Best close_k": "-",
        "Best dilate_k": "-",
        "Best min_area": "-",
        "Best tolerance": "-",
        "Val F1": "-",
    })

    for dataset in DATASETS:
        print(f"\n[3] Tune {dataset}")
        best, sweep = tune_one_dataset(val_cache, dataset)
        all_sweeps.append(sweep)

        print("[Best]", best)

        print(f"[4] Test {dataset}")
        row = eval_one_dataset(test_cache, dataset, best)
        f1 = row["F1-score"]
        row["Domain Shift"] = max(0.0, 1.0 - f1 / source_f1)
        table_rows.append(row)

    sweep_all = pd.concat(all_sweeps, ignore_index=True)
    sweep_path = table_dir / "gap_seg_per_domain_tuning_sweep.csv"
    sweep_all.to_csv(sweep_path, index=False, encoding="utf-8-sig")

    out = pd.DataFrame(table_rows)

    for c in ["Recall", "Precision", "F1-score", "Domain Shift"]:
        out[c] = out[c].apply(lambda x: x if isinstance(x, str) else f"{x:.4f}")

    table_path = table_dir / "table3_cross_domain_per_domain_final.csv"
    out.to_csv(table_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(out)
    print("Saved:", table_path)
    print("Sweep:", sweep_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
