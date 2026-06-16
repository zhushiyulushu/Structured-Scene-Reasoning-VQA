from pathlib import Path
import argparse
import json
import yaml
import pandas as pd
from tqdm import tqdm

from src.baselines.baseline_gap_methods import heuristic_rule_based
from src.reasoning.dsi import box_iou


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def match_predictions(preds, gts, iou_thr=0.30):
    matched = set()
    tp = 0

    for gt in gts:
        best_i = None
        best_iou = 0.0

        for i, p in enumerate(preds):
            if i in matched:
                continue

            if int(p["row"]) != int(gt["row"]):
                continue
            if int(p["col"]) != int(gt["col"]):
                continue

            iou = box_iou(p["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_i = i

        if best_i is not None and best_iou >= iou_thr:
            matched.add(best_i)
            tp += 1

    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def evaluate(records, gap_factor, sigma, iou_thr):
    total_tp, total_fp, total_fn = 0, 0, 0

    for rec in records:
        preds = heuristic_rule_based(
            rec,
            gap_factor=gap_factor,
            sigma=sigma
        )
        gts = rec["missing_objects"]
        tp, fp, fn = match_predictions(preds, gts, iou_thr=iou_thr)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--iou-thr", type=float, default=0.30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    val_path = processed_root / "masked" / "val_masked_img.jsonl"
    test_path = processed_root / "masked" / "test_masked_img.jsonl"

    val_records = read_jsonl(val_path)
    test_records = read_jsonl(test_path)

    gap_factors = [1.2, 1.5, 1.8, 2.0, 2.3, 2.5, 3.0]
    sigmas = [0.05, 0.10, 0.15, 0.20, 0.30]

    rows = []
    best = None

    for gf in tqdm(gap_factors, desc="Tuning gap_factor"):
        for sig in sigmas:
            r = evaluate(val_records, gf, sig, args.iou_thr)
            row = {
                "gap_factor": gf,
                "sigma": sig,
                **r
            }
            rows.append(row)

            if best is None or r["F1"] > best["F1"]:
                best = row

    out_dir = output_root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    for c in ["Precision", "Recall", "F1"]:
        df[c] = df[c].round(4)

    tune_path = out_dir / "heuristic_tuning_val.csv"
    df.to_csv(tune_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("[Best on val]")
    print(best)

    test_result = evaluate(
        test_records,
        best["gap_factor"],
        best["sigma"],
        args.iou_thr
    )

    test_row = {
        "Method": "Heuristic Rule-based tuned on val",
        "gap_factor": best["gap_factor"],
        "sigma": best["sigma"],
        **test_result
    }

    test_df = pd.DataFrame([test_row])
    for c in ["Precision", "Recall", "F1"]:
        test_df[c] = test_df[c].round(4)

    test_out = out_dir / "heuristic_tuned_test.csv"
    test_df.to_csv(test_out, index=False, encoding="utf-8-sig")

    print("[Test with best val params]")
    print(test_df)
    print(f"Val tuning saved to: {tune_path}")
    print(f"Test result saved to: {test_out}")
    print("=" * 80)


if __name__ == "__main__":
    main()