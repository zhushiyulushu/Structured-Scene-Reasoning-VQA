from pathlib import Path
import argparse
import json
import yaml
import pandas as pd
from tqdm import tqdm

from src.reasoning.dsi import box_iou, infer_missing_by_dsi
from src.baselines.baseline_gap_methods import heuristic_rule_based, planogram_row_alignment


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


def match_predictions_iou_only(preds, gts, iou_thr=0.30):
    matched = set()
    tp = 0

    for gt in gts:
        best_i = None
        best_iou = 0.0

        for i, p in enumerate(preds):
            if i in matched:
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


def eval_method(records, method_name, method_fn, iou_thr=0.30):
    total_tp, total_fp, total_fn = 0, 0, 0

    for rec in tqdm(records, desc=f"Eval {method_name}"):
        preds = method_fn(rec)
        gts = rec["missing_objects"]

        tp, fp, fn = match_predictions_iou_only(preds, gts, iou_thr=iou_thr)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "Method": method_name,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--iou-thr", type=float, default=0.30)
    parser.add_argument("--heuristic-gap-factor", type=float, default=1.2)
    parser.add_argument("--heuristic-sigma", type=float, default=0.1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    in_path = processed_root / "masked" / f"{args.split}_masked_img.jsonl"
    if not in_path.exists():
        in_path = processed_root / "masked" / f"{args.split}_masked.jsonl"

    records = read_jsonl(in_path)

    methods = [
        (
            "Heuristic Rule-based tuned",
            lambda rec: heuristic_rule_based(
                rec,
                gap_factor=args.heuristic_gap_factor,
                sigma=args.heuristic_sigma
            )
        ),
        (
            "GT-Planogram Alignment",
            lambda rec: planogram_row_alignment(rec, sigma=args.sigma)
        ),
        (
            "Oracle Grid DSI",
            lambda rec: infer_missing_by_dsi(rec, sigma=args.sigma)
        ),
    ]

    rows = []
    for name, fn in methods:
        rows.append(eval_method(records, name, fn, iou_thr=args.iou_thr))

    df = pd.DataFrame(rows)
    for col in ["Precision", "Recall", "F1"]:
        df[col] = df[col].round(4)

    out_dir = output_root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"baseline_comparison_iou_only_{args.split}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(df)
    print("=" * 80)
    print(f"[完成] IoU-only baseline comparison saved to: {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()