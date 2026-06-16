from pathlib import Path
import argparse
import json
import yaml
import pandas as pd

from src.reasoning.dsi import box_iou


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_iou(preds, gts, iou_thr=0.30):
    matched = set()
    tp = 0

    for gt in gts:
        gt_box = gt["bbox"] if isinstance(gt, dict) else gt

        best_i = None
        best_iou = 0.0

        for i, p in enumerate(preds):
            if i in matched:
                continue

            pred_box = p["bbox"]
            iou = box_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_i = i

        if best_i is not None and best_iou >= iou_thr:
            matched.add(best_i)
            tp += 1

    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--yolo-json", type=str, required=True)
    parser.add_argument("--learnable-json", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--iou-thr", type=float, default=0.30)
    parser.add_argument("--mode", type=str, default="no_pred_fallback",
                        choices=["no_pred_fallback", "union"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["output_root"])

    yolo_result = read_json(args.yolo_json)
    learnable_result = read_json(args.learnable_json)

    learnable_map = {
        s["masked_id"]: s
        for s in learnable_result["per_sample"]
    }

    total_tp, total_fp, total_fn = 0, 0, 0
    per_sample = []

    for rec in yolo_result["per_sample"]:
        masked_id = rec["masked_id"]
        yolo_preds = rec["preds"]
        gts = rec["gts"]

        preds = []

        if args.mode == "no_pred_fallback":
            # YOLO 有预测就用 YOLO；没有预测才用结构模型补位
            if len(yolo_preds) > 0:
                preds.extend(yolo_preds)
            else:
                if masked_id in learnable_map:
                    lg = learnable_map[masked_id]
                    preds.append({
                        "bbox": lg["pred_bbox"],
                        "score": float(lg.get("iou", 0.0)),
                        "source": "learnable_grid_fallback"
                    })

        elif args.mode == "union":
            # YOLO 与结构模型都保留，可能提升召回，但也可能增加 FP
            preds.extend(yolo_preds)
            if masked_id in learnable_map:
                lg = learnable_map[masked_id]
                preds.append({
                    "bbox": lg["pred_bbox"],
                    "score": float(lg.get("iou", 0.0)),
                    "source": "learnable_grid"
                })

        tp, fp, fn = match_iou(preds, gts, iou_thr=args.iou_thr)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_sample.append({
            "masked_id": masked_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "preds": preds,
            "gts": gts
        })

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    result = {
        "Method": "SSR-Full Fusion",
        "split": args.split,
        "mode": args.mode,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "per_sample": per_sample
    }

    result_dir = output_root / "results"
    table_dir = output_root / "tables"
    result_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    out_json = result_dir / f"ssr_full_fusion_{args.split}_{args.mode}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    out_csv = table_dir / f"ssr_full_fusion_{args.split}_{args.mode}.csv"
    df = pd.DataFrame([{
        "Method": "SSR-Full Fusion",
        "Mode": args.mode,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4)
    }])
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(df)
    print("Saved:", out_json)
    print("Saved:", out_csv)
    print("=" * 80)


if __name__ == "__main__":
    main()