from pathlib import Path
import argparse
import json
import yaml
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from src.reasoning.dsi import box_iou


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def match_iou(preds, gts, iou_thr=0.30):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou-thr", type=float, default=0.30)
    parser.add_argument("--vacancy-class", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    yolo_root = processed_root / "yolo_retaildet_style"
    manifest_path = yolo_root / f"{args.split}_manifest.jsonl"
    rows = read_jsonl(manifest_path)

    model = YOLO(args.weights)

    total_tp, total_fp, total_fn = 0, 0, 0
    per_sample = []

    for rec in tqdm(rows, desc="Eval YOLO vacancy"):
        img_path = rec["image_path"]

        result = model.predict(
            source=img_path,
            conf=args.conf,
            iou=0.5,
            imgsz=640,
            verbose=False
        )[0]

        preds = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for box, c, cf in zip(xyxy, cls, confs):
                if int(c) == args.vacancy_class:
                    preds.append({
                        "bbox": [float(v) for v in box.tolist()],
                        "score": float(cf)
                    })

        gts = rec["gt_missing"]
        tp, fp, fn = match_iou(preds, gts, iou_thr=args.iou_thr)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_sample.append({
            "masked_id": rec["masked_id"],
            "image_path": img_path,
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
        "Method": "RetailDet-style YOLO vacancy detector",
        "split": args.split,
        "weights": args.weights,
        "conf": args.conf,
        "iou_thr": args.iou_thr,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "per_sample": per_sample
    }

    out_dir = output_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"retaildet_style_yolo_{args.split}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    out_csv = output_root / "tables" / f"retaildet_style_yolo_{args.split}.csv"
    pd.DataFrame([{
        "Method": result["Method"],
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "conf": args.conf
    }]).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(pd.read_csv(out_csv))
    print("Saved:", out_json)
    print("Saved:", out_csv)
    print("=" * 80)


if __name__ == "__main__":
    main()