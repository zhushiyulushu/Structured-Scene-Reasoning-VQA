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


def evaluate(model, rows, conf, vacancy_class=1, iou_thr=0.30):
    total_tp, total_fp, total_fn = 0, 0, 0

    for rec in tqdm(rows, desc=f"conf={conf}"):
        result = model.predict(
            source=rec["image_path"],
            conf=conf,
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
                if int(c) == vacancy_class:
                    preds.append({
                        "bbox": [float(v) for v in box.tolist()],
                        "score": float(cf)
                    })

        gts = rec["gt_missing"]
        tp, fp, fn = match_iou(preds, gts, iou_thr=iou_thr)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "conf": conf,
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
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--iou-thr", type=float, default=0.30)
    parser.add_argument("--vacancy-class", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    yolo_root = processed_root / "yolo_retaildet_style"
    val_rows = read_jsonl(yolo_root / "val_manifest.jsonl")
    test_rows = read_jsonl(yolo_root / "test_manifest.jsonl")

    model = YOLO(args.weights)

    confs = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    val_results = []
    for c in confs:
        val_results.append(
            evaluate(
                model,
                val_rows,
                conf=c,
                vacancy_class=args.vacancy_class,
                iou_thr=args.iou_thr
            )
        )

    val_df = pd.DataFrame(val_results)
    best_row = val_df.sort_values("F1", ascending=False).iloc[0]
    best_conf = float(best_row["conf"])

    test_result = evaluate(
        model,
        test_rows,
        conf=best_conf,
        vacancy_class=args.vacancy_class,
        iou_thr=args.iou_thr
    )

    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    val_path = table_dir / "retaildet_style_yolo_conf_tuning_val.csv"
    test_path = table_dir / "retaildet_style_yolo_test_val_tuned.csv"

    val_df_round = val_df.copy()
    for col in ["Precision", "Recall", "F1"]:
        val_df_round[col] = val_df_round[col].round(4)
    val_df_round.to_csv(val_path, index=False, encoding="utf-8-sig")

    test_df = pd.DataFrame([test_result])
    for col in ["Precision", "Recall", "F1"]:
        test_df[col] = test_df[col].round(4)
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("[Best val conf]")
    print(best_row)
    print("\n[Test with val-tuned conf]")
    print(test_df)
    print("Saved:", val_path)
    print("Saved:", test_path)
    print("=" * 80)


if __name__ == "__main__":
    main()