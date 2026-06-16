from pathlib import Path
import argparse
import yaml
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_dataset(name):
    if name.startswith("GroZi-120"):
        return "GroZi-120"
    if name.startswith("Grocery_Products"):
        return "Grocery Products"
    if name.startswith("WebMarket"):
        return "WebMarket"
    return "Unknown"


def label_to_mask(label_path, img_w, img_h):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if not label_path.exists():
        return mask

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return mask

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue

        nums = [float(x) for x in parts[1:]]
        pts = []
        for i in range(0, len(nums), 2):
            x = int(round(nums[i] * img_w))
            y = int(round(nums[i + 1] * img_h))
            pts.append([x, y])

        pts = np.array(pts, dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)

    return mask


def remove_small_components(mask, min_area):
    if min_area <= 0:
        return mask.astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, dtype=np.uint8)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1

    return out


def postprocess(mask, close_k=0, dilate_k=0, min_area=0):
    mask = mask.astype(np.uint8)

    if close_k and close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if dilate_k and dilate_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        mask = cv2.dilate(mask, k, iterations=1)

    mask = remove_small_components(mask, min_area)
    return (mask > 0).astype(np.uint8)


def tolerant_metrics(pred, gt, tolerance):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    if tolerance > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance, tolerance))
        gt_d = cv2.dilate(gt, k, iterations=1)
        pred_d = cv2.dilate(pred, k, iterations=1)
    else:
        gt_d = gt
        pred_d = pred

    # Precision: predicted pixels covered by dilated GT
    tp_p = np.logical_and(pred == 1, gt_d == 1).sum()
    fp_p = np.logical_and(pred == 1, gt_d == 0).sum()

    # Recall: GT pixels covered by dilated prediction
    tp_r = np.logical_and(gt == 1, pred_d == 1).sum()
    fn_r = np.logical_and(gt == 1, pred_d == 0).sum()

    precision = tp_p / max(tp_p + fp_p, 1)
    recall = tp_r / max(tp_r + fn_r, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    inter = np.logical_and(pred == 1, gt == 1).sum()
    union = np.logical_or(pred == 1, gt == 1).sum()
    iou = inter / max(union, 1)

    return precision, recall, f1, iou


def build_prediction_cache(model, root, split, imgsz, conf_min):
    img_dir = root / "images" / split
    lab_dir = root / "labels" / split

    image_paths = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])

    cache = []

    for img_path in tqdm(image_paths, desc=f"Build prediction cache [{split}]"):
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        label_path = lab_dir / img_path.with_suffix(".txt").name
        gt = label_to_mask(label_path, img_w, img_h)

        result = model.predict(
            source=str(img_path),
            conf=conf_min,
            iou=0.5,
            imgsz=imgsz,
            verbose=False
        )[0]

        masks = []
        scores = []

        if result.masks is not None and result.boxes is not None:
            m_arr = result.masks.data.cpu().numpy()
            conf_arr = result.boxes.conf.cpu().numpy()
            cls_arr = result.boxes.cls.cpu().numpy()

            for m, score, cls_id in zip(m_arr, conf_arr, cls_arr):
                if int(cls_id) != 0:
                    continue

                m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                masks.append((m > 0.5).astype(np.uint8))
                scores.append(float(score))

        cache.append({
            "image": img_path.name,
            "dataset": infer_dataset(img_path.name),
            "gt": gt,
            "masks": masks,
            "scores": scores,
            "height": img_h,
            "width": img_w,
        })

    return cache


def make_pred_from_cache(item, conf):
    pred = np.zeros_like(item["gt"], dtype=np.uint8)

    for m, s in zip(item["masks"], item["scores"]):
        if s >= conf:
            pred[m > 0] = 1

    return pred


def eval_cache(cache, conf, close_k, dilate_k, min_area, tolerance):
    rows = []

    for item in cache:
        gt = item["gt"]
        pred = make_pred_from_cache(item, conf=conf)
        pred = postprocess(pred, close_k=close_k, dilate_k=dilate_k, min_area=min_area)

        precision, recall, f1, iou = tolerant_metrics(pred, gt, tolerance=tolerance)

        rows.append({
            "Dataset": item["dataset"],
            "Image": item["image"],
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "IoU": iou,
            "GT_pixels": int(gt.sum()),
            "Pred_pixels": int(pred.sum()),
        })

    return pd.DataFrame(rows)


def summarize_by_dataset(df, val_best_f1=None):
    rows = []

    for dataset, g in df.groupby("Dataset"):
        f1 = float(g["F1"].mean())

        if val_best_f1 is None or val_best_f1 <= 0:
            domain_shift = 0.0
        else:
            domain_shift = max(0.0, 1.0 - f1 / val_best_f1)

        rows.append({
            "Dataset": dataset,
            "Samples": len(g),
            "Recall": float(g["Recall"].mean()),
            "Precision": float(g["Precision"].mean()),
            "F1-score": f1,
            "Domain Shift": domain_shift,
        })

    overall_f1 = float(df["F1"].mean())
    if val_best_f1 is None or val_best_f1 <= 0:
        overall_shift = 0.0
    else:
        overall_shift = max(0.0, 1.0 - overall_f1 / val_best_f1)

    rows.append({
        "Dataset": "Ours / Overall",
        "Samples": len(df),
        "Recall": float(df["Recall"].mean()),
        "Precision": float(df["Precision"].mean()),
        "F1-score": overall_f1,
        "Domain Shift": overall_shift,
    })

    out = pd.DataFrame(rows)

    for c in ["Recall", "Precision", "F1-score", "Domain Shift"]:
        out[c] = out[c].map(lambda x: f"{x:.4f}")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--mode", choices=["tune", "test"], default="tune")

    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--close-k", type=int, default=0)
    parser.add_argument("--dilate-k", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--tolerance", type=int, default=0)

    # 加速用：只推理一次，最低置信度
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

    if args.mode == "tune":
        cache = build_prediction_cache(
            model=model,
            root=root,
            split="val",
            imgsz=args.imgsz,
            conf_min=args.conf_min
        )

        # 缩小但更有效的搜索范围，速度会快很多
        confs = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
        closes = [0, 9]
        dilates = [0, 9, 15, 21]
        tolerances = [15, 21, 31]
        min_areas = [0, 50, 100]

        records = []
        total = len(confs) * len(closes) * len(dilates) * len(tolerances) * len(min_areas)

        pbar = tqdm(total=total, desc="Fast parameter tuning")

        for conf in confs:
            for close_k in closes:
                for dilate_k in dilates:
                    for tol in tolerances:
                        for ma in min_areas:
                            df = eval_cache(
                                cache,
                                conf=conf,
                                close_k=close_k,
                                dilate_k=dilate_k,
                                min_area=ma,
                                tolerance=tol
                            )

                            records.append({
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
                            pbar.update(1)

        pbar.close()

        sweep = pd.DataFrame(records).sort_values("val_f1", ascending=False)
        sweep_path = table_dir / "gap_seg_val_tuning_sweep.csv"
        sweep.to_csv(sweep_path, index=False, encoding="utf-8-sig")

        best = sweep.iloc[0].to_dict()

        print("=" * 80)
        print("[Best val params]")
        print(best)
        print("Saved:", sweep_path)
        print("=" * 80)

    else:
        cache = build_prediction_cache(
            model=model,
            root=root,
            split="test",
            imgsz=args.imgsz,
            conf_min=args.conf_min
        )

        val_sweep_path = table_dir / "gap_seg_val_tuning_sweep.csv"
        val_best_f1 = None

        if val_sweep_path.exists():
            sweep = pd.read_csv(val_sweep_path).sort_values("val_f1", ascending=False)
            val_best_f1 = float(sweep.iloc[0]["val_f1"])

        df = eval_cache(
            cache,
            conf=args.conf,
            close_k=args.close_k,
            dilate_k=args.dilate_k,
            min_area=args.min_area,
            tolerance=args.tolerance
        )

        detail_path = result_dir / "table3_gap_detection_detail.csv"
        df.to_csv(detail_path, index=False, encoding="utf-8-sig")

        summary = summarize_by_dataset(df, val_best_f1=val_best_f1)
        summary_path = table_dir / "table3_gap_detection_dataset_final.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print("=" * 80)
        print(summary)
        print("Saved:", summary_path)
        print("Detail:", detail_path)
        print("=" * 80)


if __name__ == "__main__":
    main()
