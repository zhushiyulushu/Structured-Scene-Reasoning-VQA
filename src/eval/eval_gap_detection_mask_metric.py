from pathlib import Path
import argparse
import yaml
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw
from ultralytics import YOLO


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_all_images(dataset_dir):
    return [
        p for p in dataset_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]


def find_all_masks(dataset_dir):
    return [
        p for p in dataset_dir.rglob("*")
        if p.suffix.lower() in {".tiff", ".tif", ".png"} and "Masks" in str(p)
    ]


def find_image_for_mask(mask_path, image_paths):
    stem = mask_path.stem

    candidates = []
    for img in image_paths:
        s = img.stem
        if s == stem or s == f"{stem}_Image" or f"{stem}" in s:
            candidates.append(img)

    if len(candidates) == 0:
        return None

    candidates = sorted(candidates, key=lambda p: len(str(p)))
    return candidates[0]


def load_gt_mask(mask_path, img_w, img_h, auto_invert=True):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # resize to image size
    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    binary = (mask > 127).astype(np.uint8)
    inv_binary = 1 - binary

    if auto_invert:
        # gap region should usually be the minority region
        r1 = binary.mean()
        r2 = inv_binary.mean()

        if r1 == 0 and r2 > 0:
            gt = inv_binary
        elif r2 == 0 and r1 > 0:
            gt = binary
        else:
            gt = binary if r1 <= r2 else inv_binary
    else:
        gt = binary

    # remove tiny noise
    gt = (gt * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    gt = cv2.morphologyEx(gt, cv2.MORPH_OPEN, kernel, iterations=1)
    gt = (gt > 0).astype(np.uint8)

    return gt


def pred_boxes_to_mask(pred_boxes, img_w, img_h):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for b in pred_boxes:
        x1, y1, x2, y2 = b
        x1 = int(max(0, min(img_w - 1, round(x1))))
        y1 = int(max(0, min(img_h - 1, round(y1))))
        x2 = int(max(0, min(img_w - 1, round(x2))))
        y2 = int(max(0, min(img_h - 1, round(y2))))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1

    return mask


def mask_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)

    return precision, recall, f1, iou, int(tp), int(fp), int(fn)


def draw_debug(img_path, gt_mask, pred_mask, out_path):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).copy()

    # GT: yellow overlay
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)

    arr[gt] = (0.6 * arr[gt] + 0.4 * np.array([255, 180, 0])).astype(np.uint8)
    arr[pred] = (0.6 * arr[pred] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)

    out = Image.fromarray(arr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def evaluate_dataset(model, dataset_dir, dataset_name, conf, vacancy_class, auto_invert, vis_dir, vis_num):
    image_paths = find_all_images(dataset_dir)
    mask_paths = find_all_masks(dataset_dir)

    pairs = []
    for m in mask_paths:
        img = find_image_for_mask(m, image_paths)
        if img is not None:
            pairs.append((img, m))

    rows = []
    vis_count = 0

    for img_path, mask_path in tqdm(pairs, desc=f"GapD mask {dataset_name}"):
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        gt = load_gt_mask(mask_path, img_w, img_h, auto_invert=auto_invert)
        if gt is None:
            continue

        result = model.predict(
            source=str(img_path),
            conf=conf,
            iou=0.5,
            imgsz=640,
            verbose=False
        )[0]

        pred_boxes = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()

            for box, c in zip(xyxy, cls):
                if int(c) == vacancy_class:
                    pred_boxes.append([float(v) for v in box.tolist()])

        pred = pred_boxes_to_mask(pred_boxes, img_w, img_h)

        p, r, f1, iou, tp, fp, fn = mask_metrics(pred, gt)

        rows.append({
            "Dataset": dataset_name,
            "Image": str(img_path),
            "Mask": str(mask_path),
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "IoU": iou,
            "TP_pixels": tp,
            "FP_pixels": fp,
            "FN_pixels": fn,
            "GT_pixels": int(gt.sum()),
            "Pred_pixels": int(pred.sum())
        })

        if vis_count < vis_num:
            out_path = vis_dir / dataset_name / f"{vis_count:03d}_{img_path.name}"
            draw_debug(img_path, gt, pred, out_path)
            vis_count += 1

    return rows


def summarize(df):
    rows = []

    for dataset, g in df.groupby("Dataset"):
        rows.append({
            "Dataset": dataset,
            "Images": len(g),
            "Precision": g["Precision"].mean(),
            "Recall": g["Recall"].mean(),
            "F1": g["F1"].mean(),
            "IoU": g["IoU"].mean()
        })

    rows.append({
        "Dataset": "Overall",
        "Images": len(df),
        "Precision": df["Precision"].mean(),
        "Recall": df["Recall"].mean(),
        "F1": df["F1"].mean(),
        "IoU": df["IoU"].mean()
    })

    out = pd.DataFrame(rows)
    for c in ["Precision", "Recall", "F1", "IoU"]:
        out[c] = out[c].round(4)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--vacancy-class", type=int, default=1)
    parser.add_argument("--auto-invert", type=int, default=1)
    parser.add_argument("--vis-num", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    gap_root = Path(cfg["gap_root"])
    output_root = Path(cfg["output_root"])

    model = YOLO(args.weights)

    all_rows = []
    vis_dir = output_root / "figures" / "gap_detection_mask_metric_vis"

    for dataset_dir in gap_root.iterdir():
        if not dataset_dir.is_dir():
            continue

        rows = evaluate_dataset(
            model=model,
            dataset_dir=dataset_dir,
            dataset_name=dataset_dir.name,
            conf=args.conf,
            vacancy_class=args.vacancy_class,
            auto_invert=bool(args.auto_invert),
            vis_dir=vis_dir,
            vis_num=args.vis_num
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    table_dir = output_root / "tables"
    result_dir = output_root / "results"
    table_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    detail_path = result_dir / "gap_detection_mask_metric_detail.csv"
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary = summarize(df)
    summary_path = table_dir / "table3_gap_detection_mask_metric.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(summary)
    print("Saved:", summary_path)
    print("Details:", detail_path)
    print("Visualizations:", vis_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()