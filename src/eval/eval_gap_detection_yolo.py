from pathlib import Path
import argparse
import json
import yaml
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from src.reasoning.dsi import box_iou


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mask_to_boxes(
    mask_path,
    img_w,
    img_h,
    min_area=80,
    max_area_ratio=0.08,
    invert=False,
    ignore_border=True
):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    if invert:
        mask = 255 - mask

    # 二值化
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 形态学去噪，避免大面积粘连
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    mh, mw = binary.shape[:2]
    sx = img_w / float(mw)
    sy = img_h / float(mh)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    image_area = mw * mh

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < min_area:
            continue

        # 过滤过大的区域，避免整张货架/整片商品被当成 gap
        if area / float(image_area) > max_area_ratio:
            continue

        # 过滤贴边大区域，很多 mask 的背景/货架区域会贴边
        if ignore_border:
            touches_border = (
                x <= 1 or y <= 1 or
                x + w >= mw - 2 or
                y + h >= mh - 2
            )
            if touches_border and area / float(image_area) > 0.01:
                continue

        x1 = x * sx
        y1 = y * sy
        x2 = (x + w) * sx
        y2 = (y + h) * sy

        boxes.append([float(x1), float(y1), float(x2), float(y2)])

    return boxes

def match_iou(preds, gts, iou_thr=0.30):
    matched = set()
    tp = 0

    for gt in gts:
        best_i = None
        best_iou = 0.0

        for i, p in enumerate(preds):
            if i in matched:
                continue

            iou = box_iou(p["bbox"], gt)
            if iou > best_iou:
                best_iou = iou
                best_i = i

        if best_i is not None and best_iou >= iou_thr:
            matched.add(best_i)
            tp += 1

    fp = len(preds) - tp
    fn = len(gts) - tp

    return tp, fp, fn


def get_font(size=16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_boxes(image_path, gt_boxes, pred_boxes, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = get_font(16)

    for b in gt_boxes:
        draw.rectangle(b, outline=(255, 180, 0), width=4)
        draw.text((b[0], max(0, b[1] - 18)), "GT", fill=(255, 180, 0), font=font)

    for p in pred_boxes:
        b = p["bbox"]
        draw.rectangle(b, outline=(255, 0, 0), width=4)
        draw.text((b[0], max(0, b[1] - 18)), f"Pred {p['score']:.2f}", fill=(255, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def find_pairs(root, dataset_name, split):
    base = root / dataset_name / split
    img_dir = base / "Images"
    mask_dir = base / "Masks"

    pairs = []

    if not img_dir.exists() or not mask_dir.exists():
        return pairs

    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue

        stem = img_path.stem
        mask_candidates = [
            mask_dir / f"{stem}.tiff",
            mask_dir / f"{stem}.tif",
            mask_dir / f"{stem}.png",
            mask_dir / f"{stem}.jpg",
        ]

        mask_path = None
        for m in mask_candidates:
            if m.exists():
                mask_path = m
                break

        if mask_path is not None:
            pairs.append((img_path, mask_path))

    return pairs


def eval_dataset(model, pairs, dataset_name, split, conf, iou_thr, vacancy_class,
                 min_area, invert, max_area_ratio=0.08, ignore_border=True,
                 vis_dir=None, vis_num=20):
    total_tp, total_fp, total_fn = 0, 0, 0
    per_sample = []
    vis_count = 0
    max_area_ratio=max_area_ratio,
    ignore_border=ignore_border,
    for img_path, mask_path in tqdm(pairs, desc=f"GapDetection {dataset_name}-{split}"):
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        gt_boxes = mask_to_boxes(
            mask_path,
            img_w,
            img_h,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            invert=invert,
            ignore_border=ignore_border
        )
        result = model.predict(
            source=str(img_path),
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

        tp, fp, fn = match_iou(preds, gt_boxes, iou_thr=iou_thr)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        rec = {
            "dataset": dataset_name,
            "split": split,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "num_gt": len(gt_boxes),
            "num_pred": len(preds),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_boxes": gt_boxes,
            "preds": preds
        }
        per_sample.append(rec)

        if vis_dir is not None and vis_count < vis_num:
            out_path = vis_dir / dataset_name / split / f"{vis_count:03d}_{img_path.name}"
            draw_boxes(img_path, gt_boxes, preds, out_path)
            vis_count += 1

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "Dataset": dataset_name,
        "Split": split,
        "Images": len(pairs),
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "per_sample": per_sample
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="Test", choices=["Train", "Test"])
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou-thr", type=float, default=0.30)
    parser.add_argument("--vacancy-class", type=int, default=1)
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--invert", type=int, default=0)
    parser.add_argument("--vis-num", type=int, default=20)
    parser.add_argument("--max-area-ratio", type=float, default=0.08)
    parser.add_argument("--ignore-border", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    gap_root = Path(cfg["gap_root"])
    output_root = Path(cfg["output_root"])

    model = YOLO(args.weights)

    dataset_names = ["GroZi-120", "Grocery Products", "WebMarket"]

    results = []
    all_per_sample = []

    vis_dir = output_root / "figures" / "gap_detection_yolo_vis"

    for d in dataset_names:
        pairs = find_pairs(gap_root, d, args.split)

        if len(pairs) == 0:
            print(f"[跳过] {d}/{args.split}: no image-mask pairs")
            continue

        r = eval_dataset(
            model,
            pairs,
            dataset_name=d,
            split=args.split,
            conf=args.conf,
            iou_thr=args.iou_thr,
            vacancy_class=args.vacancy_class,
            min_area=args.min_area,
            invert=bool(args.invert),
            vis_dir=vis_dir,
            vis_num=args.vis_num
        )

        results.append({k: v for k, v in r.items() if k != "per_sample"})
        all_per_sample.extend(r["per_sample"])

    # overall
    total_tp = sum(r["TP"] for r in results)
    total_fp = sum(r["FP"] for r in results)
    total_fn = sum(r["FN"] for r in results)
    total_images = sum(r["Images"] for r in results)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    results.append({
        "Dataset": "Overall",
        "Split": args.split,
        "Images": total_images,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    table_dir = output_root / "tables"
    result_dir = output_root / "results"
    table_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    for c in ["Precision", "Recall", "F1"]:
        df[c] = df[c].round(4)

    out_csv = table_dir / f"gap_detection_yolo_{args.split.lower()}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_json = result_dir / f"gap_detection_yolo_{args.split.lower()}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "settings": vars(args),
            "summary": results,
            "per_sample": all_per_sample
        }, f, indent=4, ensure_ascii=False)

    print("=" * 80)
    print(df)
    print("Saved:", out_csv)
    print("Saved:", out_json)
    print("Visualizations:", vis_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()