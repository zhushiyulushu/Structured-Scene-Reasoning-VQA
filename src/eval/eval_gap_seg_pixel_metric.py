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


def label_to_mask(label_path, img_w, img_h):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if not label_path.exists():
        return mask

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
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


def predict_mask(model, img_path, img_w, img_h, conf, imgsz):
    result = model.predict(
        source=str(img_path),
        conf=conf,
        iou=0.5,
        imgsz=imgsz,
        verbose=False
    )[0]

    pred = np.zeros((img_h, img_w), dtype=np.uint8)

    if result.masks is None:
        return pred

    masks = result.masks.data.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy() if result.boxes is not None else np.zeros(len(masks))

    for m, c in zip(masks, cls):
        if int(c) != 0:
            continue
        m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        pred[m > 0.5] = 1

    return pred


def compute_metrics(pred, gt):
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


def eval_split(model, root, split, conf, imgsz):
    img_dir = root / "images" / split
    lab_dir = root / "labels" / split

    image_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])

    rows = []
    global_tp, global_fp, global_fn = 0, 0, 0

    for img_path in tqdm(image_paths, desc=f"Pixel metric {split} conf={conf}"):
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        label_path = lab_dir / img_path.with_suffix(".txt").name

        gt = label_to_mask(label_path, img_w, img_h)
        pred = predict_mask(model, img_path, img_w, img_h, conf=conf, imgsz=imgsz)

        p, r, f1, iou, tp, fp, fn = compute_metrics(pred, gt)

        global_tp += tp
        global_fp += fp
        global_fn += fn

        rows.append({
            "image": img_path.name,
            "precision": p,
            "recall": r,
            "f1": f1,
            "iou": iou,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_pixels": int(gt.sum()),
            "pred_pixels": int(pred.sum()),
        })

    df = pd.DataFrame(rows)

    macro_p = df["precision"].mean()
    macro_r = df["recall"].mean()
    macro_f1 = df["f1"].mean()
    macro_iou = df["iou"].mean()

    global_p = global_tp / max(global_tp + global_fp, 1)
    global_r = global_tp / max(global_tp + global_fn, 1)
    global_f1 = 2 * global_p * global_r / max(global_p + global_r, 1e-8)
    global_iou = global_tp / max(global_tp + global_fp + global_fn, 1)

    summary = {
        "split": split,
        "conf": conf,
        "images": len(df),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "macro_iou": macro_iou,
        "global_precision": global_p,
        "global_recall": global_r,
        "global_f1": global_f1,
        "global_iou": global_iou,
    }

    return df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--sweep", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    root = processed_root / "yolo_gap_seg"
    model = YOLO(args.weights)

    table_dir = output_root / "tables"
    result_dir = output_root / "results"
    table_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        confs = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
        summaries = []

        for c in confs:
            _, summary = eval_split(model, root, args.split, conf=c, imgsz=args.imgsz)
            summaries.append(summary)

        out = pd.DataFrame(summaries)
        out_path = table_dir / f"gap_seg_pixel_sweep_{args.split}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")

        best = out.sort_values("global_f1", ascending=False).iloc[0]
        print("=" * 80)
        print(out)
        print("\n[Best by global_f1]")
        print(best)
        print("Saved:", out_path)
        print("=" * 80)

    else:
        detail, summary = eval_split(model, root, args.split, conf=args.conf, imgsz=args.imgsz)

        detail_path = result_dir / f"gap_seg_pixel_detail_{args.split}_conf_{args.conf}.csv"
        summary_path = table_dir / f"gap_seg_pixel_summary_{args.split}_conf_{args.conf}.csv"

        detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
        pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")

        print("=" * 80)
        print(pd.DataFrame([summary]))
        print("Saved:", summary_path)
        print("Detail:", detail_path)
        print("=" * 80)


if __name__ == "__main__":
    main()