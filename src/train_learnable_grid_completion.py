from pathlib import Path
import argparse
import csv
import json
import random
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def xyxy_to_xywh_norm(box, width, height):
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return [xc, yc, w, h]


def xywh_norm_to_xyxy(box, width, height):
    xc, yc, w, h = box
    xc *= width
    yc *= height
    w *= width
    h *= height
    return [
        float(xc - w / 2.0),
        float(yc - h / 2.0),
        float(xc + w / 2.0),
        float(yc + h / 2.0),
    ]


def center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def build_sample_from_record(rec):
    """
    每条 masked record 里通常有 1 个 missing object。
    找到它左右邻居，用左右邻居和结构信息预测 missing bbox。
    """
    missing = rec["missing_objects"][0]
    r = int(missing["row"])
    c = int(missing["col"])

    obs_map = {(int(o["row"]), int(o["col"])): o for o in rec["observed_objects"]}

    left = obs_map.get((r, c - 1), None)
    right = obs_map.get((r, c + 1), None)

    if left is None or right is None:
        return None

    width = float(rec["width"])
    height = float(rec["height"])

    left_norm = xyxy_to_xywh_norm(left["bbox"], width, height)
    right_norm = xyxy_to_xywh_norm(right["bbox"], width, height)
    target_norm = xyxy_to_xywh_norm(missing["bbox"], width, height)

    lx, ly = center(left["bbox"])
    rx, ry = center(right["bbox"])

    lw = left["bbox"][2] - left["bbox"][0]
    lh = left["bbox"][3] - left["bbox"][1]
    rw = right["bbox"][2] - right["bbox"][0]
    rh = right["bbox"][3] - right["bbox"][1]

    num_rows = max(float(rec.get("num_rows", 1)), 1.0)
    max_cols = max(float(rec.get("max_cols", 1)), 1.0)

    # 结构与几何特征
    structural = [
        r / max(num_rows - 1.0, 1.0),
        c / max(max_cols - 1.0, 1.0),
        (c - int(left["col"])) / max(max_cols, 1.0),
        (int(right["col"]) - c) / max(max_cols, 1.0),
        abs(ry - ly) / height,
        abs(rx - lx) / width,
        (rw / max(lw, 1.0)),
        (rh / max(lh, 1.0)),
        ((left["bbox"][2] - left["bbox"][0]) + (right["bbox"][2] - right["bbox"][0])) / (2.0 * width),
        ((left["bbox"][3] - left["bbox"][1]) + (right["bbox"][3] - right["bbox"][1])) / (2.0 * height),
    ]

    x = left_norm + right_norm + structural
    y = target_norm

    return {
        "x": x,
        "y": y,
        "masked_id": rec["masked_id"],
        "image_id": rec["image_id"],
        "row": r,
        "col": c,
        "width": rec["width"],
        "height": rec["height"],
        "gt_bbox": missing["bbox"],
        "left_bbox": left["bbox"],
        "right_bbox": right["bbox"],
    }


class GapCompletionDataset(Dataset):
    def __init__(self, jsonl_path):
        records = read_jsonl(jsonl_path)
        self.samples = []

        for rec in records:
            s = build_sample_from_record(rec)
            if s is not None:
                self.samples.append(s)

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {jsonl_path}")

        self.feature_dim = len(self.samples[0]["x"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = torch.tensor(s["x"], dtype=torch.float32)
        y = torch.tensor(s["y"], dtype=torch.float32)
        return x, y


class LearnableGridCompletion(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def evaluate_model(model, dataset, device, iou_thr=0.30):
    model.eval()

    tp = 0
    fp = 0
    fn = 0
    ious = []

    rows = []

    with torch.no_grad():
        for s in dataset.samples:
            x = torch.tensor(s["x"], dtype=torch.float32).unsqueeze(0).to(device)
            pred_norm = model(x).squeeze(0).cpu().numpy().tolist()

            pred_box = xywh_norm_to_xyxy(pred_norm, s["width"], s["height"])
            gt_box = s["gt_bbox"]

            iou = box_iou(pred_box, gt_box)
            ious.append(iou)

            ok = iou >= iou_thr
            if ok:
                tp += 1
            else:
                fp += 1
                fn += 1

            rows.append({
                "masked_id": s["masked_id"],
                "image_id": s["image_id"],
                "row": s["row"],
                "col": s["col"],
                "iou": iou,
                "correct": int(ok),
                "pred_bbox": pred_box,
                "gt_bbox": gt_box,
            })

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MeanIoU": float(np.mean(ious)),
        "MedianIoU": float(np.median(ious)),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--iou-thr", type=float, default=0.30)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)

    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    train_path = processed_root / "masked" / "train_masked_img.jsonl"
    val_path = processed_root / "masked" / "val_masked_img.jsonl"
    test_path = processed_root / "masked" / "test_masked_img.jsonl"

    train_set = GapCompletionDataset(train_path)
    val_set = GapCompletionDataset(val_path)
    test_set = GapCompletionDataset(test_path)

    print("=" * 80)
    print("Train samples:", len(train_set))
    print("Val samples:", len(val_set))
    print("Test samples:", len(test_set))
    print("Feature dim:", train_set.feature_dim)
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = LearnableGridCompletion(
        input_dim=train_set.feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    run_dir = output_root / "checkpoints" / f"learnable_grid_seed_{args.seed}"
    log_dir = output_root / "logs"
    result_dir = output_root / "results"
    table_dir = output_root / "tables"

    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    best_path = run_dir / "best.pt"

    log_path = log_dir / f"learnable_grid_train_seed_{args.seed}.csv"

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_precision",
                "val_recall",
                "val_f1",
                "val_mean_iou",
                "val_median_iou"
            ]
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_n = 0

            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_n += x.size(0)

            train_loss = total_loss / max(total_n, 1)

            val_metrics = evaluate_model(model, val_set, device, iou_thr=args.iou_thr)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_precision": val_metrics["Precision"],
                "val_recall": val_metrics["Recall"],
                "val_f1": val_metrics["F1"],
                "val_mean_iou": val_metrics["MeanIoU"],
                "val_median_iou": val_metrics["MedianIoU"],
            }
            writer.writerow(row)
            f.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"loss {train_loss:.5f} | "
                f"val F1 {val_metrics['F1']:.4f} | "
                f"mean IoU {val_metrics['MeanIoU']:.4f}"
            )

            if val_metrics["F1"] > best_val_f1:
                best_val_f1 = val_metrics["F1"]
                torch.save({
                    "model_state": model.state_dict(),
                    "input_dim": train_set.feature_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "seed": args.seed,
                    "best_val_f1": best_val_f1,
                }, best_path)

    # load best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    train_metrics = evaluate_model(model, train_set, device, iou_thr=args.iou_thr)
    val_metrics = evaluate_model(model, val_set, device, iou_thr=args.iou_thr)
    test_metrics = evaluate_model(model, test_set, device, iou_thr=args.iou_thr)

    summary_rows = []
    for split, metrics in [
        ("train", train_metrics),
        ("val", val_metrics),
        ("test", test_metrics)
    ]:
        summary_rows.append({
            "Method": "Learnable Grid Completion",
            "Split": split,
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "Precision": round(metrics["Precision"], 4),
            "Recall": round(metrics["Recall"], 4),
            "F1": round(metrics["F1"], 4),
            "MeanIoU": round(metrics["MeanIoU"], 4),
            "MedianIoU": round(metrics["MedianIoU"], 4),
        })

        out_json = result_dir / f"learnable_grid_{split}_seed_{args.seed}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "split": split,
                "metrics": {k: v for k, v in metrics.items() if k != "rows"},
                "per_sample": metrics["rows"]
            }, f, indent=4, ensure_ascii=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = table_dir / f"learnable_grid_summary_seed_{args.seed}.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(summary_df)
    print("=" * 80)
    print("Best checkpoint:", best_path)
    print("Train log:", log_path)
    print("Summary:", summary_csv)
    print("=" * 80)


if __name__ == "__main__":
    main()