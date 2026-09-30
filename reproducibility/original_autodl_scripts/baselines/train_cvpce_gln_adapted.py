import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.ops import box_iou

from cvpce.models import proposals
from cvpce.datautils import (
    generate_gaussians,
    generate_via_simple_and_scaled,
    join_via_max,
)


CONF_LIST = [
    0.03, 0.05, 0.07, 0.10, 0.15,
    0.20, 0.30, 0.40, 0.50
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RetailVacancyDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        train=False,
        with_gaussian=False,
        limit=None,
    ):
        self.root = Path(root)
        self.split = split
        self.train = train
        self.with_gaussian = with_gaussian

        image_dir = self.root / "images" / split
        label_dir = self.root / "labels" / split

        exts = {
            ".jpg", ".jpeg", ".png",
            ".JPG", ".JPEG", ".PNG"
        }

        images = [
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix in exts
        ]
        images = sorted(images)

        if limit is not None:
            images = images[:limit]

        self.samples = []

        for img in images:
            lab = label_dir / f"{img.stem}.txt"
            if not lab.exists():
                raise FileNotFoundError(lab)
            self.samples.append((img, lab))

        print(
            f"{split}: {len(self.samples)} images, "
            f"gaussian={with_gaussian}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        boxes = []
        labels = []
        seen = set()

        with open(label_path, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5:
                    continue

                cls = int(float(p[0]))
                xc, yc, bw, bh = map(float, p[1:5])

                key = (
                    cls,
                    round(xc, 8),
                    round(yc, 8),
                    round(bw, 8),
                    round(bh, 8),
                )

                # Match Ultralytics' duplicate-label removal.
                if key in seen:
                    continue
                seen.add(key)

                x1 = (xc - bw / 2.0) * W
                y1 = (yc - bh / 2.0) * H
                x2 = (xc + bw / 2.0) * W
                y2 = (yc + bh / 2.0) * H

                x1 = max(0.0, min(float(W), x1))
                y1 = max(0.0, min(float(H), y1))
                x2 = max(0.0, min(float(W), x2))
                y2 = max(0.0, min(float(H), y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(cls)

        boxes = torch.tensor(
            boxes,
            dtype=torch.float32
        )

        labels = torch.tensor(
            labels,
            dtype=torch.int64
        )

        if self.train and random.random() < 0.5:
            img = TF.hflip(img)

            old_x1 = boxes[:, 0].clone()
            old_x2 = boxes[:, 2].clone()

            boxes[:, 0] = W - old_x2
            boxes[:, 2] = W - old_x1

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.with_gaussian:
            gaussian = generate_gaussians(
                W,
                H,
                boxes,
                generate_method=generate_via_simple_and_scaled(),
                join_method=join_via_max,
                tanh=True,
            )

            target["gaussians"] = gaussian

        image_tensor = TF.to_tensor(img)

        return image_tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def strip_module_prefix(sd):
    result = {}

    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        result[k] = v

    return result


def load_compatible_pretrained(model, checkpoint):
    state = torch.load(
        checkpoint,
        map_location="cpu"
    )

    sd = state.get(
        "model_state_dict",
        state
    )

    sd = strip_module_prefix(sd)

    current = model.state_dict()

    compatible = {}
    mismatched = []

    for k, v in sd.items():
        if (
            k in current
            and tuple(v.shape)
            == tuple(current[k].shape)
        ):
            compatible[k] = v
        elif k in current:
            mismatched.append(
                (
                    k,
                    tuple(v.shape),
                    tuple(current[k].shape),
                )
            )

    result = model.load_state_dict(
        compatible,
        strict=False
    )

    print(
        "Loaded pretrained tensors:",
        len(compatible),
        "/",
        len(current),
    )

    print("Shape-mismatched tensors:")
    for item in mismatched:
        print("  ", item)

    print(
        "Missing after compatible load:",
        result.missing_keys,
    )

    return mismatched


def greedy_match(
    pred_boxes,
    gt_boxes,
    iou_thr,
):
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    ious = box_iou(
        pred_boxes,
        gt_boxes
    )

    used_p = set()
    used_g = set()

    pairs = []

    for p in range(ious.shape[0]):
        for g in range(ious.shape[1]):
            v = float(ious[p, g])
            if v >= iou_thr:
                pairs.append((v, p, g))

    pairs.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    tp = 0

    for _, p, g in pairs:
        if p in used_p or g in used_g:
            continue

        used_p.add(p)
        used_g.add(g)
        tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn


def metrics_from_counts(tp, fp, fn):
    p = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    r = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * p * r / (p + r)
        if p + r > 0
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": p,
        "Recall": r,
        "F1": f1,
    }


@torch.no_grad()
def evaluate_stream(
    model,
    dataset,
    conf,
    iou_thr,
    device,
):
    model.eval()

    tp = fp = fn = 0

    for i in range(len(dataset)):
        image, target = dataset[i]

        image = image.to(device)

        out = model([image])[0]

        mask = (
            (out["labels"] == 1)
            & (out["scores"] >= conf)
        )

        pred_boxes = (
            out["boxes"][mask]
            .detach()
            .cpu()
        )

        gt_mask = (
            target["labels"] == 1
        )

        gt_boxes = (
            target["boxes"][gt_mask]
            .cpu()
        )

        a, b, c = greedy_match(
            pred_boxes,
            gt_boxes,
            iou_thr,
        )

        tp += a
        fp += b
        fn += c

    result = metrics_from_counts(
        tp,
        fp,
        fn,
    )

    result["conf"] = conf

    model.train()

    return result


@torch.no_grad()
def collect_predictions(
    model,
    dataset,
    device,
):
    model.eval()

    cache = []

    for i in range(len(dataset)):
        image, target = dataset[i]

        image = image.to(device)

        out = model([image])[0]

        cache.append({
            "boxes": out["boxes"].cpu(),
            "scores": out["scores"].cpu(),
            "labels": out["labels"].cpu(),
            "gt_boxes":
                target["boxes"][
                    target["labels"] == 1
                ].cpu(),
        })

    return cache


def evaluate_cache(
    cache,
    conf,
    iou_thr,
):
    tp = fp = fn = 0

    for item in cache:
        mask = (
            (item["labels"] == 1)
            & (item["scores"] >= conf)
        )

        pred_boxes = item["boxes"][mask]
        gt_boxes = item["gt_boxes"]

        a, b, c = greedy_match(
            pred_boxes,
            gt_boxes,
            iou_thr,
        )

        tp += a
        fp += b
        fn += c

    r = metrics_from_counts(
        tp,
        fp,
        fn,
    )

    r["conf"] = conf

    return r


def write_csv(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    keys = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=keys
        )
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True
    )
    parser.add_argument(
        "--source-checkpoint",
        required=True
    )
    parser.add_argument(
        "--output-dir",
        required=True
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=11
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=3
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.30
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None
    )
    parser.add_argument(
        "--limit-val",
        type=int,
        default=None
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=None
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda:0")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print("CVPCE-GLN (Adapted)")
    print("seed =", args.seed)
    print("epochs =", args.epochs)
    print("data =", args.data_root)
    print("output =", out_dir)
    print("=" * 80)

    train_ds = RetailVacancyDataset(
        args.data_root,
        "train",
        train=True,
        with_gaussian=True,
        limit=args.limit_train,
    )

    val_ds = RetailVacancyDataset(
        args.data_root,
        "val",
        train=False,
        with_gaussian=False,
        limit=args.limit_val,
    )

    test_ds = RetailVacancyDataset(
        args.data_root,
        "test",
        train=False,
        with_gaussian=False,
        limit=args.limit_test,
    )

    loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = proposals.gln(
        num_classes=2,
        pretrained_backbone=False,
        tanh=True,
        gaussian_loss_params={
            "tanh": True,
            "negative_threshold": -1,
            "positive_threshold": 0.3,
        },
        detections_per_img=1000,
    )

    # Keep detections down to our lowest
    # confidence sweep threshold.
    if hasattr(model, "score_thresh"):
        model.score_thresh = 0.01

    load_compatible_pretrained(
        model,
        args.source_checkpoint,
    )

    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.0025,
        momentum=0.9,
        weight_decay=0.0001,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .MultiplicativeLR(
            optimizer,
            lr_lambda=lambda _: 0.995,
        )
    )

    best_f1 = -1.0
    best_epoch = -1

    history = []

    best_path = (
        out_dir / "best.pt"
    )

    last_path = (
        out_dir / "last.pt"
    )

    for epoch in range(args.epochs):
        model.train()

        start = time.time()

        sum_cls = 0.0
        sum_box = 0.0
        sum_gauss = 0.0

        n = 0

        for step, (images, targets) in enumerate(loader):
            images = [
                x.to(
                    device,
                    non_blocking=True
                )
                for x in images
            ]

            target_gpu = []

            for t in targets:
                target_gpu.append({
                    k: v.to(
                        device,
                        non_blocking=True
                    )
                    for k, v in t.items()
                })

            optimizer.zero_grad(
                set_to_none=True
            )

            losses = model(
                images,
                target_gpu
            )

            total = (
                losses["classification"]
                + losses["bbox_regression"]
                + losses["gaussian"]
            )

            if not torch.isfinite(total):
                raise RuntimeError(
                    f"Non-finite loss: {losses}"
                )

            total.backward()
            optimizer.step()

            sum_cls += float(
                losses["classification"]
            )
            sum_box += float(
                losses["bbox_regression"]
            )
            sum_gauss += float(
                losses["gaussian"]
            )

            n += 1

            if step % 200 == 0:
                print(
                    f"epoch={epoch:02d} "
                    f"step={step:05d}/{len(loader)} "
                    f"cls={float(losses['classification']):.4f} "
                    f"box={float(losses['bbox_regression']):.4f} "
                    f"gauss={float(losses['gaussian']):.4f}"
                )

        scheduler.step()

        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "seconds": elapsed,
            "classification_loss":
                sum_cls / max(n, 1),
            "bbox_loss":
                sum_box / max(n, 1),
            "gaussian_loss":
                sum_gauss / max(n, 1),
            "val_conf": "",
            "val_precision": "",
            "val_recall": "",
            "val_f1": "",
        }

        should_eval = (
            epoch % args.eval_interval == 0
            or epoch == args.epochs - 1
        )

        if should_eval:
            val_result = evaluate_stream(
                model,
                val_ds,
                conf=0.30,
                iou_thr=args.iou_thr,
                device=device,
            )

            row["val_conf"] = 0.30
            row["val_precision"] = (
                val_result["Precision"]
            )
            row["val_recall"] = (
                val_result["Recall"]
            )
            row["val_f1"] = (
                val_result["F1"]
            )

            print(
                f"EPOCH {epoch} "
                f"elapsed={elapsed/60:.2f} min "
                f"VAL@0.30 "
                f"P={val_result['Precision']:.4f} "
                f"R={val_result['Recall']:.4f} "
                f"F1={val_result['F1']:.4f}"
            )

            if val_result["F1"] > best_f1:
                best_f1 = (
                    val_result["F1"]
                )
                best_epoch = epoch

                torch.save({
                    "model_state_dict":
                        model.state_dict(),
                    "epoch": epoch,
                    "seed": args.seed,
                    "val_f1_conf_030":
                        best_f1,
                }, best_path)

                print(
                    "NEW BEST:",
                    best_path
                )

        else:
            print(
                f"EPOCH {epoch} "
                f"elapsed={elapsed/60:.2f} min"
            )

        torch.save({
            "model_state_dict":
                model.state_dict(),
            "epoch": epoch,
            "seed": args.seed,
        }, last_path)

        history.append(row)

        write_csv(
            out_dir / "training_history.csv",
            history,
        )

    print()
    print(
        "BEST EPOCH =",
        best_epoch,
        "BEST VAL F1@0.30 =",
        best_f1,
    )

    print()
    print("========== FINAL THRESHOLD TUNING ==========")

    best_state = torch.load(
        best_path,
        map_location=device
    )

    model.load_state_dict(
        best_state["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("Caching validation predictions...")
    val_cache = collect_predictions(
        model,
        val_ds,
        device,
    )

    val_rows = []

    for conf in CONF_LIST:
        r = evaluate_cache(
            val_cache,
            conf,
            args.iou_thr,
        )

        val_rows.append(r)

        print(
            f"VAL conf={conf:.2f} "
            f"P={r['Precision']:.4f} "
            f"R={r['Recall']:.4f} "
            f"F1={r['F1']:.4f}"
        )

    best_val = max(
        val_rows,
        key=lambda x: x["F1"]
    )

    best_conf = best_val["conf"]

    print()
    print(
        "BEST VAL CONF =",
        best_conf
    )

    write_csv(
        out_dir / "conf_tuning_val.csv",
        val_rows,
    )

    print("Caching test predictions...")

    test_cache = collect_predictions(
        model,
        test_ds,
        device,
    )

    test_result = evaluate_cache(
        test_cache,
        best_conf,
        args.iou_thr,
    )

    write_csv(
        out_dir / "test_val_tuned.csv",
        [test_result],
    )

    print()
    print("=" * 80)
    print("CVPCE-GLN (Adapted)")
    print("Seed =", args.seed)
    print("Best epoch =", best_epoch)
    print("Best val conf =", best_conf)
    print(
        "TP =", test_result["TP"],
        "FP =", test_result["FP"],
        "FN =", test_result["FN"],
    )
    print(
        "Precision =",
        test_result["Precision"]
    )
    print(
        "Recall =",
        test_result["Recall"]
    )
    print(
        "F1 =",
        test_result["F1"]
    )
    print("=" * 80)

    print("CVPCE_GLN_ADAPTED_PASS")


if __name__ == "__main__":
    main()
