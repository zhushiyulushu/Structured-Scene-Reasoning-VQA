from pathlib import Path
import argparse
import json
import yaml
from tqdm import tqdm

from src.reasoning.dsi import infer_missing_by_dsi, box_iou


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def match_predictions(preds, gts, iou_thr=0.30):
    """
    匹配规则：
    1. row/col 必须一致；
    2. bbox IoU >= iou_thr。
    """
    matched_pred = set()
    tp = 0

    for gt in gts:
        gt_row = int(gt["row"])
        gt_col = int(gt["col"])
        gt_box = gt["bbox"]

        best_idx = None
        best_iou = 0.0

        for i, pred in enumerate(preds):
            if i in matched_pred:
                continue

            if int(pred["row"]) != gt_row:
                continue
            if int(pred["col"]) != gt_col:
                continue

            iou = box_iou(pred["bbox"], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx is not None and best_iou >= iou_thr:
            matched_pred.add(best_idx)
            tp += 1

    fp = len(preds) - tp
    fn = len(gts) - tp

    return tp, fp, fn


def evaluate_vqa_from_predictions(preds, gts):
    """
    结构化 VQA 评价。
    因为答案由 missing set 确定性生成，所以评价 missing set 是否正确即可。
    """
    gt_locs = {(int(g["row"]), int(g["col"])) for g in gts}
    pred_locs = {(int(p["row"]), int(p["col"])) for p in preds}

    count_correct = int(len(preds) == len(gts))
    locate_correct = int(gt_locs == pred_locs)
    verify_correct = int(len(preds) > 0 and len(gts) > 0)
    identify_correct = locate_correct  # 当前 SKU-110K 是 object 单类，identity 依赖位置正确

    return {
        "count_correct": count_correct,
        "locate_correct": locate_correct,
        "verify_correct": verify_correct,
        "identify_correct": identify_correct,
    }


def eval_split(split, cfg, sigma=0.15, iou_thr=0.30):
    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    # 优先使用 image-level masked jsonl
    in_path = processed_root / "masked" / f"{split}_masked_img.jsonl"
    if not in_path.exists():
        in_path = processed_root / "masked" / f"{split}_masked.jsonl"

    records = read_jsonl(in_path)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    count_correct = 0
    locate_correct = 0
    verify_correct = 0
    identify_correct = 0

    per_sample = []

    for rec in tqdm(records, desc=f"Oracle DSI eval {split}"):
        preds = infer_missing_by_dsi(rec, sigma=sigma)
        gts = rec["missing_objects"]

        tp, fp, fn = match_predictions(preds, gts, iou_thr=iou_thr)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        vqa = evaluate_vqa_from_predictions(preds, gts)
        count_correct += vqa["count_correct"]
        locate_correct += vqa["locate_correct"]
        verify_correct += vqa["verify_correct"]
        identify_correct += vqa["identify_correct"]

        per_sample.append({
            "masked_id": rec["masked_id"],
            "image_id": rec["image_id"],
            "gt_missing": gts,
            "pred_missing": preds,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            **vqa
        })

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    n = len(records)
    result = {
        "split": split,
        "num_samples": n,
        "sigma": sigma,
        "iou_thr": iou_thr,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "vqa_count_acc": count_correct / max(n, 1),
        "vqa_locate_acc": locate_correct / max(n, 1),
        "vqa_verify_acc": verify_correct / max(n, 1),
        "vqa_identify_acc": identify_correct / max(n, 1),
        "per_sample": per_sample
    }

    out_dir = output_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"oracle_dsi_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("=" * 80)
    print(f"[Oracle DSI] split={split}")
    print(f"samples: {n}")
    print(f"TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    print(f"VQA count acc={result['vqa_count_acc']:.4f}")
    print(f"VQA locate acc={result['vqa_locate_acc']:.4f}")
    print(f"VQA verify acc={result['vqa_verify_acc']:.4f}")
    print(f"VQA identify acc={result['vqa_identify_acc']:.4f}")
    print(f"saved to: {out_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--iou-thr", type=float, default=0.30)
    args = parser.parse_args()

    cfg = load_config(args.config)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        eval_split(split, cfg, sigma=args.sigma, iou_thr=args.iou_thr)


if __name__ == "__main__":
    main()