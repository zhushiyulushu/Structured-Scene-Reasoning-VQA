from pathlib import Path
import argparse
import json
import re
import yaml
from tqdm import tqdm
import pandas as pd
from PIL import Image

import torch
from transformers import BlipProcessor, BlipForQuestionAnswering


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


def build_masked_map(masked_path):
    m = {}
    for rec in read_jsonl(masked_path):
        m[rec["masked_id"]] = rec
    return m


def parse_count(text):
    text = text.lower()
    nums = re.findall(r"\d+", text)
    if nums:
        return int(nums[0])
    if "one" in text or "a missing" in text:
        return 1
    if "no" in text or "none" in text:
        return 0
    return None


def parse_yes_no(text):
    text = text.lower()
    if text.startswith("yes") or " yes" in text:
        return "yes"
    if text.startswith("no") or " no" in text:
        return "no"
    return None


def parse_row_col(text):
    text = text.lower()
    row = None
    col = None

    m_row = re.search(r"row\s*(\d+)", text)
    m_col = re.search(r"(column|col)\s*(\d+)", text)

    if m_row:
        row = int(m_row.group(1))
    if m_col:
        col = int(m_col.group(2))

    return row, col


def eval_answer(qa, pred_text):
    qtype = qa["type"]
    gts = qa["missing_objects"]

    if qtype == "count":
        pred_count = parse_count(pred_text)
        return int(pred_count == len(gts))

    if qtype == "verify":
        gt = "yes" if len(gts) > 0 else "no"
        pred = parse_yes_no(pred_text)
        return int(pred == gt)

    if qtype == "locate":
        gt_locs = {(int(g["row"]) + 1, int(g["col"]) + 1) for g in gts}
        row, col = parse_row_col(pred_text)
        if row is None or col is None:
            return 0
        return int((row, col) in gt_locs)

    if qtype == "identify":
        t = pred_text.lower()
        return int(any(w in t for w in ["item", "product", "object", "goods", "bottle", "box"]))

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--model-name", type=str, default="Salesforce/blip-vqa-base")
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    qa_path = processed_root / "vqa" / f"{args.split}_qa.jsonl"
    masked_path = processed_root / "masked" / f"{args.split}_masked_img.jsonl"

    qa_records = read_jsonl(qa_path)
    if args.max_samples > 0:
        qa_records = qa_records[:args.max_samples]

    masked_map = build_masked_map(masked_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForQuestionAnswering.from_pretrained(args.model_name).to(device)
    model.eval()

    rows = []
    correct_by_type = {}
    total_by_type = {}

    for qa in tqdm(qa_records, desc="BLIP VQA baseline"):
        masked_id = qa["masked_id"]
        rec = masked_map[masked_id]
        image_path = rec.get("masked_image_path", rec["image_path"])

        image = Image.open(image_path).convert("RGB")
        question = qa["question"]

        inputs = processor(images=image, text=question, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20)

        pred = processor.decode(out[0], skip_special_tokens=True)
        correct = eval_answer(qa, pred)
        qtype = qa["type"]

        correct_by_type[qtype] = correct_by_type.get(qtype, 0) + correct
        total_by_type[qtype] = total_by_type.get(qtype, 0) + 1

        rows.append({
            "qid": qa["qid"],
            "type": qtype,
            "question": question,
            "gt_answer": qa["answer"],
            "pred_answer": pred,
            "correct": correct,
            "image_path": image_path
        })

    out_dir = output_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    pred_csv = out_dir / f"blip_baseline_{args.split}_{args.max_samples}.csv"
    df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    summary = []
    for t in sorted(total_by_type.keys()):
        summary.append({
            "type": t,
            "correct": correct_by_type.get(t, 0),
            "total": total_by_type[t],
            "accuracy": correct_by_type.get(t, 0) / max(total_by_type[t], 1)
        })

    summary_df = pd.DataFrame(summary)
    summary_csv = output_root / "tables" / f"blip_baseline_summary_{args.split}_{args.max_samples}.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(summary_df)
    print("=" * 80)
    print("Prediction CSV:", pred_csv)
    print("Summary CSV:", summary_csv)
    print("=" * 80)


if __name__ == "__main__":
    main()