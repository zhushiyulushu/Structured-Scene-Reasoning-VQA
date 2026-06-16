from pathlib import Path
import argparse
import json
import yaml
from PIL import Image, ImageDraw, ImageFont


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                mapping[rec["masked_id"]] = rec
    return mapping


def get_font(size=18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_box(draw, box, color, width=4, text=None, font=None):
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    if text:
        tx, ty = x1, max(0, y1 - 22)
        draw.rectangle([tx, ty, tx + max(100, len(text) * 8), ty + 22], fill=color)
        draw.text((tx + 3, ty + 3), text, fill=(255, 255, 255), font=font)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    result_path = output_root / "results" / f"oracle_dsi_{args.split}.json"
    masked_path = processed_root / "masked" / f"{args.split}_masked_img.jsonl"

    result = read_json(result_path)
    masked_map = read_jsonl(masked_path)

    out_dir = output_root / "figures" / "failure_cases" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    font = get_font(18)
    count = 0

    # 筛选失败：有 FP 或 FN
    failures = [
        s for s in result["per_sample"]
        if s.get("fp", 0) > 0 or s.get("fn", 0) > 0
    ]

    for sample in failures:
        if count >= args.num:
            break

        masked_id = sample["masked_id"]
        if masked_id not in masked_map:
            continue

        rec = masked_map[masked_id]
        img_path = rec.get("masked_image_path", rec["image_path"])
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for obj in rec["observed_objects"]:
            draw_box(draw, obj["bbox"], color=(0, 180, 0), width=1)

        for gt in sample["gt_missing"]:
            draw_box(
                draw,
                gt["bbox"],
                color=(255, 180, 0),
                width=5,
                text=f"GT r{int(gt['row'])+1}c{int(gt['col'])+1}",
                font=font
            )

        for pred in sample["pred_missing"]:
            draw_box(
                draw,
                pred["bbox"],
                color=(255, 0, 0),
                width=5,
                text=f"Pred r{int(pred['row'])+1}c{int(pred['col'])+1}",
                font=font
            )

        title = f"Failure case | TP={sample['tp']} FP={sample['fp']} FN={sample['fn']}"
        draw.rectangle([10, 10, 600, 45], fill=(0, 0, 0))
        draw.text((15, 15), title, fill=(255, 255, 255), font=font)

        out_path = out_dir / f"{count:03d}_{masked_id.replace('/', '_').replace(':', '_')}.jpg"
        img.save(out_path)
        count += 1

    print("=" * 80)
    print("[完成] failure cases visualization")
    print(f"输出目录：{out_dir}")
    print(f"生成数量：{count}")
    print("=" * 80)


if __name__ == "__main__":
    main()