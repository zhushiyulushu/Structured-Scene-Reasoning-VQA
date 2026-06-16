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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num", type=int, default=30)
    parser.add_argument("--only-correct", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    result_path = output_root / "results" / f"learnable_grid_{args.split}_seed_{args.seed}.json"
    masked_path = processed_root / "masked" / f"{args.split}_masked_img.jsonl"

    result = read_json(result_path)
    masked_map = read_jsonl(masked_path)

    out_dir = output_root / "figures" / "learnable_grid_vis" / args.split / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    font = get_font(18)
    count = 0

    samples = result["per_sample"]

    for sample in samples:
        if count >= args.num:
            break

        if args.only_correct and int(sample["correct"]) != 1:
            continue

        masked_id = sample["masked_id"]
        if masked_id not in masked_map:
            continue

        rec = masked_map[masked_id]
        img_path = rec.get("masked_image_path", rec["image_path"])
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for obj in rec["observed_objects"]:
            draw_box(draw, obj["bbox"], color=(0, 180, 0), width=1)

        gt_box = sample["gt_bbox"]
        pred_box = sample["pred_bbox"]

        draw_box(
            draw,
            gt_box,
            color=(255, 180, 0),
            width=5,
            text="GT",
            font=font
        )

        draw_box(
            draw,
            pred_box,
            color=(255, 0, 0),
            width=5,
            text=f"Pred IoU={sample['iou']:.2f}",
            font=font
        )

        title = f"Learned Grid Completion | correct={sample['correct']} | IoU={sample['iou']:.3f}"
        draw.rectangle([10, 10, min(img.width - 10, 800), 45], fill=(0, 0, 0))
        draw.text((15, 15), title, fill=(255, 255, 255), font=font)

        out_name = f"{count:03d}_{masked_id.replace('/', '_').replace(':', '_')}.jpg"
        out_path = out_dir / out_name
        img.save(out_path)

        count += 1

    print("=" * 80)
    print("[完成] Learnable Grid 可视化")
    print("输出目录:", out_dir)
    print("生成数量:", count)
    print("=" * 80)


if __name__ == "__main__":
    main()