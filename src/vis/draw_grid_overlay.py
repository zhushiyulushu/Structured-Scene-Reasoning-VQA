from pathlib import Path
import argparse
import json
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_font(size=18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_box(draw, box, color, width=3, text=None, font=None):
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if text is not None:
        tx, ty = x1, max(0, y1 - 18)
        draw.rectangle([tx, ty, tx + 90, ty + 18], fill=color)
        draw.text((tx + 2, ty + 2), text, fill=(255, 255, 255), font=font)


def visualize_grid(record, out_path):
    image_path = record.get("masked_image_path", record["image_path"])
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = get_font(16)

    for obj in record["objects"]:
        r = obj["row"] + 1
        c = obj["col"] + 1
        draw_box(
            draw,
            obj["bbox"],
            color=(0, 180, 255),
            width=2,
            text=f"r{r}c{c}",
            font=font
        )

    img.save(out_path)


def visualize_masked(record, out_path):
    img = Image.open(record["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = get_font(16)

    # observed objects：绿色
    for obj in record["observed_objects"]:
        draw_box(
            draw,
            obj["bbox"],
            color=(0, 200, 0),
            width=2,
            text=None,
            font=font
        )

    # missing objects：红色
    for obj in record["missing_objects"]:
        r = obj["row"] + 1
        c = obj["col"] + 1
        draw_box(
            draw,
            obj["bbox"],
            color=(255, 0, 0),
            width=5,
            text=f"M r{r}c{c}",
            font=font
        )

    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--mode", type=str, default="masked", choices=["grid", "masked"])
    parser.add_argument("--num", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["processed_root"])
    out_dir = Path(cfg["output_root"]) / "figures" / "debug_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "grid":
        in_path = processed_root / "grid_labels" / f"{args.split}_grid.jsonl"
        records = read_jsonl(in_path)
        for i, record in enumerate(tqdm(records[:args.num], desc="Drawing grid overlays")):
            out_path = out_dir / f"{args.split}_grid_{i}_{record['image_id']}"
            visualize_grid(record, out_path)

    else:
        candidate = processed_root / "masked" / f"{args.split}_masked_img.jsonl"
        if candidate.exists():
            in_path = candidate
        else:
            in_path = processed_root / "masked" / f"{args.split}_masked.jsonl"
        records = read_jsonl(in_path)
        for i, record in enumerate(tqdm(records[:args.num], desc="Drawing masked overlays")):
            out_path = out_dir / f"{args.split}_masked_{i}_{record['image_id']}"
            visualize_masked(record, out_path)

    print("=" * 80)
    print("[完成] 可视化输出目录：")
    print(out_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()