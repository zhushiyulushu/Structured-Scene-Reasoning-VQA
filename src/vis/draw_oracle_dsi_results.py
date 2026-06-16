from pathlib import Path
import argparse
import json
import yaml
from PIL import Image, ImageDraw, ImageFont


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
        text_w = max(90, len(text) * 8)
        draw.rectangle([tx, ty, tx + text_w, ty + 22], fill=color)
        draw.text((tx + 3, ty + 3), text, fill=(255, 255, 255), font=font)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def visualize_one(sample, out_path):
    # per_sample 里没有 image_path，所以 image_path 放在 sample 的 gt 结构里不一定有
    # 我们在 eval_oracle_dsi.py 里没有存 image_path，因此这里从 result JSON 里拿不到原图路径。
    # 解决方式：后面由 masked jsonl 对 masked_id 做映射。
    pass


def read_jsonl(path):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records[rec["masked_id"]] = rec
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    result_path = output_root / "results" / f"oracle_dsi_{args.split}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"找不到结果文件：{result_path}，请先运行 eval_oracle_dsi.py")

    masked_path = processed_root / "masked" / f"{args.split}_masked_img.jsonl"
    if not masked_path.exists():
        masked_path = processed_root / "masked" / f"{args.split}_masked.jsonl"

    result = read_json(result_path)
    masked_map = read_jsonl(masked_path)

    out_dir = output_root / "figures" / "oracle_dsi_vis" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    font = get_font(18)

    count = 0
    for sample in result["per_sample"]:
        if count >= args.num:
            break

        masked_id = sample["masked_id"]
        if masked_id not in masked_map:
            continue

        rec = masked_map[masked_id]
        image_path = rec.get("masked_image_path", rec["image_path"])

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # observed boxes: green, optional light
        for obj in rec["observed_objects"]:
            draw_box(
                draw,
                obj["bbox"],
                color=(0, 180, 0),
                width=1,
                text=None,
                font=font
            )

        # GT missing: yellow
        for gt in sample["gt_missing"]:
            r = int(gt["row"]) + 1
            c = int(gt["col"]) + 1
            draw_box(
                draw,
                gt["bbox"],
                color=(255, 180, 0),
                width=5,
                text=f"GT r{r}c{c}",
                font=font
            )

        # Predicted missing: red
        for pred in sample["pred_missing"]:
            r = int(pred["row"]) + 1
            c = int(pred["col"]) + 1
            draw_box(
                draw,
                pred["bbox"],
                color=(255, 0, 0),
                width=5,
                text=f"Pred r{r}c{c}",
                font=font
            )

        # Answer text
        pred_locs = [
            f"row {int(p['row']) + 1}, column {int(p['col']) + 1}"
            for p in sample["pred_missing"]
        ]

        if len(pred_locs) == 0:
            answer = "Predicted answer: No missing item detected."
        elif len(pred_locs) == 1:
            answer = f"Predicted answer: Missing item at {pred_locs[0]}."
        else:
            answer = "Predicted answer: Missing items at " + "; ".join(pred_locs) + "."

        draw.rectangle([10, 10, min(img.width - 10, 900), 45], fill=(0, 0, 0))
        draw.text((15, 15), answer, fill=(255, 255, 255), font=font)

        out_name = f"{count:03d}_{masked_id.replace('/', '_').replace(':', '_')}.jpg"
        out_path = out_dir / out_name
        img.save(out_path)

        count += 1

    print("=" * 80)
    print("[完成] Oracle DSI 可视化")
    print(f"输出目录：{out_dir}")
    print(f"生成数量：{count}")
    print("=" * 80)


if __name__ == "__main__":
    main()