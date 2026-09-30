from pathlib import Path
import argparse
import yaml
import pandas as pd
from tqdm import tqdm


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_raw_annotation(csv_path: Path):
    """
    SKU-110K 常见标注格式为：
    image_name, x1, y1, x2, y2, class_name

    这里不强行假设有没有表头：
    如果第一行是表头，会因为坐标无法转成数字而自动丢弃。
    """
    df = pd.read_csv(csv_path, header=None)

    if df.shape[1] < 5:
        raise ValueError(f"{csv_path} 至少应包含 image_name,x1,y1,x2,y2 五列，但当前只有 {df.shape[1]} 列")

    # 坐标列转数字；如果第一行是 header，会变成 NaN，后面自动删除
    for col in [1, 2, 3, 4]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[0, 1, 2, 3, 4]).copy()

    records = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Reading {csv_path.name}"):
        image_raw = str(row[0]).replace("\\", "/")
        image_id = Path(image_raw).name

        x1 = float(row[1])
        y1 = float(row[2])
        x2 = float(row[3])
        y2 = float(row[4])

        # 如果数据偶然是 x,y,w,h，则做一个兜底修正
        if x2 <= x1:
            x2 = x1 + abs(float(row[3]))
        if y2 <= y1:
            y2 = y1 + abs(float(row[4]))

        class_name = "object"
        if len(row) >= 6 and pd.notna(row[5]):
            class_name = str(row[5])

        records.append({
            "image_id": image_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "class_name": class_name
        })

    return pd.DataFrame(records)


def convert_split(split: str, cfg: dict, limit_images: int = None):
    ann_dir = Path(cfg["sku_annotations"])
    img_dir = Path(cfg["sku_images"])
    out_dir = Path(cfg["processed_root"]) / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = ann_dir / f"annotations_{split}.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"找不到文件：{raw_csv}")

    df = read_raw_annotation(raw_csv)
    df["split"] = split

    # 拼接图片路径
    df["image_path"] = df["image_id"].apply(lambda x: str(img_dir / x))

    # 检查图片是否存在
    exists_mask = df["image_path"].apply(lambda p: Path(p).exists())
    missing_count = int((~exists_mask).sum())
    if missing_count > 0:
        print(f"[警告] {split} 中有 {missing_count} 个标注框对应的图片路径不存在，将删除这些记录。")
        df = df[exists_mask].copy()

    if limit_images is not None:
        keep_images = df["image_id"].drop_duplicates().head(limit_images)
        df = df[df["image_id"].isin(keep_images)].copy()

    df = df[[
        "image_id", "split",
        "x1", "y1", "x2", "y2",
        "class_name", "image_path"
    ]]

    out_csv = out_dir / f"{split}_objects.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"[完成] {split}")
    print(f"输出文件：{out_csv}")
    print(f"图片数：{df['image_id'].nunique()}")
    print(f"标注框数：{len(df)}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--limit-images", type=int, default=None,
                        help="本地调试用，例如 100；全量运行时不填")
    args = parser.parse_args()

    cfg = load_config(args.config)

    for split in ["train", "val", "test"]:
        convert_split(split, cfg, limit_images=args.limit_images)


if __name__ == "__main__":
    main()