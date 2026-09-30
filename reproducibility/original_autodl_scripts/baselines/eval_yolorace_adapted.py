from pathlib import Path
import argparse
import pandas as pd
from ultralytics import YOLO

from src.eval.tune_yolo_vacancy_conf import (
    load_config,
    read_jsonl,
    evaluate,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iou-thr", type=float, default=0.30)
    parser.add_argument("--vacancy-class", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    yolo_root = processed_root / "yolo_retaildet_style"
    val_rows = read_jsonl(yolo_root / "val_manifest.jsonl")
    test_rows = read_jsonl(yolo_root / "test_manifest.jsonl")

    print("Loading:", args.weights)
    model = YOLO(args.weights)

    confs = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    val_results = []
    for c in confs:
        r = evaluate(
            model,
            val_rows,
            conf=c,
            vacancy_class=args.vacancy_class,
            iou_thr=args.iou_thr,
        )
        val_results.append(r)

    val_df = pd.DataFrame(val_results)
    best_row = val_df.sort_values("F1", ascending=False).iloc[0]
    best_conf = float(best_row["conf"])

    print("BEST VAL CONF =", best_conf)

    test_result = evaluate(
        model,
        test_rows,
        conf=best_conf,
        vacancy_class=args.vacancy_class,
        iou_thr=args.iou_thr,
    )

    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    val_path = table_dir / f"yolorace_adapted_seed{args.seed}_conf_tuning_val.csv"
    test_path = table_dir / f"yolorace_adapted_seed{args.seed}_test_val_tuned.csv"

    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([test_result]).to_csv(
        test_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 80)
    print("YOLO-RACE (Adapted)")
    print("Seed:", args.seed)
    print("Best val conf:", best_conf)
    print("Test:")
    print(pd.DataFrame([test_result]))
    print("Saved:", val_path)
    print("Saved:", test_path)
    print("=" * 80)

if __name__ == "__main__":
    main()
