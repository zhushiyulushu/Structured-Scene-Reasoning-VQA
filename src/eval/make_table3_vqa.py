from pathlib import Path
import argparse
import yaml
import pandas as pd
import json


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["output_root"])
    table_dir = output_root / "tables"

    blip_path = table_dir / "blip_baseline_summary_test_-1.csv"
    oracle_path = output_root / "results" / "oracle_dsi_test.json"

    blip = pd.read_csv(blip_path)
    oracle = read_json(oracle_path)

    def acc_for(t):
        r = blip[blip["type"] == t].iloc[0]
        return float(r["accuracy"])

    rows = []

    rows.append({
        "Method": "BLIP VQA",
        "Count Acc": f"{acc_for('count'):.4f}",
        "Locate Acc": f"{acc_for('locate'):.4f}",
        "Verify Acc": f"{acc_for('verify'):.4f}",
        "Identify Acc": f"{acc_for('identify'):.4f}",
    })

    rows.append({
        "Method": "Oracle Grid DSI + Symbolic Answer",
        "Count Acc": f"{oracle['vqa_count_acc']:.4f}",
        "Locate Acc": f"{oracle['vqa_locate_acc']:.4f}",
        "Verify Acc": f"{oracle['vqa_verify_acc']:.4f}",
        "Identify Acc": f"{oracle['vqa_identify_acc']:.4f}",
    })

    out = pd.DataFrame(rows)
    out_path = table_dir / "table3_structured_vqa.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(out)
    print("=" * 80)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()