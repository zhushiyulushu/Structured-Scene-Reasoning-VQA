from pathlib import Path
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["output_root"])

    log_path = output_root / "logs" / f"learnable_grid_train_seed_{args.seed}.csv"
    df = pd.read_csv(log_path)

    out_dir = output_root / "figures" / "training_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("SmoothL1 loss")
    plt.title("Learnable Grid Completion Training Loss")
    plt.legend()
    plt.tight_layout()
    out1 = out_dir / f"learnable_grid_loss_seed_{args.seed}.png"
    plt.savefig(out1, dpi=300)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["val_f1"], label="Validation F1")
    plt.plot(df["epoch"], df["val_mean_iou"], label="Validation mean IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Learnable Grid Completion Validation Performance")
    plt.legend()
    plt.tight_layout()
    out2 = out_dir / f"learnable_grid_val_seed_{args.seed}.png"
    plt.savefig(out2, dpi=300)
    plt.close()

    print("Saved:", out1)
    print("Saved:", out2)


if __name__ == "__main__":
    main()