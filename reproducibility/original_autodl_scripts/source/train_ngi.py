from pathlib import Path
import argparse
import csv
import json
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.grid_dataset import GridObjectDataset
from src.models.ngi import NeuralGridInference


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()

    total = 0
    row_correct = 0
    col_correct = 0
    joint_correct = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, row, col in loader:
            x = x.to(device)
            row = row.to(device)
            col = col.to(device)

            row_logits, col_logits = model(x)
            loss = criterion(row_logits, row) + criterion(col_logits, col)

            row_pred = row_logits.argmax(dim=1)
            col_pred = col_logits.argmax(dim=1)

            total += x.size(0)
            row_correct += (row_pred == row).sum().item()
            col_correct += (col_pred == col).sum().item()
            joint_correct += ((row_pred == row) & (col_pred == col)).sum().item()
            total_loss += loss.item() * x.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "row_acc": row_correct / max(total, 1),
        "col_acc": col_correct / max(total, 1),
        "joint_acc": joint_correct / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--col-bins", type=int, default=64)
    parser.add_argument("--row-bins", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    train_path = processed_root / "grid_labels" / "train_grid.jsonl"
    val_path = processed_root / "grid_labels" / "val_grid.jsonl"

    train_set = GridObjectDataset(
        train_path,
        row_bins=args.row_bins,
        col_bins=args.col_bins
    )

    val_set = GridObjectDataset(
        val_path,
        row_bins=args.row_bins,
        col_bins=args.col_bins
    )

    num_rows = max(train_set.num_rows, val_set.num_rows)
    num_cols = max(train_set.num_cols, val_set.num_cols)

    print("=" * 80)
    print("Train objects:", len(train_set))
    print("Val objects:", len(val_set))
    print("num_rows:", num_rows)
    print("num_cols:", num_cols)
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = NeuralGridInference(
        num_rows=num_rows,
        num_cols=num_cols,
        num_freqs=16,
        hidden_dim=256,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    run_dir = output_root / "checkpoints" / f"ngi_seed_{args.seed}"
    log_dir = output_root / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"train_ngi_seed_{args.seed}.csv"
    best_path = run_dir / "best.pt"

    best_joint = -1.0

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_row_acc",
                "train_col_acc",
                "train_joint_acc",
                "val_loss",
                "val_row_acc",
                "val_col_acc",
                "val_joint_acc"
            ]
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()

            total = 0
            total_loss = 0.0
            row_correct = 0
            col_correct = 0
            joint_correct = 0

            for x, row, col in train_loader:
                x = x.to(device)
                row = row.to(device)
                col = col.to(device)

                optimizer.zero_grad()
                row_logits, col_logits = model(x)

                loss = criterion(row_logits, row) + criterion(col_logits, col)
                loss.backward()
                optimizer.step()

                row_pred = row_logits.argmax(dim=1)
                col_pred = col_logits.argmax(dim=1)

                total += x.size(0)
                total_loss += loss.item() * x.size(0)
                row_correct += (row_pred == row).sum().item()
                col_correct += (col_pred == col).sum().item()
                joint_correct += ((row_pred == row) & (col_pred == col)).sum().item()

            train_metrics = {
                "loss": total_loss / max(total, 1),
                "row_acc": row_correct / max(total, 1),
                "col_acc": col_correct / max(total, 1),
                "joint_acc": joint_correct / max(total, 1),
            }

            val_metrics = evaluate(model, val_loader, device)

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_row_acc": train_metrics["row_acc"],
                "train_col_acc": train_metrics["col_acc"],
                "train_joint_acc": train_metrics["joint_acc"],
                "val_loss": val_metrics["loss"],
                "val_row_acc": val_metrics["row_acc"],
                "val_col_acc": val_metrics["col_acc"],
                "val_joint_acc": val_metrics["joint_acc"],
            }

            writer.writerow(row)
            f.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"train loss {train_metrics['loss']:.4f}, "
                f"row {train_metrics['row_acc']:.4f}, "
                f"col {train_metrics['col_acc']:.4f}, "
                f"joint {train_metrics['joint_acc']:.4f} | "
                f"val loss {val_metrics['loss']:.4f}, "
                f"row {val_metrics['row_acc']:.4f}, "
                f"col {val_metrics['col_acc']:.4f}, "
                f"joint {val_metrics['joint_acc']:.4f}"
)

            if val_metrics["joint_acc"] > best_joint:
                best_joint = val_metrics["joint_acc"]
                torch.save({
                    "model_state": model.state_dict(),
                    "num_rows": num_rows,
                    "num_cols": num_cols,
                    "seed": args.seed,
                    "best_val_joint_acc": best_joint,
                }, best_path)

            if args.debug and epoch >= 5:
                break

    summary = {
        "seed": args.seed,
        "best_val_joint_acc": best_joint,
        "best_checkpoint": str(best_path),
        "log_path": str(log_path),
        "row_bins": args.row_bins,
        "col_bins": args.col_bins,
        "weight_decay": args.weight_decay,
    }
    

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("=" * 80)
    print("[完成] NGI training")
    print("Best val joint acc:", best_joint)
    print("Checkpoint:", best_path)
    print("Log:", log_path)
    print("=" * 80)


if __name__ == "__main__":
    main()