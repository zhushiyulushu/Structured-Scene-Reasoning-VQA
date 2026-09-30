import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SRC = ROOT / "SSR_GATE0_CURVES_EXPORT"
SEEDS = [42, 2024, 3407]
SEED_FILES = {
    42: SRC / "seed42_detector_epoch_metrics.csv",
    2024: SRC / "seed2024_detector_epoch_metrics.csv",
    3407: SRC / "seed3407_detector_epoch_metrics.csv",
}

# ---------------------------------------------------------------------------
# Figure style knobs
# ---------------------------------------------------------------------------
# Change these values if you want to adjust the camera-ready appearance later.
# The script reads the same CSV data regardless of styling changes.
FONT_FAMILY = "Book Antiqua"
BASE_FONT_SIZE = 10
TITLE_FONT_SIZE = 10
AXIS_LABEL_SIZE = 10
TICK_LABEL_SIZE = 10
LEGEND_FONT_SIZE = 10

# Color and line controls. Use hex colors or Matplotlib color names here.
MEAN_COLOR = "#1F2937"
STD_BAND_COLOR = "#8DA0CB"
GRID_COLOR = "#E5E7EB"
AXIS_COLOR = "#6B7280"
TICK_COLOR = "#374151"
MEAN_LINE_WIDTH = 1.5
STD_BAND_ALPHA = 0.14

# Canvas and output controls.
FIG_SIZE = (7.1, 3.05)
PREVIEW_DPI = 180
OUTPUT_DPI = 600


def to_float(value):
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_seed(seed, path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epoch_raw = int(float(row["epoch"]))
            rows.append(
                {
                    "seed": seed,
                    "epoch_raw": epoch_raw,
                    "epoch": epoch_raw + 1,
                    "train_total_loss": to_float(row["train_total_loss"]),
                    "mAP50": to_float(row["mAP50"]),
                    "precision": to_float(row["precision"]),
                    "recall": to_float(row["recall"]),
                    "f1": to_float(row["f1"]),
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
    return rows


def mean(values):
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else math.nan


def sample_std(values):
    valid = [v for v in values if not math.isnan(v)]
    if len(valid) <= 1:
        return 0.0
    m = mean(valid)
    return math.sqrt(sum((v - m) ** 2 for v in valid) / (len(valid) - 1))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    seed_rows = []
    for seed in SEEDS:
        seed_rows.extend(read_seed(seed, SEED_FILES[seed]))

    stats = []
    epochs = sorted({r["epoch"] for r in seed_rows})
    for epoch in epochs:
        rows = [r for r in seed_rows if r["epoch"] == epoch]
        losses = [r["train_total_loss"] for r in rows]
        maps = [r["mAP50"] for r in rows]
        stats.append(
            {
                "epoch": epoch,
                "n_seeds": len(rows),
                "seeds": ";".join(str(s) for s in SEEDS),
                "train_total_loss_mean": mean(losses),
                "train_total_loss_std": sample_std(losses),
                "mAP50_mean": mean(maps),
                "mAP50_std": sample_std(maps),
            }
        )

    write_csv(
        OUT / "fig3_detector_three_seed_epoch.csv",
        seed_rows,
        ["seed", "epoch_raw", "epoch", "train_total_loss", "mAP50", "precision", "recall", "f1", "source_file"],
    )
    write_csv(
        OUT / "fig3_detector_three_seed_mean_std.csv",
        stats,
        ["epoch", "n_seeds", "seeds", "train_total_loss_mean", "train_total_loss_std", "mAP50_mean", "mAP50_std"],
    )

    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=PREVIEW_DPI)
    fig.patch.set_facecolor("white")

    x = [r["epoch"] for r in stats]
    loss_mean = [r["train_total_loss_mean"] for r in stats]
    loss_std = [r["train_total_loss_std"] for r in stats]
    map_mean = [r["mAP50_mean"] for r in stats]
    map_std = [r["mAP50_std"] for r in stats]

    panels = [
        (axes[0], "Loss", loss_mean, loss_std, "(a) Training Total Loss"),
        (axes[1], "mAP@0.5", map_mean, map_std, "(b) Validation mAP@0.5"),
    ]

    for ax, ylabel, avg, sd, panel_label in panels:
        lower = [m - s for m, s in zip(avg, sd)]
        upper = [m + s for m, s in zip(avg, sd)]
        ax.fill_between(x, lower, upper, color=STD_BAND_COLOR, alpha=STD_BAND_ALPHA, linewidth=0, label="±1 std")
        ax.plot(x, avg, color=MEAN_COLOR, linewidth=MEAN_LINE_WIDTH, label="mean")

        ax.set_title(panel_label, loc="left", fontweight="normal", pad=6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.grid(True, color=GRID_COLOR, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(AXIS_COLOR)
        ax.spines["bottom"].set_color(AXIS_COLOR)
        ax.tick_params(colors=TICK_COLOR, width=0.8)

    axes[0].set_xlim(1, 20)
    axes[1].set_xlim(1, 20)
    axes[0].set_ylim(4.35, 7.75)
    axes[1].set_ylim(0.67, 0.93)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    handles, labels = axes[1].get_legend_handles_labels()
    # Legend order: mean first, then the shaded standard-deviation band.
    fig.legend(
        [handles[1], handles[0]],
        [labels[1], labels[0]],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
        handlelength=2.4,
        columnspacing=1.2,
    )

    # No overall title: panel titles carry the figure labels.
    fig.tight_layout(rect=(0.012, 0.055, 0.995, 0.995), w_pad=1.45)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"figure3_detector_training_dynamics_final_v2.{ext}", bbox_inches="tight", pad_inches=0.025, dpi=OUTPUT_DPI)

    notes = OUT / "figure3_detector_data_notes.md"
    notes.write_text(
        "# Figure 3 Detector Data Notes\n\n"
        "This figure uses the three detector seed logs from `SSR_GATE0_CURVES_EXPORT`.\n\n"
        "- Source seed files: `seed42_detector_epoch_metrics.csv`, `seed2024_detector_epoch_metrics.csv`, "
        "`seed3407_detector_epoch_metrics.csv`.\n"
        "- Epochs are displayed as 1-20; the original YOLO-style logs store them as 0-19.\n"
        "- Left panel: `train_total_loss`. Right panel: validation `mAP50`.\n"
        "- The figure shows only the mean curve and a light +/-1 sample-std band.\n"
        "- Styling parameters, including font family, font sizes, colors, linewidth, and band alpha, are grouped near the top of `plot_fig3_detector_final.py`.\n",
        encoding="utf-8",
    )

    print(OUT / "figure3_detector_training_dynamics_final.png")
    print(OUT / "figure3_detector_training_dynamics_final.pdf")
    print(OUT / "figure3_detector_training_dynamics_final.svg")


if __name__ == "__main__":
    main()