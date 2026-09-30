from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/root/autodl-tmp/cvpce_adapted/outputs")
SEEDS = [42, 2024, 3407]

rows = []

for seed in SEEDS:
    f = ROOT / f"seed{seed}" / "test_val_tuned.csv"
    if not f.exists():
        raise FileNotFoundError(f)

    r = pd.read_csv(f).iloc[0]

    rows.append({
        "Seed": seed,
        "Confidence": float(r["conf"]),
        "TP": int(r["TP"]),
        "FP": int(r["FP"]),
        "FN": int(r["FN"]),
        "Precision": float(r["Precision"]),
        "Recall": float(r["Recall"]),
        "F1": float(r["F1"]),
    })

raw = pd.DataFrame(rows)

summary = []

for metric in ["Precision", "Recall", "F1"]:
    x = raw[metric].to_numpy(dtype=float)
    summary.append({
        "Metric": metric,
        "Mean": x.mean(),
        "Std": x.std(ddof=0),
        "Mean_percent": x.mean() * 100,
        "Std_percent": x.std(ddof=0) * 100,
    })

summary = pd.DataFrame(summary)

raw.to_csv(
    ROOT / "CVPCE_GLN_Adapted_3seed_raw.csv",
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    ROOT / "CVPCE_GLN_Adapted_3seed_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("========== THREE SEEDS ==========")
print(raw.to_string(index=False))

print()
print("========== MEAN ± STD ==========")

for _, r in summary.iterrows():
    print(
        f'{r["Metric"]:10s}: '
        f'{r["Mean"]:.4f} ± {r["Std"]:.4f} '
        f'({r["Mean_percent"]:.2f} ± {r["Std_percent"]:.2f}%)'
    )

print()
print("CVPCE_3SEED_SUMMARY_PASS")
