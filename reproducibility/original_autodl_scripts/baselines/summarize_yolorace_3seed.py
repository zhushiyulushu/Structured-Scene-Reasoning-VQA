from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(
    "/root/autodl-tmp/ssr_yolorace_exp/"
    "code/SSR_release_clean/SSR-VQA/outputs/tables"
)

seeds = [42, 2024, 3407]
rows = []

for seed in seeds:
    f = ROOT / f"yolorace_adapted_seed{seed}_test_val_tuned.csv"

    if not f.exists():
        raise FileNotFoundError(f)

    df = pd.read_csv(f)
    r = df.iloc[0]

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

summary_rows = []

for metric in ["Precision", "Recall", "F1"]:
    values = raw[metric].to_numpy(dtype=float)

    summary_rows.append({
        "Metric": metric,
        "Mean": np.mean(values),
        "Std": np.std(values, ddof=0),
        "Mean_percent": np.mean(values) * 100,
        "Std_percent": np.std(values, ddof=0) * 100,
    })

summary = pd.DataFrame(summary_rows)

raw_path = ROOT / "yolorace_adapted_3seed_raw.csv"
summary_path = ROOT / "yolorace_adapted_3seed_summary.csv"

raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

print()
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
print("Saved:", raw_path)
print("Saved:", summary_path)
