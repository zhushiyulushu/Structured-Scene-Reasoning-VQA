from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper_results"

required = [
    "table2_sku110k_main.csv",
    "table3_egocart_oos.csv",
    "table4_structured_vqa.csv",
    "table5_ablation.csv",
]

for name in required:
    p = RESULTS / name
    if not p.exists():
        raise FileNotFoundError(p)

def rows(name):
    with open(RESULTS / name, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

t2 = rows("table2_sku110k_main.csv")
t3 = rows("table3_egocart_oos.csv")
t4 = rows("table4_structured_vqa.csv")
t5 = rows("table5_ablation.csv")

ssr2 = next(r for r in t2 if r["Method"] == "SSR")
ssr3 = next(r for r in t3 if r["Method"] == "SSR")
ssr4 = next(r for r in t4 if r["Method"] == "SSR")

assert "95.41" in ssr2["F1_pct"]
assert ssr3["AUPRC_pct"] == "81.20"
assert ssr4["Locate_EM"] == "92.42"
assert ssr4["Overall_EM"] == "94.65"

topology = next(
    r for r in t5
    if r["Configuration"] == "w/o_Topology_Context"
)
assert topology["Delta_F1_pp"] == "-19.29"

print("PAPER_RESULT_CHECK_PASS")
print("SKU-110K F1:", ssr2["F1_pct"])
print("EgoCart AUPRC:", ssr3["AUPRC_pct"])
print("Locate EM:", ssr4["Locate_EM"])
print("Overall VQA EM:", ssr4["Overall_EM"])
print("w/o Topology Context:", topology["Delta_F1_pp"])
