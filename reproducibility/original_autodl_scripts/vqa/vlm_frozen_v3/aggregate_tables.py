#!/usr/bin/env python3
import csv
from pathlib import Path
OUT=Path("/root/autodl-tmp/gate2_vlm_baselines_final")
SSR=Path("/root/autodl-tmp/gate2_vqa_region_final/GATE2_VQA_TEST_SSR_SUMMARY.csv")
models=[("BLIP","blip"),("Qwen2-VL-2B-Instruct","qwen2"),("LLaVA-1.5-7B-HF","llava")]

def readcsv(p):
    with open(p,encoding="utf-8-sig") as f:return list(csv.DictReader(f))

if not SSR.exists():raise FileNotFoundError(SSR)
allm={}
for display,key in models:
    p=OUT/f"{key}_summary.csv"
    if not p.exists():raise FileNotFoundError(p)
    allm[display]=readcsv(p)
allm["Ours (SSR)"]=readcsv(SSR)

main=[]
for method,rows in allm.items():
    d={r["type"]:r for r in rows};o=d["Overall"]
    main.append({"Method":method,"Answer_EM":float(o["EM"]),"Token_F1":float(o["TokenF1"]),"BLEU_1":float(o["BLEU1"])})
with open(OUT/"TABLE4_MAIN.csv","w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=main[0].keys());w.writeheader();w.writerows(main)

per=[]
for method,rows in allm.items():
    d={r["type"]:r for r in rows}
    per.append({"Method":method,
                "Count_EM":float(d["Count"]["EM"]),
                "Verify_EM":float(d["Verify"]["EM"]),
                "Locate_EM":float(d["Locate"]["EM"]),
                "Identify_EM":float(d["Identify"]["EM"]),
                "Overall_EM":float(d["Overall"]["EM"])})
with open(OUT/"TABLE4_PER_TYPE_EM.csv","w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=per[0].keys());w.writeheader();w.writerows(per)

print("=== TABLE4 MAIN ===")
for r in main:print(r)
print("=== TABLE4 PER-TYPE EM ===")
for r in per:print(r)
