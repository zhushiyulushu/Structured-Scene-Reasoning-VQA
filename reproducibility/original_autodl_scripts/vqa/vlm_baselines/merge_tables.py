from pathlib import Path
import pandas as pd
from common_eval import OUTROOT
def main():
 frames=[]
 for p in [OUTROOT/"blip"/"BLIP_summary.csv",OUTROOT/"qwen25_vl_7b"/"Qwen2.5-VL-7B_summary.csv",OUTROOT/"qwen3_vl_8b"/"Qwen3-VL-8B_summary.csv"]:
  if p.exists():frames.append(pd.read_csv(p))
 ssr=Path("/root/autodl-tmp/gate2_vqa_region_final/GATE2_VQA_TEST_SSR_SUMMARY.csv")
 if ssr.exists():
  s=pd.read_csv(ssr).rename(columns={"type":"Type","n":"N"});s.insert(0,"Method","Ours (SSR)");s["InvalidRate"]=0.0;frames.append(s[["Method","Type","N","EM","TokenF1","BLEU1","InvalidRate"]])
 if not frames:raise RuntimeError("No summaries")
 a=pd.concat(frames,ignore_index=True);a.to_csv(OUTROOT/"FINAL_ALL_METHODS_BY_TYPE.csv",index=False)
 ov=a[a.Type=="Overall"].copy();ov.to_csv(OUTROOT/"FINAL_TABLE4_OVERALL.csv",index=False)
 em=a.pivot(index="Method",columns="Type",values="EM").reset_index();cols=["Method"]+[c for c in ["Count","Verify","Locate","Identify","Overall"] if c in em.columns];em[cols].to_csv(OUTROOT/"FINAL_TABLE4_BY_TYPE_EM.csv",index=False)
 print(ov.to_string(index=False));print(em[cols].to_string(index=False))
if __name__=="__main__":main()
