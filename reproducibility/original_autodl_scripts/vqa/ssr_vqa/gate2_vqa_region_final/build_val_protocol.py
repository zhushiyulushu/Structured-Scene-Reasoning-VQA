from pathlib import Path
import json,csv
from common_vqa import *
OUT=Path('/root/autodl-tmp/gate2_vqa_region_final'); OUT.mkdir(parents=True,exist_ok=True)
MP=P/'data/processed/masked/val_masked_img.jsonl'; CP=CACHE_DIR/'val_pred_cache.json'
print('GATE2_VQA_REGION_VAL TEST_NOT_OPENED=True',flush=True)
M={z['masked_id']:z for z in lines(MP)}; C=load_cache(CP); choices=[]
for pc in [0.10,0.20,0.30,0.50]:
  for vt in [0.20,0.35,0.50]:
    for df in [1.5,2.0,3.0]:
      rows=[]
      for rec in C:
        mid=rec.get('masked_id'); m=M.get(mid)
        if m is None:continue
        qs=build(mid,m,rec,pc,vt,df)
        if qs is None:continue
        for typ,qq,g,p,sp in qs: rows.append({'masked_id':mid,'type':typ,'question':qq,'gold':g,'ssr_pred':p,'EM':em(p,g),'TokenF1':tf1(p,g),'BLEU1':bleu1(p,g)})
      sm=summarize(rows); D={r['type']:r for r in sm}; choices.append((D['Overall']['EM'],D['Identify']['EM'],pc,vt,df,rows,sm))
      print('VAL',pc,vt,df,'overall',round(D['Overall']['EM'],4),'locate',round(D['Locate']['EM'],4),'identify',round(D['Identify']['EM'],4),flush=True)
choices.sort(key=lambda x:(x[0],x[1]),reverse=True)
overall,ident,pc,vt,df,rows,sm=choices[0]; D={r['type']:r for r in sm}
gate='PASS_VQA_PROTOCOL_FREEZE_READY' if D['Overall']['EM']>=0.78 and D['Locate']['EM']>=0.80 else 'BLOCKED_VQA_PROTOCOL_WEAK'
pro={'gate':gate,'test_opened':False,'protocol':'balanced region-grounded structured VQA','gate0_top1_conf':TOP1_CONF,'selected_on_val':{'product_conf':pc,'vertical_overlap':vt,'distance_factor':df},'val_summary':sm,'reference_policy':'GT used only for reference construction/scoring; SSR uses frozen detector cache.'}
json.dump(pro,open(OUT/'GATE2_VQA_VAL_PROTOCOL.json','w'),indent=2)
with open(OUT/'GATE2_VQA_VAL_SUMMARY.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=sm[0].keys());w.writeheader();w.writerows(sm)
print('='*88);print('GATE',gate);print('SELECTED',pc,vt,df);[print(r) for r in sm];print('='*88)
