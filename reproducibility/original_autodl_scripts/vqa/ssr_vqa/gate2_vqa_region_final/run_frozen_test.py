from pathlib import Path
import json,csv
from common_vqa import *
OUT=Path('/root/autodl-tmp/gate2_vqa_region_final'); VP=OUT/'GATE2_VQA_VAL_PROTOCOL.json'
pro=json.load(open(VP))
if pro['gate']!='PASS_VQA_PROTOCOL_FREEZE_READY': raise RuntimeError('Validation protocol did not pass; Test blocked.')
sel=pro['selected_on_val']; pc=float(sel['product_conf']);vt=float(sel['vertical_overlap']);df=float(sel['distance_factor'])
MP=P/'data/processed/masked/test_masked_img.jsonl';CP=CACHE_DIR/'test_pred_cache.json';M={z['masked_id']:z for z in lines(MP)};C=load_cache(CP)
rows=[]; frozen=[]
for rec in C:
    mid=rec.get('masked_id');m=M.get(mid)
    if m is None:continue
    qs=build(mid,m,rec,pc,vt,df)
    if qs is None:continue
    for typ,qq,g,p,sp in qs:
        r={'masked_id':mid,'image_path':image_path(m),'type':typ,'question':qq,'answer_space':' | '.join(sp),'gold':g,'ssr_pred':p,'EM':em(p,g),'TokenF1':tf1(p,g),'BLEU1':bleu1(p,g)}
        rows.append(r);frozen.append({k:r[k] for k in ['masked_id','image_path','type','question','answer_space','gold']})
sm=summarize(rows)
with open(OUT/'GATE2_VQA_TEST_PER_QUESTION.csv','w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
with open(OUT/'GATE2_VQA_TEST_SSR_SUMMARY.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=sm[0].keys());w.writeheader();w.writerows(sm)
with open(OUT/'GATE2_FROZEN_VLM_INPUTS.jsonl','w',encoding='utf-8') as f:
    for r in frozen:f.write(json.dumps(r,ensure_ascii=False)+'\n')
meta={'protocol':'Val-frozen balanced region-grounded VQA Test','n_samples':len(set(r['masked_id'] for r in rows)),'n_questions':len(rows),'questions_per_sample':4,'selected_on_val':sel,'ssr_summary':sm,'same_frozen_inputs_required_for_all_vlm_baselines':True}
json.dump(meta,open(OUT/'GATE2_VQA_FINAL_PROTOCOL.json','w'),indent=2)
print('GATE2_REGION_VQA_TEST_COMPLETE');print(json.dumps(meta,indent=2));print('FROZEN_INPUTS',OUT/'GATE2_FROZEN_VLM_INPUTS.jsonl')
