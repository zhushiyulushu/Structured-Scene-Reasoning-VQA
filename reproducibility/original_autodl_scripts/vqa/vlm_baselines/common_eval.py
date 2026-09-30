import json,re,math,csv,collections
from pathlib import Path
INPUT=Path("/root/autodl-tmp/gate2_vqa_region_final/GATE2_FROZEN_VLM_INPUTS.jsonl")
OUTROOT=Path("/root/autodl-tmp/gate2_vqa_region_final/vlm_baselines_final")
def read_jsonl(p):
 rows=[]
 with open(p,encoding="utf-8") as f:
  for line in f:
   line=line.strip()
   if line: rows.append(json.loads(line))
 return rows
def append_jsonl(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 with open(p,"a",encoding="utf-8") as f: f.write(json.dumps(obj,ensure_ascii=False)+"\n"); f.flush()
def norm(s):
 s=str(s).lower().strip().replace("_"," ").replace("-"," ")
 s=re.sub(r"[^a-z0-9 ]+"," ",s); s=re.sub(r"\s+"," ",s).strip()
 return {"0":"zero","1":"one","none":"neither"}.get(s,s)
def allowed_of(r): return [x.strip() for x in str(r["answer_space"]).split("|")]
def canonicalize(raw,allowed):
 nr=norm(raw); opts=[norm(x) for x in allowed]
 for a,o in zip(allowed,opts):
  if nr==o:return a
 for i in sorted(range(len(opts)), key=lambda i:len(opts[i]), reverse=True):
  if re.search(r"(?<![a-z0-9])"+re.escape(opts[i])+r"(?![a-z0-9])",nr): return allowed[i]
 if "zero" in opts and re.search(r"(?<!\d)0(?!\d)",str(raw)): return allowed[opts.index("zero")]
 if "one" in opts and re.search(r"(?<!\d)1(?!\d)",str(raw)): return allowed[opts.index("one")]
 return "__invalid__"
def toks(s): return re.findall(r"[a-z0-9]+",str(s).lower())
def em(a,b): return int(norm(a)==norm(b))
def tf1(a,b):
 A,B=toks(a),toks(b)
 if not A and not B:return 1.0
 ca,cb=collections.Counter(A),collections.Counter(B);m=sum((ca&cb).values());p=m/max(len(A),1);r=m/max(len(B),1)
 return 2*p*r/max(p+r,1e-12)
def bleu1(a,b):
 A,B=toks(a),toks(b)
 if not A:return 0.0
 ca,cb=collections.Counter(A),collections.Counter(B);m=sum((ca&cb).values());prec=m/len(A);bp=1.0 if len(A)>=len(B) else math.exp(1-len(B)/max(len(A),1))
 return bp*prec
def key(r): return str(r["masked_id"])+"|"+str(r["type"])
def load_done(p):
 return {key(r):r for r in read_jsonl(p)} if p.exists() else {}
def prompt_of(r):
 return "Answer the shelf-image question using exactly one option from the allowed answer space. Do not explain. Return only the answer option.\nQuestion: "+r["question"]+"\nAllowed answers: "+r["answer_space"]+"\nAnswer:"
def score_file(name,p):
 rows=read_jsonl(p);sc=[]
 for r in rows:
  z=dict(r);z["EM"]=em(z["pred"],z["gold"]);z["TokenF1"]=tf1(z["pred"],z["gold"]);z["BLEU1"]=bleu1(z["pred"],z["gold"]);sc.append(z)
 if sc:
  fields=["masked_id","type","image_path","question","answer_space","gold","raw_text","pred","EM","TokenF1","BLEU1"]
  with open(p.with_suffix(".csv"),"w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(sc)
 sm=[]
 for typ in ["Count","Verify","Locate","Identify","Overall"]:
  z=sc if typ=="Overall" else [r for r in sc if r["type"]==typ]
  sm.append({"Method":name,"Type":typ,"N":len(z),"EM":sum(r["EM"] for r in z)/max(len(z),1),"TokenF1":sum(r["TokenF1"] for r in z)/max(len(z),1),"BLEU1":sum(r["BLEU1"] for r in z)/max(len(z),1),"InvalidRate":sum(r["pred"]=="__invalid__" for r in z)/max(len(z),1)})
 sp=p.parent/(name+"_summary.csv")
 with open(sp,"w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=sm[0].keys());w.writeheader();w.writerows(sm)
 return sm
