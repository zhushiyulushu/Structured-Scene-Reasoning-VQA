#!/usr/bin/env python3
import argparse, json, os, re, math, csv, gc
from pathlib import Path
from collections import Counter
from PIL import Image
import torch

FROZEN = Path("/root/autodl-tmp/gate2_vqa_region_final/GATE2_FROZEN_VLM_INPUTS.jsonl")
OUTROOT = Path("/root/autodl-tmp/gate2_vlm_baselines_final")
PROJECT = Path("/root/autodl-tmp/ssr_yolorace_exp/code/SSR_release_clean/SSR-VQA")

MODEL_CANDIDATES = {
    "blip": [
        PROJECT/"models/blip-vqa-base",
        Path("/root/autodl-tmp/models/blip-vqa-base"),
    ],
    "qwen2": [
        PROJECT/"Qwen2-VL-2B-Instruct/extracted/Qwen2-VL-2B-Instruct",
        Path("/root/autodl-tmp/Qwen2-VL-2B-Instruct/extracted/Qwen2-VL-2B-Instruct"),
        Path("/root/autodl-tmp/models/Qwen2-VL-2B-Instruct"),
    ],
    "llava": [
        PROJECT/"llava-1.5-7b-hf",
        Path("/root/autodl-tmp/llava-1.5-7b-hf"),
        Path("/root/autodl-tmp/models/llava-1.5-7b-hf"),
    ],
}
HF_IDS = {
    "blip":"Salesforce/blip-vqa-base",
    "qwen2":"Qwen/Qwen2-VL-2B-Instruct",
    "llava":"llava-hf/llava-1.5-7b-hf",
}

def read_jsonl(path):
    out=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out

def norm_text(s):
    s=str(s).lower().replace("-"," ").replace("_"," ")
    s=re.sub(r"[^a-z0-9\s]"," ",s)
    return " ".join(s.split())

def parse_allowed(s):
    return [x.strip() for x in str(s).split("|") if x.strip()]

def normalize_prediction(raw, allowed):
    """Shared deterministic closed-set parser. Never uses gold."""
    t=norm_text(raw)
    amap={norm_text(a):a for a in allowed}
    if t in amap: return amap[t]
    if set(amap)=={"zero","one"}:
        aliases={"0":"zero","zero":"zero","none":"zero","1":"one","one":"one","single":"one"}
        for token in re.findall(r"\b(?:0|1|zero|one|none|single)\b",t):
            a=aliases[token]
            if a in amap: return amap[a]
    hits=[]
    for na,a in sorted(amap.items(),key=lambda kv:len(kv[0]),reverse=True):
        if re.search(r"(?<![a-z0-9])"+re.escape(na)+r"(?![a-z0-9])",t):
            if a not in hits: hits.append(a)
    return hits[0] if len(hits)==1 else "__invalid__"

def tokens(s): return re.findall(r"[a-z0-9]+",str(s).lower())
def em(a,b): return int(tokens(a)==tokens(b))
def tf1(a,b):
    A,B=tokens(a),tokens(b)
    if not A and not B:return 1.0
    ca,cb=Counter(A),Counter(B);m=sum((ca&cb).values())
    p=m/max(len(A),1);r=m/max(len(B),1)
    return 2*p*r/max(p+r,1e-12)
def bleu1(a,b):
    A,B=tokens(a),tokens(b)
    if not A:return 0.0
    ca,cb=Counter(A),Counter(B);m=sum((ca&cb).values())
    prec=m/len(A);bp=1.0 if len(A)>=len(B) else math.exp(1-len(B)/max(len(A),1))
    return bp*prec

def prompt_for(r):
    allowed=parse_allowed(r["answer_space"])
    return ("Answer this controlled shelf visual question. "
            "Choose exactly one answer from the allowed answers and return only that answer, with no explanation.\n"
            f"Question: {r['question']}\n"
            f"Allowed answers: {' | '.join(allowed)}\nAnswer:")

def find_model(name):
    override=os.environ.get(name.upper()+"_MODEL","").strip()
    if override:
        p=Path(override)
        if not p.exists(): raise FileNotFoundError(p)
        return str(p)
    for p in MODEL_CANDIDATES[name]:
        if (p/"config.json").exists(): return str(p)
    return HF_IDS[name]

def local_only(path):
    return Path(path).exists()

def load_done(path):
    done={}
    if path.exists():
        with open(path,encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r=json.loads(line);done[(r["masked_id"],r["type"])]=r
    return done

def to_device(inputs,device):
    return {k:(v.to(device) if hasattr(v,"to") else v) for k,v in inputs.items()}

def load_model(name,path):
    from transformers import AutoProcessor
    device="cuda" if torch.cuda.is_available() else "cpu"
    dtype=torch.float16 if device=="cuda" else torch.float32
    kw=dict(torch_dtype=dtype,local_files_only=local_only(path))
    if name=="blip":
        from transformers import BlipProcessor,BlipForQuestionAnswering
        proc=BlipProcessor.from_pretrained(path,local_files_only=local_only(path))
        model=BlipForQuestionAnswering.from_pretrained(path,**kw).to(device).eval()
    elif name=="qwen2":
        try:
            from transformers import Qwen2VLForConditionalGeneration as C
        except Exception:
            from transformers import AutoModelForVision2Seq as C
        # Qwen2-VL uses dynamic image resolution. Bound the visual-token budget
        # for high-resolution SKU-110K shelf images on a 24-GB GPU.
        proc=AutoProcessor.from_pretrained(
            path,
            local_files_only=local_only(path),
            min_pixels=128*28*28,
            max_pixels=512*28*28,
        )
        model=C.from_pretrained(path,**kw).to(device).eval()
        try:
            model.generation_config.temperature=None
            model.generation_config.top_p=None
            model.generation_config.top_k=None
        except Exception:
            pass
    elif name=="llava":
        try:
            from transformers import LlavaForConditionalGeneration as C
        except Exception:
            from transformers import AutoModelForVision2Seq as C
        proc=AutoProcessor.from_pretrained(path,local_files_only=local_only(path))
        model=C.from_pretrained(path,**kw).to(device).eval()
    else: raise ValueError(name)
    return proc,model,device

@torch.inference_mode()
def infer(name,proc,model,device,rows):
    images=[Image.open(r["image_path"]).convert("RGB") for r in rows]
    prompts=[prompt_for(r) for r in rows]
    try:
        if name=="blip":
            inputs=proc(images=images,text=prompts,padding=True,return_tensors="pt")
            inputs=to_device(inputs,device)
            ids=model.generate(**inputs,do_sample=False,max_new_tokens=12)
            texts=proc.batch_decode(ids,skip_special_tokens=True)
        elif name=="qwen2":
            chats=[]
            for p in prompts:
                msgs=[{"role":"user","content":[{"type":"image"},{"type":"text","text":p}]}]
                chats.append(proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True))
            inputs=proc(text=chats,images=images,padding=True,return_tensors="pt")
            inputs=to_device(inputs,device)
            ids=model.generate(**inputs,do_sample=False,max_new_tokens=16,use_cache=True)
            trimmed=[o[len(i):] for i,o in zip(inputs["input_ids"],ids)]
            texts=proc.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        else:
            chats=[]
            for p in prompts:
                try:
                    msgs=[{"role":"user","content":[{"type":"image"},{"type":"text","text":p}]}]
                    txt=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
                except Exception:
                    txt="USER: <image>\n"+p+"\nASSISTANT:"
                chats.append(txt)
            inputs=proc(images=images,text=chats,padding=True,return_tensors="pt")
            inputs=to_device(inputs,device)
            ids=model.generate(**inputs,do_sample=False,max_new_tokens=16,use_cache=True)
            trimmed=[o[len(i):] for i,o in zip(inputs["input_ids"],ids)]
            texts=proc.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        return texts
    finally:
        for im in images: im.close()

def summarize(records):
    out=[]
    for typ in ["Count","Verify","Locate","Identify","Overall"]:
        z=records if typ=="Overall" else [r for r in records if r["type"]==typ]
        out.append({"type":typ,"n":len(z),
                    "EM":sum(float(r["EM"]) for r in z)/max(len(z),1),
                    "TokenF1":sum(float(r["TokenF1"]) for r in z)/max(len(z),1),
                    "BLEU1":sum(float(r["BLEU1"]) for r in z)/max(len(z),1),
                    "InvalidRate":sum(r["normalized_pred"]=="__invalid__" for r in z)/max(len(z),1)})
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",choices=["blip","qwen2","llava"],required=True)
    ap.add_argument("--batch-size",type=int,default=None)
    ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args()
    data=read_jsonl(FROZEN)
    if a.limit>0:data=data[:a.limit]
    elif len(data)!=11448:raise RuntimeError(f"Expected 11448 frozen questions, got {len(data)}")
    defaults={"blip":16,"qwen2":2,"llava":1};bs=a.batch_size or defaults[a.model]
    OUTROOT.mkdir(parents=True,exist_ok=True)
    predfile=OUTROOT/f"{a.model}_predictions.jsonl"
    done=load_done(predfile)
    remaining=[r for r in data if (r["masked_id"],r["type"]) not in done]
    print(f"MODEL={a.model} TOTAL={len(data)} DONE={len(done)} REMAINING={len(remaining)} BATCH={bs}",flush=True)
    if remaining:
        path=find_model(a.model);print("MODEL_PATH",path,flush=True)
        proc,model,device=load_model(a.model,path);print("DEVICE",device,flush=True)
        with open(predfile,"a",encoding="utf-8") as f:
            for s in range(0,len(remaining),bs):
                batch=remaining[s:s+bs]
                try: raw=infer(a.model,proc,model,device,batch)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    raise RuntimeError("CUDA OOM. Re-run this model with --batch-size 1; completed predictions will resume.")
                for r,txt in zip(batch,raw):
                    pred=normalize_prediction(txt,parse_allowed(r["answer_space"]))
                    rec={"masked_id":r["masked_id"],"image_path":r["image_path"],"type":r["type"],
                         "question":r["question"],"answer_space":r["answer_space"],"gold":r["gold"],
                         "raw_output":txt.strip(),"normalized_pred":pred,
                         "EM":em(pred,r["gold"]),"TokenF1":tf1(pred,r["gold"]),"BLEU1":bleu1(pred,r["gold"])}
                    f.write(json.dumps(rec,ensure_ascii=False)+"\n")
                f.flush()
                n=len(done)+min(s+len(batch),len(remaining))
                if n%100<=bs:print(f"PROGRESS {n}/{len(data)}",flush=True)
        del model,proc;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    records=list(load_done(predfile).values())
    if len(records)!=len(data):raise RuntimeError(f"Incomplete predictions {len(records)}/{len(data)}")
    sm=summarize(records)
    with open(OUTROOT/f"{a.model}_summary.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=sm[0].keys());w.writeheader();w.writerows(sm)
    with open(OUTROOT/f"{a.model}_protocol.json","w",encoding="utf-8") as f:
        json.dump({"model":a.model,"n_questions":len(records),"decoding":"greedy",
                   "shared_prompt":True,"shared_parser":True,"summary":sm},f,indent=2)
    print("SUMMARY",a.model)
    for r in sm:print(r)

if __name__=="__main__":main()
