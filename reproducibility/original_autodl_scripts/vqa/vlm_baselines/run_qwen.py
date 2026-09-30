import argparse,torch
from PIL import Image
from transformers import AutoProcessor
try:
 from transformers import AutoModelForImageTextToText as AutoVisionModel
except Exception:
 from transformers import AutoModelForVision2Seq as AutoVisionModel
from common_eval import *
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--model",required=True);ap.add_argument("--name",required=True);ap.add_argument("--limit",type=int,default=0);a=ap.parse_args()
 items=read_jsonl(INPUT);items=items[:a.limit] if a.limit else items
 safe=a.name.lower().replace(".","").replace("-","_");out=OUTROOT/safe/"predictions.jsonl";done=load_done(out)
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True,trust_remote_code=True);model=AutoVisionModel.from_pretrained(a.model,local_files_only=True,trust_remote_code=True,torch_dtype="auto",device_map="auto").eval()
 for i,r in enumerate(items,1):
  if key(r) in done: continue
  im=Image.open(r["image_path"]).convert("RGB");prompt=prompt_of(r);messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
  text=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True) if hasattr(proc,"apply_chat_template") else prompt
  inputs=proc(text=[text],images=[im],padding=True,return_tensors="pt");dev=next(model.parameters()).device;inputs={k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inputs.items()}
  with torch.inference_mode(): gen=model.generate(**inputs,max_new_tokens=12,do_sample=False,num_beams=1)
  if "input_ids" in inputs and gen.shape[1]>inputs["input_ids"].shape[1]: gen=gen[:,inputs["input_ids"].shape[1]:]
  raw=proc.batch_decode(gen,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip();pred=canonicalize(raw,allowed_of(r));append_jsonl(out,{**r,"raw_text":raw,"pred":pred})
  if i%50==0: print(a.name,i,"/",len(items),flush=True)
 print(score_file(a.name,out),flush=True)
if __name__=="__main__":main()
