import argparse,torch
from PIL import Image
from transformers import BlipProcessor,BlipForQuestionAnswering
from common_eval import *
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--model",required=True);ap.add_argument("--limit",type=int,default=0);a=ap.parse_args()
 items=read_jsonl(INPUT);items=items[:a.limit] if a.limit else items
 out=OUTROOT/"blip"/"predictions.jsonl";done=load_done(out)
 proc=BlipProcessor.from_pretrained(a.model,local_files_only=True);model=BlipForQuestionAnswering.from_pretrained(a.model,local_files_only=True).to("cuda").eval()
 for i,r in enumerate(items,1):
  if key(r) in done: continue
  im=Image.open(r["image_path"]).convert("RGB");inp=proc(images=im,text=prompt_of(r),return_tensors="pt").to("cuda")
  with torch.inference_mode(): ids=model.generate(**inp,max_new_tokens=8,num_beams=1,do_sample=False)
  raw=proc.decode(ids[0],skip_special_tokens=True).strip();pred=canonicalize(raw,allowed_of(r));append_jsonl(out,{**r,"raw_text":raw,"pred":pred})
  if i%100==0: print("BLIP",i,"/",len(items),flush=True)
 print(score_file("BLIP",out),flush=True)
if __name__=="__main__":main()
