#!/usr/bin/env python
import argparse, json
import _bootstrap  # noqa: F401
from pathlib import Path
import torch
from transformers import AutoTokenizer
from extract.inspect_model import load_model
from extract.capture_activations import capture, DEFAULT_PROMPTS

def texts(path):
 if not path:return list(DEFAULT_PROMPTS)
 p=Path(path)
 if p.suffix==".jsonl": return [json.loads(x).get("text",json.loads(x).get("prompt")) for x in p.read_text().splitlines() if x.strip()]
 return [x for x in p.read_text().splitlines() if x.strip()]
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--model",default="Qwen/Qwen3.8-27B"); p.add_argument("--layer",type=int,default=0); p.add_argument("--dtype",default="bfloat16"); p.add_argument("--device-map",default="auto"); p.add_argument("--device"); p.add_argument("--offload-folder"); p.add_argument("--revision"); p.add_argument("--input"); p.add_argument("--max-tokens",type=int,default=2000); p.add_argument("--max-samples",type=int); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--shard-size",type=int,default=512); p.add_argument("--output-dir",type=Path,default=Path("results/layer0")); p.add_argument("--seed",type=int,default=42); p.add_argument("--include-special",action="store_true"); p.add_argument("--storage-dtype",choices=["float16","bfloat16"],default="float16")
 a,_=p.parse_known_args(argv); limit=min(x for x in (a.max_tokens,a.max_samples) if x is not None); model=load_model(a.model,a.dtype,a.device_map,a.offload_folder,a.revision); tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision); tok.pad_token=tok.pad_token or tok.eos_token
 print(json.dumps(capture(model,tok,a.layer,texts(a.input),a.output_dir,limit,a.batch_size,a.shard_size,a.seed,getattr(torch,a.storage_dtype),not a.include_special,a.model,a.revision),indent=2))
if __name__=="__main__": main()
