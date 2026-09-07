#!/usr/bin/env python
import argparse,json
from pathlib import Path
import _bootstrap  # noqa: F401
from transformers import AutoTokenizer
from end_to_end.evaluation_runner import validate_wrapper
from end_to_end.model_patcher import OracleMLPPatcher
from extract.inspect_model import load_model

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--model",default="Qwen/Qwen3.8-27B");p.add_argument("--text",default="The quick brown fox tests a sparse feed-forward wrapper.");p.add_argument("--device-map",default="auto");p.add_argument("--offload-folder");p.add_argument("--cache-dir");p.add_argument("--dtype",default="bfloat16");p.add_argument("--revision");p.add_argument("--output",type=Path,default=Path("results/end_to_end/wrapper_validation.json"));a,_=p.parse_known_args(argv)
 tokenizer=AutoTokenizer.from_pretrained(a.model,cache_dir=a.cache_dir,revision=a.revision);model=load_model(a.model,a.dtype,a.device_map,a.offload_folder,a.revision,full_model=True,cache_dir=a.cache_dir);patcher=OracleMLPPatcher(model)
 try:result=validate_wrapper(model,tokenizer,patcher,a.text)
 finally:patcher.restore()
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
if __name__=="__main__":main()

