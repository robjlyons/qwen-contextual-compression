#!/usr/bin/env python
import argparse, json
import _bootstrap  # noqa: F401
from extract.inspect_model import load_model, inspect

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--model",default="Qwen/Qwen3.8-27B"); p.add_argument("--layer",type=int,default=0); p.add_argument("--dtype",default="bfloat16"); p.add_argument("--device-map",default="auto"); p.add_argument("--offload-folder"); p.add_argument("--revision")
 a,_=p.parse_known_args(argv); model=load_model(a.model,a.dtype,a.device_map,a.offload_folder,a.revision); print(json.dumps(inspect(model,a.layer),indent=2))
if __name__=="__main__": main()
