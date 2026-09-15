#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap
from extract.inspect_model import load_model
from extract.extract_ffn import locate_ffn
from predictor.targets import build_targets
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--activation-dir",type=Path,required=True);p.add_argument("--layers",default="0,40,63");p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--device",default="cuda");p.add_argument("--dtype",default="bfloat16");p.add_argument("--max-samples",type=int);p.add_argument("--transform",default="distribution");a,_=p.parse_known_args(argv)
 for layer in map(int,a.layers.split(",")):
  model=load_model(a.model,a.dtype,layer=layer,device=a.device);build_targets(locate_ffn(model,layer),a.activation_dir/f"layer_{layer:03d}",a.output_dir/f"layer_{layer:03d}/targets",layer,transform=a.transform,max_samples=a.max_samples);del model
if __name__=="__main__":main()
