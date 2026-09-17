#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap
from runtime.weights import extract_ffn_layer_weights
def main(argv=None):
 p=argparse.ArgumentParser();source=p.add_mutually_exclusive_group(required=True);source.add_argument("--model");source.add_argument("--model-dir",type=Path);p.add_argument("--output",type=Path,default=Path("results/runtime_weights/layer_000.safetensors"));p.add_argument("--layer",type=int,default=0);p.add_argument("--revision");p.add_argument("--cache-dir",type=Path);p.add_argument("--token");p.add_argument("--dtype",choices=["fp16","bf16"],default="fp16");print(extract_ffn_layer_weights(**vars(p.parse_args(argv))))
if __name__=="__main__":main()
