#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap
from runtime.benchmark import run_benchmark
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--runtime-weights",type=Path,required=True);p.add_argument("--layer",type=int,default=0);p.add_argument("--stage1-run",required=True);p.add_argument("--retention",type=float,default=.5);p.add_argument("--device",default="cuda");p.add_argument("--dtype",choices=["fp16","bf16"],default="fp16");p.add_argument("--warmup",type=int,default=20);p.add_argument("--iterations",type=int,default=200);p.add_argument("--profile",action="store_true");p.add_argument("--benchmark-host-transfer",action="store_true");result=run_benchmark(**vars(p.parse_args(argv)));print(result["output"]);print(result["warning"])
if __name__=="__main__":main()
