#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap
from runtime.selector_benchmark import run_selector_benchmark
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--runtime-weights",type=Path,required=True);p.add_argument("--layer",type=int,default=0);p.add_argument("--stage1-run",required=True);p.add_argument("--retention",type=float,default=.5);p.add_argument("--warmup",type=int,default=20);p.add_argument("--iterations",type=int,default=200);p.add_argument("--device",default="cuda");p.add_argument("--compile-variant",action="store_true");print(run_selector_benchmark(**vars(p.parse_args(argv)))["output"])
if __name__=="__main__":main()
