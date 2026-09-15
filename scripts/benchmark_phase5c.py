#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap
from runtime.phase5c_benchmark import run_phase5c
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--runtime-weights",type=Path,required=True);p.add_argument("--stage1-run",required=True);p.add_argument("--layer",type=int,default=0);p.add_argument("--warmup",type=int,default=20);p.add_argument("--iterations",type=int,default=100);p.add_argument("--rounds",type=int,default=7);p.add_argument("--seed",type=int,default=42);p.add_argument("--device",default="cuda");print(run_phase5c(**vars(p.parse_args(argv)))["output"])
if __name__=="__main__":main()
