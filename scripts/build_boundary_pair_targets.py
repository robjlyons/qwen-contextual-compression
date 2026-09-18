#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap
from predictor.boundary_pairs import build_boundary_pair_targets
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--layer",type=int,default=0);p.add_argument("--stage1-run",required=True);p.add_argument("--lock-retention",type=float,default=.45);p.add_argument("--final-retention",type=float,default=.5);p.add_argument("--candidate-retention",type=float,default=.65);p.add_argument("--remove-shortlist",type=int,default=32);p.add_argument("--add-shortlist",type=int,default=64);p.add_argument("--positive-pairs",type=int,default=64);p.add_argument("--negative-pairs",type=int,default=64);p.add_argument("--device",default="cuda");print(build_boundary_pair_targets(**vars(p.parse_args(argv))))
if __name__=="__main__":main()
