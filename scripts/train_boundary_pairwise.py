#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap
from predictor.boundary_pair_training import train_boundary_pairwise
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--layer",type=int,default=0);p.add_argument("--stage1-run",required=True);p.add_argument("--pair-target-dir",type=Path,required=True);p.add_argument("--run-name",required=True);p.add_argument("--lock-retention",type=float,default=.45);p.add_argument("--final-retention",type=float,default=.5);p.add_argument("--candidate-retention",type=float,default=.65);p.add_argument("--rerank-dim",type=int,choices=[16],default=16);p.add_argument("--boundary-init",choices=["zero_embedding_alpha_one"],default="zero_embedding_alpha_one");p.add_argument("--loss",choices=["pairwise_swap"],default="pairwise_swap");p.add_argument("--pair-temperature",type=float,default=1.);p.add_argument("--negative-weight",type=float,default=.5);p.add_argument("--epochs",type=int,default=50);p.add_argument("--patience",type=int,default=8);p.add_argument("--lr",type=float,default=5e-5);p.add_argument("--weight-decay",type=float,default=1e-4);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--seed",type=int,default=42);p.add_argument("--device",default="cuda");print(train_boundary_pairwise(**vars(p.parse_args(argv))))
if __name__=="__main__":main()
