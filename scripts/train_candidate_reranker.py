#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap
from predictor.candidate_training import train_candidate_reranker

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--layer",type=int,default=0);p.add_argument("--stage1-run",required=True);p.add_argument("--candidate-retention",type=float,required=True);p.add_argument("--final-retention",type=float,default=.5);p.add_argument("--rerank-dim",type=int,choices=[16,32],default=16);p.add_argument("--loss",choices=["distribution_ce","output_hybrid_rank"],default="distribution_ce");p.add_argument("--run-name",required=True);p.add_argument("--init-checkpoint",type=Path);p.add_argument("--normalize-stage1-scores",action=argparse.BooleanOptionalAction,default=True);p.add_argument("--ranking-weight",type=float,default=.1);p.add_argument("--cosine-weight",type=float,default=1.);p.add_argument("--relative-weight",type=float,default=.25);p.add_argument("--lr",type=float);p.add_argument("--epochs",type=int,default=50);p.add_argument("--patience",type=int,default=8);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--weight-decay",type=float,default=1e-4);p.add_argument("--seed",type=int,default=42);p.add_argument("--device",default="cuda");a=p.parse_args(argv);print(train_candidate_reranker(**vars(a)))
if __name__=="__main__":main()
