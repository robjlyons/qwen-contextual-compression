#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap
from predictor.candidate_training import evaluate_candidate_reranker

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--layer",type=int,default=0);p.add_argument("--run-name",required=True);p.add_argument("--device",default="cuda");p.add_argument("--split",choices=["validation","test"],default="test");a=p.parse_args(argv);print(evaluate_candidate_reranker(a.results_dir,a.layer,a.run_name,a.device,a.split))
if __name__=="__main__":main()
