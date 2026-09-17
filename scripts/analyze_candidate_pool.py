#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap
from predictor.candidate_analysis import run_candidate_analysis

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--layer",type=int,default=0);p.add_argument("--stage1-run",required=True);p.add_argument("--candidate-retention",default=".55,.60,.65");p.add_argument("--final-retention",type=float,default=.5);p.add_argument("--device",default="cuda");p.add_argument("--split",choices=["validation","test"],default="validation");a=p.parse_args(argv);print(run_candidate_analysis(a.results_dir,a.layer,a.stage1_run,[float(x) for x in a.candidate_retention.split(",")],a.final_retention,a.device,a.split))
if __name__=="__main__":main()
