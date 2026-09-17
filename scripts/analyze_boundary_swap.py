#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap
from predictor.boundary_analysis import run_boundary_analysis
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--layer",type=int,default=0);p.add_argument("--stage1-run",required=True);p.add_argument("--lock-fractions",default=".40,.45");p.add_argument("--candidate-fractions",default=".60,.65");p.add_argument("--final-retention",type=float,default=.5);p.add_argument("--split",choices=["validation","test"],default="validation");p.add_argument("--device",default="cuda");a=p.parse_args(argv);print(run_boundary_analysis(a.results_dir,a.layer,a.stage1_run,[float(x) for x in a.lock_fractions.split(",")],[float(x) for x in a.candidate_fractions.split(",")],a.final_retention,a.split,a.device))
if __name__=="__main__":main()
