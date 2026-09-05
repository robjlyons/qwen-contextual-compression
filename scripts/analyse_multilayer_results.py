#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap  # noqa: F401
from evaluation.layer_comparison import analyse_multilayer
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--min-report-samples",type=int,default=1000);p.add_argument("--bootstrap-resamples",type=int,default=500);p.add_argument("--seed",type=int,default=42);a,_=p.parse_known_args(argv);print(analyse_multilayer(a.results_dir,a.min_report_samples,a.bootstrap_resamples))
if __name__=="__main__":main()

