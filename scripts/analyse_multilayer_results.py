#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap  # noqa: F401
from evaluation.layer_comparison import analyse_multilayer
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--min-report-samples",type=int,default=1000);p.add_argument("--bootstrap-resamples",type=int,default=500);p.add_argument("--seed",type=int,default=42);p.add_argument("--output-suffix",default="");p.add_argument("--monotonic-tolerance",type=float,default=1e-5);a,_=p.parse_known_args(argv);print(analyse_multilayer(a.results_dir,a.min_report_samples,a.bootstrap_resamples,a.seed,a.output_suffix,a.monotonic_tolerance))
if __name__=="__main__":main()
