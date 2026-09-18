#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap  # noqa: F401
from end_to_end.report import analyse_end_to_end
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--bootstrap-resamples",type=int,default=500);p.add_argument("--seed",type=int,default=42);p.add_argument("--top1-threshold",type=float,default=.98);p.add_argument("--relative-ppl-threshold",type=float,default=.03);a,_=p.parse_known_args(argv);print(analyse_end_to_end(a.results_dir,a.bootstrap_resamples,a.seed,a.top1_threshold,a.relative_ppl_threshold))
if __name__=="__main__":main()
