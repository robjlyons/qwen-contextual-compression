#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap  # noqa: F401
from evaluation.permutation_report import analyse_permutation
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--clustering-dir",type=Path,required=True); p.add_argument("--bootstrap-repeats",type=int,default=2000); p.add_argument("--seed",type=int,default=42); a,_=p.parse_known_args(argv); print(analyse_permutation(a.clustering_dir,a.bootstrap_repeats,a.seed))
if __name__=="__main__": main()
