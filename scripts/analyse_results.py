#!/usr/bin/env python
import argparse
import _bootstrap  # noqa: F401
from pathlib import Path
from evaluation.report import analyse
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--results-dir",type=Path,required=True); a,_=p.parse_known_args(argv); analyse(a.results_dir)
if __name__=="__main__": main()
