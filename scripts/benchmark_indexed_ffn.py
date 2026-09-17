#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap
from runtime.indexed_benchmark import run_indexed_benchmark
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--runtime-weights",type=Path,required=True);p.add_argument("--layer",type=int,default=0);p.add_argument("--stage1-run",required=True);p.add_argument("--selector-variant",choices=["baseline_fp32","folded_norm_fp32","fp16_predictor","fp16_folded_norm"],default="fp16_folded_norm");p.add_argument("--id-order",choices=["all","unsorted","index_sorted","score_sorted"],default="all");p.add_argument("--warmup",type=int,default=20);p.add_argument("--iterations",type=int,default=200);p.add_argument("--device",default="cuda");a=vars(p.parse_args(argv));order=a.pop("id_order");orders=("unsorted","index_sorted","score_sorted") if order=="all" else (order,);print("\n".join(run_indexed_benchmark(id_order=value,**a)["output"] for value in orders))
if __name__=="__main__":main()
