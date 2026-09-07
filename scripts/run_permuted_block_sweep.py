#!/usr/bin/env python
import argparse,json
from pathlib import Path
import numpy as np
import _bootstrap  # noqa: F401
from clustering.block_sweep import run_permuted_block_sweep
from extract.extract_ffn import locate_ffn
from extract.inspect_model import load_model

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--activation-dir",type=Path,required=True); p.add_argument("--clustering-dir",type=Path,required=True); p.add_argument("--model",default="Qwen/Qwen3.8-27B"); p.add_argument("--layer",type=int,default=0); p.add_argument("--retention",default="0.30,0.40,0.50"); p.add_argument("--block-sizes",default="16,32,64,128"); p.add_argument("--modes",default="equal_budget,equal_coverage"); p.add_argument("--coverage-target",type=float,default=.99); p.add_argument("--importance",default="weighted_activation"); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--dtype",default="bfloat16"); p.add_argument("--device"); p.add_argument("--cache-dir"); p.add_argument("--full-model",action="store_true"); a,_=p.parse_known_args(argv)
 config=json.loads((a.clustering_dir/"config.json").read_text()); model=load_model(a.model,a.dtype,layer=a.layer,device=a.device,full_model=a.full_model,cache_dir=a.cache_dir); data=np.load(a.clustering_dir/"orderings.npz"); orders={name:data[name] for name in data.files if name in {"original","frequency","clustered","random","hybrid","greedy","signature"}}
 _,_,validation=run_permuted_block_sweep(locate_ffn(model,a.layer),a.activation_dir,a.clustering_dir/"block_sweep.csv",orders,int(config["calibration_samples"]),[float(x) for x in a.retention.split(",")],[int(x) for x in a.block_sizes.split(",")],a.importance,a.modes.split(","),a.batch_size,a.coverage_target); print("Dense permutation validation"); print(json.dumps(validation,indent=2))
if __name__=="__main__": main()
