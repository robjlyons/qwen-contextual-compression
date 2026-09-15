#!/usr/bin/env python
import argparse, json
import _bootstrap  # noqa: F401
from pathlib import Path
from extract.inspect_model import load_model, inspect
from extract.extract_ffn import locate_ffn
from oracle.sweep_sparsity import run_sweep
DEFAULT="1,.9,.75,.6,.5,.4,.3,.25,.2,.15,.1,.05"
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--model",default="Qwen/Qwen3.8-27B"); p.add_argument("--layer",type=int,default=0); p.add_argument("--activation-dir",type=Path,required=True); p.add_argument("--output",type=Path); p.add_argument("--retention",default=DEFAULT); p.add_argument("--importance",default="weighted_activation"); p.add_argument("--block-sizes",default="32,64,128,256"); p.add_argument("--random-repeats",type=int,default=3); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--seed",type=int,default=42); p.add_argument("--dtype",default="bfloat16"); p.add_argument("--device-map",default="auto"); p.add_argument("--device"); p.add_argument("--offload-folder"); p.add_argument("--revision"); p.add_argument("--cache-dir"); p.add_argument("--full-model",action="store_true"); p.add_argument("--trust-remote-code",action="store_true")
 a,_=p.parse_known_args(argv); output=a.output or a.activation_dir/"oracle.csv"; model=load_model(a.model,a.dtype,a.device_map,a.offload_folder,a.revision,layer=a.layer,device=a.device,full_model=a.full_model,cache_dir=a.cache_dir,trust_remote_code=a.trust_remote_code); info=inspect(model,a.layer); (output.parent/"experiment_config.json").write_text(json.dumps(vars(a)|{"output":str(output),"activation_dir":str(a.activation_dir),"inspection":info},default=str,indent=2)+"\n")
 run_sweep(locate_ffn(model,a.layer),a.activation_dir,output,[float(x) for x in a.retention.split(",")],a.importance,a.batch_size,[int(x) for x in a.block_sizes.split(",")],a.random_repeats,a.seed)
if __name__=="__main__": main()
