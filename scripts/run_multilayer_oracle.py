#!/usr/bin/env python
import argparse,gc,json
from pathlib import Path
import pandas as pd
import torch
import _bootstrap  # noqa: F401
from extract.extract_ffn import locate_ffn
from extract.inspect_model import load_model
from oracle.multilayer import run_layer_oracle

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--activation-dir",type=Path,required=True);p.add_argument("--layers",required=True);p.add_argument("--retention",default="0.1,0.2,0.3,0.4,0.5,0.6,0.75,1.0");p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--model",default="Qwen/Qwen3.8-27B");p.add_argument("--importance",default="weighted_activation");p.add_argument("--calibration-fraction",type=float,default=.8);p.add_argument("--dtype",default="bfloat16");p.add_argument("--device");p.add_argument("--cache-dir");p.add_argument("--revision");p.add_argument("--no-resume",action="store_true")
 a,_=p.parse_known_args(argv);layers=[int(x) for x in a.layers.split(",")];retention=[float(x) for x in a.retention.split(",")];metadata_path=a.activation_dir.parent/"metadata.json"
 if not metadata_path.is_file(): raise FileNotFoundError(f"Multi-layer capture is incomplete: missing {metadata_path}. Run capture_multilayer_activations.py successfully before oracle evaluation.")
 metadata=json.loads(metadata_path.read_text());missing=[str(a.activation_dir/f"layer_{layer:03d}") for layer in layers if not (a.activation_dir/f"layer_{layer:03d}").is_dir()]
 if missing: raise FileNotFoundError("Missing captured layer directories: "+", ".join(missing))
 a.output_dir.mkdir(parents=True,exist_ok=True);runs=[]
 for layer in layers:
  model=load_model(a.model,a.dtype,layer=layer,device=a.device,cache_dir=a.cache_dir,revision=a.revision);output=a.output_dir/f"layer_{layer:03d}.csv";run=run_layer_oracle(locate_ffn(model,layer),a.activation_dir/f"layer_{layer:03d}",output,layer,int(metadata["samples_per_layer"]),retention,a.importance,a.calibration_fraction,not a.no_resume);pd.DataFrame(run.pop("stability")).to_csv(a.output_dir/f"layer_{layer:03d}.stability.csv",index=False);(a.output_dir/f"layer_{layer:03d}.metadata.json").write_text(json.dumps(run,indent=2)+"\n");runs.append(run);del model;gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None
 print(json.dumps(runs,indent=2))
if __name__=="__main__":main()
