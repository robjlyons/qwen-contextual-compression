#!/usr/bin/env python
"""Resumable capture -> per-layer oracle -> report wrapper using subprocess isolation."""
import argparse,json,subprocess,sys
from pathlib import Path
import _bootstrap  # noqa: F401
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--model",default="Qwen/Qwen3.8-27B");p.add_argument("--layers",default="0,8,16,24,32,40,48,56,63");p.add_argument("--max-tokens-per-layer",type=int,default=2000);p.add_argument("--retention",default="0.1,0.2,0.3,0.4,0.5,0.6,0.75,1.0");p.add_argument("--output-dir",type=Path,default=Path("results/multilayer"));p.add_argument("--input");p.add_argument("--device-map",default="auto");p.add_argument("--offload-folder");p.add_argument("--cache-dir");p.add_argument("--dtype",default="bfloat16");p.add_argument("--min-report-samples",type=int,default=1000);a,_=p.parse_known_args(argv)
 capture=[sys.executable,"scripts/capture_multilayer_activations.py","--model",a.model,"--layers",a.layers,"--max-tokens-per-layer",str(a.max_tokens_per_layer),"--output-dir",str(a.output_dir),"--device-map",a.device_map,"--dtype",a.dtype];
 for flag,value in (("--input",a.input),("--offload-folder",a.offload_folder),("--cache-dir",a.cache_dir)):
  if value:capture.extend([flag,str(value)])
 subprocess.run(capture,check=True);actual_layers=",".join(map(str,json.loads((a.output_dir/"metadata.json").read_text())["layers"]));subprocess.run([sys.executable,"scripts/run_multilayer_oracle.py","--model",a.model,"--activation-dir",str(a.output_dir/"activations"),"--layers",actual_layers,"--retention",a.retention,"--output-dir",str(a.output_dir/"oracle"),"--dtype",a.dtype]+(["--cache-dir",a.cache_dir] if a.cache_dir else []),check=True);subprocess.run([sys.executable,"scripts/analyse_multilayer_results.py","--results-dir",str(a.output_dir),"--min-report-samples",str(a.min_report_samples)],check=True)
if __name__=="__main__":main()
