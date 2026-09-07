#!/usr/bin/env python
"""Build compact calibration masks, signatures, and a sparse exact-affinity graph."""
import argparse,json
from pathlib import Path
import numpy as np
import yaml
import _bootstrap  # noqa: F401
from clustering.coactivation import build_packed_selection_mask,load_mask,selection_signatures,packed_sample_prefix
from clustering.similarity import build_sparse_graph,graph_statistics
from extract.extract_ffn import locate_ffn
from extract.inspect_model import load_model

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--activation-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path); p.add_argument("--model",default="Qwen/Qwen3.8-27B"); p.add_argument("--layer",type=int,default=0); p.add_argument("--retention",default="0.20,0.30,0.40,0.50"); p.add_argument("--clustering-retention",type=float,default=.4); p.add_argument("--importance",default="weighted_activation"); p.add_argument("--calibration-fraction",type=float,default=.8); p.add_argument("--signature-dimensions",type=int,default=64); p.add_argument("--similarity",choices=["raw","jaccard","cosine","conditional","npmi"],default="jaccard"); p.add_argument("--top-neighbours",type=int,default=32); p.add_argument("--min-affinity",type=float,default=.05); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--seed",type=int,default=42); p.add_argument("--dtype",default="bfloat16"); p.add_argument("--device"); p.add_argument("--cache-dir"); p.add_argument("--full-model",action="store_true")
 a,_=p.parse_known_args(argv); out=a.output_dir or a.activation_dir/"clustering"; out.mkdir(parents=True,exist_ok=True); model=load_model(a.model,a.dtype,layer=a.layer,device=a.device,full_model=a.full_model,cache_dir=a.cache_dir); ffn=locate_ffn(model,a.layer)
 masks={}
 for ratio in map(float,a.retention.split(",")):
  path=out/f"oracle_mask_r{ratio:.3f}.npz"; masks[ratio]=build_packed_selection_mask(ffn,a.activation_dir,path,ratio,a.importance,a.model,a.layer,a.batch_size,a.seed,a.calibration_fraction)
 ratio=min(masks,key=lambda x:abs(x-a.clustering_retention)); path=out/f"oracle_mask_r{ratio:.3f}.npz"; packed,meta=load_mask(path); calibration=int(meta.samples*meta.calibration_fraction)
 signatures=selection_signatures(packed,meta.samples,a.signature_dimensions,a.seed,sample_stop=calibration); np.save(out/"selection_signatures.npy",signatures)
 graph=build_sparse_graph(packed_sample_prefix(packed,calibration),calibration,signatures,a.similarity,a.top_neighbours,a.min_affinity); graph.save(out/"coactivation_graph.npz"); (out/"graph_stats.json").write_text(json.dumps(graph_statistics(graph),indent=2)+"\n")
 config=vars(a)|{"output_dir":str(out),"activation_dir":str(a.activation_dir),"mask_files":{str(k):f"oracle_mask_r{k:.3f}.npz" for k in masks},"calibration_samples":calibration,"validation_samples":meta.samples-calibration}
 serializable=json.loads(json.dumps(config,default=str)); (out/"config.json").write_text(json.dumps(serializable,indent=2)+"\n"); (out/"config.yaml").write_text(yaml.safe_dump(serializable,sort_keys=True))
if __name__=="__main__": main()
