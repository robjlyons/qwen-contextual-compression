#!/usr/bin/env python
"""Build a compact architecture/stage quality and compute comparison."""
import argparse,json
from pathlib import Path
import pandas as pd
import _bootstrap

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--layer",type=int,default=0);p.add_argument("--retention",type=float,default=.5);a=p.parse_args(argv);rows=[]
 for path in a.results_dir.glob(f"layer_{a.layer:03d}/*/metrics.json"):
  data=json.loads(path.read_text())
  for row in data.get("rows",[]):
   if abs(row.get("retention",-1)-a.retention)<1e-9:rows.append({"run":path.parent.name,**row})
 for path in a.results_dir.glob(f"layer_{a.layer:03d}/*/evaluation.json"):
  data=json.loads(path.read_text());metrics=data["metrics"];accounting=data["accounting"];config=data["config"];rows.append({"run":path.parent.name,"model":"candidate_reranker","latent_dim":config["rerank_dim"],"stage":"fine_tuned" if config.get("init_checkpoint") else "ce","method":"predictor","predictor_mac_fraction":accounting["total_mac_fraction"],**metrics})
 frame=pd.DataFrame(rows)
 columns=["run","model","latent_dim","stage","method","predictor_mac_fraction","ffn_cosine","ffn_cosine_p01","ffn_cosine_p05","relative_l2","relative_l2_p95","relative_l2_p99","captured_mass"]
 if len(frame):
  for column in columns:
   if column not in frame:frame[column]=None
  frame=frame[columns].sort_values(["method","ffn_cosine"],ascending=[True,False])
 frame.to_csv(a.results_dir/f"layer_{a.layer:03d}_phase43_comparison.csv",index=False);print(frame.to_string(index=False))
if __name__=="__main__":main()
