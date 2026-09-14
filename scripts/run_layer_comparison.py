#!/usr/bin/env python
"""Summarise completed representative-layer experiments without launching huge jobs."""
import argparse
import _bootstrap  # noqa: F401
from pathlib import Path
import pandas as pd
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--results-root",type=Path,default=Path("results")); p.add_argument("--layers",default="0,8,16,24,32,40,48,56,63"); p.add_argument("--output",type=Path,default=Path("results/layer_comparison.csv")); a,_=p.parse_known_args(argv); rows=[]
 for layer in map(int,a.layers.split(",")):
  path=a.results_root/f"layer{layer}"/"summary.csv"
  if not path.exists(): continue
  d=pd.read_csv(path); d=d[d.method=="oracle"]; at30=d.iloc[(d.retention-.3).abs().argsort()[:1]]; stats=pd.read_csv(path.parent/"hot_coverage.csv"); rows.append({"layer":layer,"keep_cosine_99":d[d.cosine_similarity>=.99].retention.min(),"keep_cosine_999":d[d.cosine_similarity>=.999].retention.min(),"top10_hot_coverage":stats.iloc[max(0,int(.1*len(stats))-1)].selection_coverage,"relative_l2_at_30":at30.relative_l2.iloc[0]})
 a.output.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False)
if __name__=="__main__": main()
