#!/usr/bin/env python
import argparse,json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import _bootstrap
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);a,_=p.parse_known_args(argv);rows=[]
 for path in a.results_dir.glob("layer_*/*/metrics.json"):
  data=json.loads(path.read_text());layer=int(path.parents[1].name.split("_")[-1]);rows.extend({"layer":layer,"predictor":path.parent.name,**row} for row in data["rows"])
 frame=pd.DataFrame(rows)
 if len(frame):
  frame["oracle_cosine"]=frame.apply(lambda r:frame[(frame.layer==r.layer)&(frame.predictor==r.predictor)&(frame.method=="oracle")&(frame.retention==r.retention)].ffn_cosine.iloc[0],axis=1)
  def overhead(row):
   candidates=frame[(frame.layer==row.layer)&(frame.predictor==row.predictor)&(frame.method=="predictor")&(frame.ffn_cosine>=row.oracle_cosine)];return float(candidates.retention.min()-row.retention) if len(candidates) else float("nan")
  frame["retention_overhead"]=frame.apply(overhead,axis=1);plots=a.results_dir/"plots";plots.mkdir(exist_ok=True)
  for metric,name in (("captured_mass","importance_mass.png"),("ffn_cosine","ffn_cosine.png"),("relative_l2","relative_l2.png")):
   for keys,g in frame.groupby(["layer","predictor","method"]):plt.plot(g.retention,g[metric],label="/".join(map(str,keys)))
   plt.xlabel("Retention");plt.ylabel(metric);plt.legend(fontsize=5);plt.tight_layout();plt.savefig(plots/name,dpi=160);plt.close()
 frame.to_csv(a.results_dir/"cross_layer_summary.csv",index=False);print(frame.to_string(index=False))
if __name__=="__main__":main()
