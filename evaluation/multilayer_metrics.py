"""Aggregation, thresholds, confidence intervals, and working-set estimates."""
from __future__ import annotations
import numpy as np
import pandas as pd


def adjacent_pair_indices(prompt_ids,token_positions)->list[tuple[int,int]]:
    """Return only within-sequence consecutive-token pairs."""
    previous={}; pairs=[]
    for index,(prompt,position) in enumerate(zip(prompt_ids,token_positions)):
        prompt,position=int(prompt),int(position)
        if prompt in previous and previous[prompt][1]+1==position: pairs.append((previous[prompt][0],index))
        previous[prompt]=(index,position)
    return pairs


def aggregate_oracle(rows:pd.DataFrame,group_columns:list[str])->pd.DataFrame:
    metrics=["cosine_similarity","relative_l2","mse","max_absolute_error"]
    result=[]
    for keys,data in rows.groupby(group_columns,dropna=False):
        keys=(keys,) if not isinstance(keys,tuple) else keys; record=dict(zip(group_columns,keys)); record["samples"]=len(data)
        for metric in metrics:
            values=data[metric].to_numpy(); record.update({f"{metric}_{name}":float(value) for name,value in (("mean",values.mean()),("median",np.median(values)),("std",values.std(ddof=1) if len(values)>1 else 0),("p90",np.quantile(values,.90)),("p95",np.quantile(values,.95)),("p99",np.quantile(values,.99)),("min",values.min()),("max",values.max()))})
        result.append(record)
    return pd.DataFrame(result)


def minimum_retention(summary:pd.DataFrame,column:str,target:float,greater:bool=True)->float:
    valid=summary[summary[column]>=target] if greater else summary[summary[column]<=target]
    return float(valid.retention.min()) if len(valid) else np.nan


def threshold_table(summary:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for layer,data in summary.groupby("layer"):
        row={"layer":layer}
        for target in (.99,.995,.999,.9995): row[f"keep_mean_cosine_{target}"]=minimum_retention(data,"cosine_similarity_mean",target)
        for target in (.10,.05,.02,.01): row[f"keep_mean_l2_{target}"]=minimum_retention(data,"relative_l2_mean",target,False)
        row["keep_p95_cosine_0.99"]=minimum_retention(data,"cosine_similarity_p95",.99)
        row["keep_p99_cosine_0.99"]=minimum_retention(data,"cosine_similarity_p99",.99); rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_intervals(rows:pd.DataFrame,repeats:int=500,seed:int=42)->pd.DataFrame:
    rng=np.random.default_rng(seed); result=[]
    subset=rows[rows.retention.isin([.4,.5])]
    for (layer,retention),data in subset.groupby(["layer","retention"]):
      for metric in ("cosine_similarity","relative_l2"):
        values=data[metric].to_numpy(); boot=np.empty(repeats)
        for index in range(repeats): boot[index]=rng.choice(values,len(values),replace=True).mean()
        result.append({"layer":layer,"retention":retention,"metric":metric,"samples":len(values),"mean":values.mean(),"ci95_low":np.quantile(boot,.025),"ci95_high":np.quantile(boot,.975),"resamples":repeats})
    return pd.DataFrame(result)


def position_bucket(position:int)->str:
    if position<=32:return "0-32"
    if position<=128:return "33-128"
    if position<=512:return "129-512"
    return "513+"


def working_set_estimates(summary:pd.DataFrame,thresholds:pd.DataFrame,total_layers:int)->pd.DataFrame:
    rows=[]; dense_by_layer=summary.groupby("layer").dense_ffn_parameters.first(); required=thresholds.set_index("layer")["keep_mean_cosine_0.999"]
    known=required.dropna(); dense_tested=float(dense_by_layer.sum()); selected=sum(float(dense_by_layer[layer])*float(retention) for layer,retention in known.items())
    fraction=selected/sum(float(dense_by_layer[layer]) for layer in known.index) if len(known) else np.nan
    for scenario,retention in [("layer_specific_cosine_0.999",fraction),("fixed_30_percent",.3),("fixed_40_percent",.4),("fixed_50_percent",.5)]:
        estimated_dense=float(dense_by_layer.mean()*total_layers); selected_parameters=estimated_dense*retention
        record={"scenario":scenario,"scope":"representative-layer estimate; no interpolation of errors","tested_layers":len(dense_by_layer),"total_model_layers":total_layers,"active_fraction":retention,"estimated_dense_ffn_parameters":estimated_dense,"estimated_selected_ffn_parameters":selected_parameters}
        for name,bits in {"bf16":16,"q8":8,"q4":4,"q3":3,"q2":2}.items(): record[f"selected_{name}_bytes"]=selected_parameters*bits/8
        rows.append(record)
    worst=float(required.max()) if len(known) else np.nan; rows.append({"scenario":"cautious_worst_tested_layer_cosine_0.999","scope":"representative-layer worst case","tested_layers":len(dense_by_layer),"total_model_layers":total_layers,"active_fraction":worst,"estimated_dense_ffn_parameters":dense_by_layer.mean()*total_layers,"estimated_selected_ffn_parameters":dense_by_layer.mean()*total_layers*worst})
    return pd.DataFrame(rows)
