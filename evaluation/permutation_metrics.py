"""Permutation locality, distribution summaries, and paired bootstrap tests."""
from __future__ import annotations
import numpy as np
import pandas as pd
from clustering.permutation import inverse_permutation
from clustering.similarity import SparseGraph


def locality_metrics(graph: SparseGraph, permutation, neighbourhoods=(16,32,64,128,256)) -> dict[str,float]:
    positions=inverse_permutation(permutation); distance=np.abs(positions[graph.source]-positions[graph.target])
    total=max(float(graph.affinity.sum()),1e-12)
    result={"weighted_mean_distance":float((distance*graph.affinity).sum()/total)}
    result.update({f"affinity_within_{width}":float(graph.affinity[distance<width].sum()/total) for width in neighbourhoods})
    return result


def aggregate_block_results(frame: pd.DataFrame) -> pd.DataFrame:
    groups=["ordering","mode","block_size","retention"]
    metrics=["cosine_similarity","relative_l2","mse","max_absolute_error","oracle_recall",
             "loaded_count","loaded_fraction","block_count","block_utilisation","block_precision",
             "block_expansion_factor","weight_traffic_fraction","selected_q4_bytes","selected_q2_bytes"]
    rows=[]
    for keys,data in frame.groupby(groups,dropna=False):
        row=dict(zip(groups,keys)); row["samples"]=len(data)
        for metric in metrics:
            values=data[metric].to_numpy(); row.update({f"{metric}_{name}":float(value) for name,value in
              (("mean",values.mean()),("median",np.median(values)),("p90",np.quantile(values,.9)),
               ("p95",np.quantile(values,.95)),("p99",np.quantile(values,.99)),("std",values.std(ddof=1) if len(values)>1 else 0))})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(frame: pd.DataFrame, best_ordering: str, block_size: int,
                     retention: float, repeats: int=2000, seed: int=42) -> pd.DataFrame:
    subset=frame[(frame.block_size==block_size)&np.isclose(frame.retention,retention)&(frame["mode"]=="equal_budget")]
    metrics=["cosine_similarity","relative_l2","block_expansion_factor","oracle_recall"]
    rng=np.random.default_rng(seed); rows=[]
    for metric in metrics:
        pivot=subset.pivot(index="sample",columns="ordering",values=metric).dropna()
        if best_ordering not in pivot or "original" not in pivot: continue
        delta=(pivot[best_ordering]-pivot["original"]).to_numpy(); means=np.empty(repeats)
        for i in range(repeats): means[i]=rng.choice(delta,size=len(delta),replace=True).mean()
        rows.append({"metric":metric,"comparison":f"{best_ordering}-original","mean_delta":delta.mean(),
                     "ci95_low":np.quantile(means,.025),"ci95_high":np.quantile(means,.975),"bootstrap_repeats":repeats})
    return pd.DataFrame(rows)
