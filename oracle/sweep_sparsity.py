"""Streaming oracle, structured, static, and random FFN experiments."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from evaluation.metrics import output_metrics
from evaluation.analyse_neurons import NeuronAccumulator, hot_coverage
from extract.extract_ffn import LocatedFFN, gated_activations
from oracle.importance import importance_scores, topk_indices, block_topk_indices, random_indices
from oracle.sparse_ffn import reconstruct, parameter_accounting, byte_estimates


def activation_batches(directory: Path, batch_size: int):
    for path in sorted(directory.glob("inputs_*.pt")):
        payload=torch.load(path,map_location="cpu",weights_only=True); values=payload["inputs"]
        for start in range(0,len(values),batch_size): yield values[start:start+batch_size]


def run_sweep(ffn: LocatedFFN, activation_dir: Path, output: Path, retention: list[float],
              metric: str="weighted_activation", batch_size: int=8, block_sizes=(32,64,128,256),
              random_repeats: int=3, seed: int=42) -> tuple[pd.DataFrame,pd.DataFrame]:
    device=ffn.gate_proj.weight.device; dtype=ffn.gate_proj.weight.dtype
    down=ffn.down_proj.weight; n=down.shape[1]; norms=torch.linalg.vector_norm(down.float(),dim=0)
    accumulator=NeuronAccumulator(n)
    # Pass one establishes global hotness without retaining per-token tensors.
    with torch.inference_mode():
        for x in activation_batches(activation_dir,batch_size):
            a=gated_activations(x.to(device=device,dtype=dtype),ffn)
            weighted=a.abs().float()*norms.to(device)
            scores=importance_scores(a,down,metric)
            accumulator.update(scores.cpu().numpy(),a.float().cpu().numpy(),weighted.cpu().numpy())
    stats=accumulator.frame(); output.parent.mkdir(parents=True,exist_ok=True)
    stats.to_csv(output.parent/"neuron_stats.csv",index=False); hot_coverage(stats).to_csv(output.parent/"hot_coverage.csv",index=False)
    hot_order=torch.tensor(stats.sort_values("selection_count",ascending=False).neuron.to_numpy(),device=device)
    weight_order=torch.argsort(norms.to(device),descending=True); rows=[]; sample_offset=0
    bias=ffn.down_proj.bias; bias_count=sum(m.bias.numel() for m in (ffn.gate_proj,ffn.up_proj,ffn.down_proj) if getattr(m,"bias",None) is not None)
    with torch.inference_mode():
      for x in activation_batches(activation_dir,batch_size):
        x=x.to(device=device,dtype=dtype); a=gated_activations(x,ffn); dense=ffn.down_proj(a); scores=importance_scores(a,down,metric)
        for ratio in retention:
          k=max(1,min(n,math.ceil(ratio*n)))
          methods=[("oracle",topk_indices(scores,k)),("static_hot",hot_order[:k].expand(len(x),-1)),
                   ("static_weight",weight_order[:k].expand(len(x),-1))]
          for block in block_sizes: methods.append((f"block{block}",block_topk_indices(scores,k,block).clamp_max(n-1)))
          for repeat in range(random_repeats): methods.append((f"random_{repeat}",random_indices(len(x),n,k,seed+repeat+sample_offset,device)))
          for method,indices in methods:
            accounting=parameter_accounting(ffn.gate_proj.weight,ffn.up_proj.weight,down,indices.shape[1],bias_count)
            sparse=reconstruct(a,down,indices,bias); metrics=output_metrics(dense,sparse)
            for i in range(len(x)):
              row={"sample":sample_offset+i,"method":method,"retention":ratio,"k":indices.shape[1],
                   "theoretical_neuron_fraction":indices.shape[1]/n,"down_weight_fraction":indices.shape[1]/n,
                   "theoretical_selected_parameters":accounting.selected_parameters,
                   "dense_ffn_parameters":accounting.dense_parameters,"theoretical_weight_traffic_fraction":accounting.selected_ratio,
                   "actual_oracle_executes_dense_gate_up":True}
              row.update({key:float(value[i].cpu()) for key,value in metrics.items()})
              row.update({f"selected_{q}_bytes":v for q,v in byte_estimates(accounting.selected_parameters).items()})
              row.update({f"dense_{q}_bytes":v for q,v in byte_estimates(accounting.dense_parameters).items()}); rows.append(row)
        sample_offset += len(x)
        pd.DataFrame(rows).to_csv(output,index=False) # checkpoint, restart-safe output replacement
    frame=pd.DataFrame(rows)
    summary=(frame.assign(method=frame.method.str.replace(r"random_\d+","random",regex=True))
             .groupby(["method","retention"],as_index=False).agg(
               cosine_similarity=("cosine_similarity","mean"), cosine_std=("cosine_similarity","std"),
               relative_l2=("relative_l2","mean"),relative_l2_std=("relative_l2","std"),
               mse=("mse","mean"),max_absolute_error=("max_absolute_error","mean"),
               theoretical_weight_traffic_fraction=("theoretical_weight_traffic_fraction","mean"),
               theoretical_selected_parameters=("theoretical_selected_parameters","mean"),dense_ffn_parameters=("dense_ffn_parameters","mean")))
    summary.to_csv(output.parent/"summary.csv",index=False)
    return frame,stats
