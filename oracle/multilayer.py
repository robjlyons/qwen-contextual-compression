"""Individual-neuron oracle evaluation over aligned captured layers."""
from __future__ import annotations
import json, math, os, resource, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from evaluation.metrics import output_metrics
from extract.extract_ffn import LocatedFFN, explicit_dense_ffn, gated_activations
from oracle.importance import importance_scores, topk_indices
from oracle.sparse_ffn import byte_estimates, parameter_accounting, reconstruct


def multilayer_shards(directory:Path):
    for path in sorted(directory.glob("shard_*.pt")): yield path,torch.load(path,map_location="cpu",weights_only=True)


def _split(sample_id:int,total:int,fraction:float)->str:
    return "calibration" if sample_id<int(total*fraction) else "validation"


def run_layer_oracle(ffn:LocatedFFN,activation_dir:Path,output:Path,layer:int,total_samples:int,
                     retention:list[float],importance:str="weighted_activation",calibration_fraction:float=.8,
                     resume:bool=True)->dict:
    """Evaluate one materialized FFN; checkpoint progress at activation-shard boundaries."""
    started=time.perf_counter(); state_path=output.with_suffix(".state.json"); state=json.loads(state_path.read_text()) if resume and state_path.exists() else {"completed_shards":[]}
    completed=set(state["completed_shards"]); append=output.exists() and bool(completed); output.parent.mkdir(parents=True,exist_ok=True)
    if append:
        committed_ids=[]
        for shard_name in completed:
            committed_ids.extend(torch.load(activation_dir/shard_name,map_location="cpu",weights_only=True)["sample_ids"].tolist())
        existing=pd.read_csv(output); existing=existing[existing.sample_id.isin(committed_ids)]; existing.to_csv(output,index=False)
    device,dtype=ffn.gate_proj.weight.device,ffn.gate_proj.weight.dtype; n=ffn.down_proj.weight.shape[1]; bias=ffn.down_proj.bias
    frequency={ratio:np.zeros(n,np.int64) for ratio in retention if ratio in (.3,.4,.5)}
    overlap={ratio:{"adjacent_intersection":0,"adjacent_pairs":0,"random_intersection":0,"random_pairs":0} for ratio in frequency}; previous={ratio:{} for ratio in frequency}
    if resume and "frequency" in state:
        for ratio in frequency: frequency[ratio]=np.asarray(state["frequency"][str(ratio)],dtype=np.int64)
        overlap={float(key):value for key,value in state.get("overlap",{}).items()}
    verification=[]; processed=int(state.get("processed_samples",0))
    for path,payload in multilayer_shards(activation_dir):
      if path.name in completed: continue
      x=payload["inputs"].to(device=device,dtype=dtype); dense=ffn.module(x); explicit=explicit_dense_ffn(x,ffn); check=output_metrics(dense,explicit)
      verification.append({k:v.detach().cpu().numpy() for k,v in check.items()})
      a=gated_activations(x,ffn); scores=importance_scores(a,ffn.down_proj.weight,importance); records=[]
      selected_masks={}
      for ratio in retention:
        k=max(1,min(n,math.ceil(ratio*n))); indices=topk_indices(scores,k); sparse=reconstruct(a,ffn.down_proj.weight,indices,bias); metrics=output_metrics(dense,sparse)
        accounting=parameter_accounting(ffn.gate_proj.weight,ffn.up_proj.weight,ffn.down_proj.weight,k)
        if ratio in frequency:
            mask=torch.zeros((len(x),n),dtype=torch.bool,device=device).scatter_(1,indices,True); selected_masks[ratio]=mask
            frequency[ratio]+=mask.sum(0).cpu().numpy()
        for row,sample_id in enumerate(payload["sample_ids"].tolist()):
          record={"sample_id":sample_id,"prompt_id":int(payload["prompt_ids"][row]),"token_position":int(payload["token_positions"][row]),"token_id":int(payload["token_ids"][row]),
            "split":_split(sample_id,total_samples,calibration_fraction),"layer":layer,"retention":ratio,"retained_neurons":k,
            "dense_ffn_parameters":accounting.dense_parameters,"selected_ffn_parameters":accounting.selected_parameters,
            "weight_traffic_fraction":accounting.selected_ratio,"selected_q4_bytes":byte_estimates(accounting.selected_parameters)["q4"],"selected_q2_bytes":byte_estimates(accounting.selected_parameters)["q2"]}
          record.update({name:float(value[row].cpu()) for name,value in metrics.items()}); records.append(record)
      for ratio,mask in selected_masks.items():
        stats=overlap[ratio]
        for row in range(len(mask)):
          prompt=int(payload["prompt_ids"][row]); position=int(payload["token_positions"][row]); current=mask[row].cpu()
          if prompt in previous[ratio] and previous[ratio][prompt][0]+1==position:
            stats["adjacent_intersection"]+=int((current&previous[ratio][prompt][1]).sum()); stats["adjacent_pairs"]+=1
          previous[ratio][prompt]=(position,current)
        if len(mask)>1:
          generator=torch.Generator(device="cpu").manual_seed(layer*1_000_003+sum(path.name.encode())+int(ratio*1000))
          permutation=torch.randperm(len(mask),generator=generator).to(mask.device)
          random_partner=mask.index_select(0,permutation); stats["random_intersection"]+=int((mask&random_partner).sum()); stats["random_pairs"]+=len(mask)
      pd.DataFrame(records).to_csv(output,mode="a" if append else "w",header=not append,index=False); append=True; processed+=len(x); completed.add(path.name)
      saved_state={"completed_shards":sorted(completed),"processed_samples":processed,"overlap":{str(key):value for key,value in overlap.items()},"frequency":{str(key):value.tolist() for key,value in frequency.items()}}
      temporary=state_path.with_suffix(".json.tmp"); temporary.write_text(json.dumps(saved_state)+"\n"); os.replace(temporary,state_path)
    verified={}
    if verification:
      values={key:np.concatenate([chunk[key] for chunk in verification]) for key in verification[0]}; verified={"mean_cosine":float(values["cosine_similarity"].mean()),"max_relative_l2":float(values["relative_l2"].max()),"max_absolute_error":float(values["max_absolute_error"].max()),"allclose":bool(values["max_absolute_error"].max()<1e-2)}
      if verified["mean_cosine"]<.9999 or not verified["allclose"]: raise RuntimeError(f"Layer {layer} explicit FFN verification failed: {verified}")
    stability=[]
    for ratio,freq in frequency.items():
      distribution=freq/max(freq.sum(),1); nonzero=distribution[distribution>0]; entropy=float(-(nonzero*np.log2(nonzero)).sum()/np.log2(n)); sorted_values=np.sort(freq); gini=float((2*np.arange(1,n+1)-n-1)@sorted_values/(n*max(sorted_values.sum(),1)))
      stats=overlap[ratio]; k=math.ceil(ratio*n); stability.append({"layer":layer,"retention":ratio,"samples":processed,"normalised_frequency_entropy":entropy,"frequency_gini":gini,"top10_selection_concentration":float(np.sort(freq)[::-1][:math.ceil(.1*n)].sum()/max(freq.sum(),1)),
        "adjacent_pairs":stats["adjacent_pairs"],"adjacent_jaccard":stats["adjacent_intersection"]/max(stats["adjacent_pairs"]*(2*k)-stats["adjacent_intersection"],1),"adjacent_reuse_fraction":stats["adjacent_intersection"]/max(stats["adjacent_pairs"]*k,1),
        "random_jaccard":stats["random_intersection"]/max(stats["random_pairs"]*(2*k)-stats["random_intersection"],1)})
    elapsed=time.perf_counter()-started
    return {"layer":layer,"processed_samples":processed,"wall_seconds":elapsed,"samples_per_second":processed/max(elapsed,1e-9),"peak_gpu_vram_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,"peak_system_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"dense_verification":verified,"stability":stability}
