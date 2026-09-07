"""Held-out reconstruction sweep for original and learned neuron layouts."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from clustering.block_layout import select_blocks
from evaluation.metrics import output_metrics
from evaluation.permutation_metrics import aggregate_block_results
from extract.extract_ffn import LocatedFFN, gated_activations
from oracle.importance import importance_scores, topk_indices
from oracle.sparse_ffn import byte_estimates, parameter_accounting
from oracle.sweep_sparsity import activation_batches


def _reconstruct_variable(a: torch.Tensor, weight: torch.Tensor, indices: torch.Tensor,
                          valid: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    mask=torch.zeros_like(a,dtype=torch.bool)
    for row in range(len(a)):
        mask[row,indices[row,valid[row]]]=True
    return torch.nn.functional.linear(a.masked_fill(~mask,0),weight,bias)


def run_permuted_block_sweep(
    ffn: LocatedFFN, activation_dir: Path, output: Path, orderings: dict[str,np.ndarray],
    calibration_samples: int, retention=(.3,.4,.5), block_sizes=(16,32,64,128),
    importance: str="weighted_activation", modes=("equal_budget","equal_coverage"),
    batch_size: int=8, coverage_target: float=.99,
) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    device,dtype=ffn.gate_proj.weight.device,ffn.gate_proj.weight.dtype
    down=ffn.down_proj.weight; n=down.shape[1]; bias=ffn.down_proj.bias
    perms={name:torch.as_tensor(value,device=device,dtype=torch.long) for name,value in orderings.items()}
    rows=[]; dense_checks={name:[] for name in orderings if name!="original"}; global_sample=0
    with torch.inference_mode():
      for hidden in activation_batches(activation_dir,batch_size):
        batch_start=global_sample; global_sample+=len(hidden)
        if global_sample<=calibration_samples: continue
        if batch_start<calibration_samples: hidden=hidden[calibration_samples-batch_start:]; batch_start=calibration_samples
        x=hidden.to(device=device,dtype=dtype); a=gated_activations(x,ffn); dense=ffn.down_proj(a)
        base_scores=importance_scores(a,down,importance)
        for ordering,perm in perms.items():
          ap=a.index_select(1,perm); wp=down.index_select(1,perm); scores=base_scores.index_select(1,perm)
          permuted_dense=torch.nn.functional.linear(ap,wp,bias)
          if ordering!="original":
            check=output_metrics(dense,permuted_dense)
            dense_checks[ordering].append({k:v.cpu().numpy() for k,v in check.items()})
          if ordering!="original": continue
          for ratio in retention:
            k=max(1,min(n,math.ceil(ratio*n))); oracle=topk_indices(scores,k)
            individual=torch.nn.functional.linear(torch.zeros_like(ap).scatter(1,oracle,ap.gather(1,oracle)),wp,bias)
            metrics=output_metrics(dense,individual)
            accounting=parameter_accounting(ffn.gate_proj.weight,ffn.up_proj.weight,down,k)
            for row in range(len(x)):
              record={"sample":batch_start+row,"ordering":"individual","mode":"equal_budget","block_size":1,
                      "retention":ratio,"loaded_count":k,"loaded_fraction":k/n,"block_count":k,
                      "oracle_recall":1.,"block_utilisation":1.,"block_precision":1.,"block_expansion_factor":1.,
                      "weight_traffic_fraction":accounting.selected_ratio,
                      "selected_q4_bytes":byte_estimates(accounting.selected_parameters)["q4"],
                      "selected_q2_bytes":byte_estimates(accounting.selected_parameters)["q2"]}
              record.update({name:float(value[row].cpu()) for name,value in metrics.items()}); rows.append(record)
            for layout_name,layout_perm in perms.items():
              layout_a=a.index_select(1,layout_perm); layout_w=down.index_select(1,layout_perm)
              layout_scores=base_scores.index_select(1,layout_perm); layout_oracle=topk_indices(layout_scores,k)
              for block_size in block_sizes:
                for mode in modes:
                  selected=select_blocks(layout_scores,layout_oracle,block_size,mode,k,coverage_target)
                  sparse=_reconstruct_variable(layout_a,layout_w,selected.indices,selected.valid_mask,bias)
                  values=output_metrics(dense,sparse)
                  for row in range(len(x)):
                    loaded=int(selected.loaded_count[row]); account=parameter_accounting(ffn.gate_proj.weight,ffn.up_proj.weight,down,loaded)
                    record={"sample":batch_start+row,"ordering":layout_name,"mode":mode,"block_size":block_size,
                      "retention":ratio,"loaded_count":loaded,"loaded_fraction":loaded/n,
                      "block_count":int(selected.block_count[row]),"oracle_recall":float(selected.oracle_recall[row]),
                      "block_utilisation":float(selected.precision[row]),"block_precision":float(selected.precision[row]),
                      "block_expansion_factor":float(selected.expansion[row]),"weight_traffic_fraction":account.selected_ratio,
                      "selected_q4_bytes":byte_estimates(account.selected_parameters)["q4"],
                      "selected_q2_bytes":byte_estimates(account.selected_parameters)["q2"]}
                    record.update({name:float(value[row].cpu()) for name,value in values.items()}); rows.append(record)
        output.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(output,index=False)
    validation={}; validation_failures=[]
    for name,chunks in dense_checks.items():
        combined={key:np.concatenate([x[key] for x in chunks]) for key in chunks[0]} if chunks else {}
        validation[name]={"samples":len(next(iter(combined.values()))) if combined else 0,
          "mean_cosine":float(combined["cosine_similarity"].mean()) if combined else None,
          "max_relative_l2":float(combined["relative_l2"].max()) if combined else None,
          "max_absolute_error":float(combined["max_absolute_error"].max()) if combined else None}
        if combined and (validation[name]["mean_cosine"] < .9999 or validation[name]["max_relative_l2"] > .02):
            validation_failures.append(name)
    (output.parent/"dense_permutation_validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    if validation_failures:
        raise RuntimeError(
            "Dense permutation validation failed; sparse results must not be interpreted. "
            f"See dense_permutation_validation.json for: {validation_failures}"
        )
    frame=pd.DataFrame(rows); summary=aggregate_block_results(frame); summary.to_csv(output.parent/"validation_summary.csv",index=False)
    return frame,summary,validation
