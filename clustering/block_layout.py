"""Fair block selection and locality accounting."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass(frozen=True)
class BlockSelection:
    indices: torch.Tensor
    valid_mask: torch.Tensor
    block_count: torch.Tensor
    loaded_count: torch.Tensor
    oracle_recall: torch.Tensor
    precision: torch.Tensor
    expansion: torch.Tensor


def block_scores(scores: torch.Tensor, block_size: int) -> tuple[torch.Tensor,int]:
    n=scores.shape[-1]; blocks=math.ceil(n/block_size); padded=blocks*block_size
    values=torch.nn.functional.pad(scores,(0,padded-n)).reshape(len(scores),blocks,block_size).sum(-1)
    return values,n


def select_blocks(scores: torch.Tensor, oracle_indices: torch.Tensor, block_size: int,
                  mode: str, budget_k: int, target_coverage: float) -> BlockSelection:
    """Select by contribution score under equal-budget or equal-coverage accounting."""
    values,n=block_scores(scores,block_size); block_order=torch.argsort(values,dim=-1,descending=True)
    oracle_mask=torch.zeros_like(scores,dtype=torch.bool).scatter_(1,oracle_indices,True)
    selected_rows=[]; counts=[]; recalls=[]; precisions=[]; expansions=[]; loaded_counts=[]
    for row in range(len(scores)):
        if mode=="equal_budget": take=min(values.shape[1],math.ceil(budget_k/block_size))
        elif mode=="equal_coverage":
            covered=torch.zeros(n,dtype=torch.bool,device=scores.device); take=0
            for block in block_order[row]:
                first=int(block)*block_size; covered[first:min(first+block_size,n)]=True; take+=1
                recall=(covered&oracle_mask[row]).sum()/oracle_indices.shape[1]
                if recall>=target_coverage: break
        else: raise ValueError("mode must be equal_budget or equal_coverage")
        blocks=block_order[row,:take]; mask=torch.zeros(n,dtype=torch.bool,device=scores.device)
        for block in blocks:
            first=int(block)*block_size; mask[first:min(first+block_size,n)]=True
        indices=torch.flatnonzero(mask); selected_rows.append(indices); loaded=len(indices)
        hits=(mask&oracle_mask[row]).sum().float(); recall=hits/oracle_indices.shape[1]
        counts.append(take); loaded_counts.append(loaded); recalls.append(recall)
        precisions.append(hits/max(loaded,1)); expansions.append(loaded/oracle_indices.shape[1])
    max_loaded=max(loaded_counts); padded_indices=torch.zeros((len(scores),max_loaded),dtype=torch.long,device=scores.device)
    valid=torch.zeros((len(scores),max_loaded),dtype=torch.bool,device=scores.device)
    for row,indices in enumerate(selected_rows): padded_indices[row,:len(indices)]=indices; valid[row,:len(indices)]=True
    # All rows normally have equal length in budget mode; callers use valid for partial final blocks.
    return BlockSelection(padded_indices,valid,torch.tensor(counts,device=scores.device),torch.tensor(loaded_counts,device=scores.device),
                          torch.stack(recalls),torch.stack(precisions),torch.tensor(expansions,device=scores.device))


def expand_block_indices(block_ids: np.ndarray, block_size: int, neurons: int) -> np.ndarray:
    values=[]
    for block in np.asarray(block_ids): values.extend(range(int(block)*block_size,min((int(block)+1)*block_size,neurons)))
    return np.asarray(values,dtype=np.int64)
