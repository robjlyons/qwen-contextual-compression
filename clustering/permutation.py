"""Exact gated-FFN neuron permutation utilities."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import numpy as np
import torch
from torch import nn


def validate_permutation(permutation, neurons: int | None = None) -> np.ndarray:
    perm=np.asarray(permutation,dtype=np.int64)
    n=len(perm) if neurons is None else neurons
    if perm.ndim!=1 or len(perm)!=n or not np.array_equal(np.sort(perm),np.arange(n)):
        raise ValueError(f"Expected a bijection of 0..{n-1}")
    return perm


def inverse_permutation(permutation) -> np.ndarray:
    perm=validate_permutation(permutation); inverse=np.empty_like(perm); inverse[perm]=np.arange(len(perm)); return inverse


def permute_ffn_(ffn: nn.Module, permutation) -> nn.Module:
    """Apply new-position -> old-neuron permutation consistently, including biases."""
    perm=torch.as_tensor(validate_permutation(permutation,ffn.gate_proj.weight.shape[0]),device=ffn.gate_proj.weight.device)
    with torch.no_grad():
        ffn.gate_proj.weight.copy_(ffn.gate_proj.weight.index_select(0,perm))
        ffn.up_proj.weight.copy_(ffn.up_proj.weight.index_select(0,perm))
        ffn.down_proj.weight.copy_(ffn.down_proj.weight.index_select(1,perm.to(ffn.down_proj.weight.device)))
        for projection in (ffn.gate_proj,ffn.up_proj):
            if projection.bias is not None: projection.bias.copy_(projection.bias.index_select(0,perm))
    return ffn


def permuted_ffn_copy(ffn: nn.Module, permutation) -> nn.Module:
    return permute_ffn_(copy.deepcopy(ffn),permutation)


def remap_old_ids(old_ids: np.ndarray, permutation) -> np.ndarray:
    """Convert original neuron IDs into positions in the permuted layout."""
    return inverse_permutation(permutation)[old_ids]

