"""Numerically stable compact token metrics for vocabulary-wide logits."""
from __future__ import annotations
import torch
from torch import Tensor


def _logit_metrics_chunk(dense:Tensor,sparse:Tensor)->dict[str,Tensor]:
    d=dense.float();s=sparse.float();delta=d-s
    log_pd=torch.log_softmax(d,-1);log_ps=torch.log_softmax(s,-1);pd=log_pd.exp();ps=log_ps.exp();m=.5*(pd+ps);log_m=torch.log(m.clamp_min(1e-30))
    dense_top=torch.topk(d,10,dim=-1).indices;sparse_top=torch.topk(s,10,dim=-1).indices
    overlap5=(dense_top[...,:5].unsqueeze(-1)==sparse_top[...,:5].unsqueeze(-2)).any(-1).sum(-1).float()/5
    overlap10=(dense_top.unsqueeze(-1)==sparse_top.unsqueeze(-2)).any(-1).sum(-1).float()/10
    return {"logit_cosine":torch.nn.functional.cosine_similarity(d,s,dim=-1),"logit_relative_l2":torch.linalg.vector_norm(delta,dim=-1)/torch.linalg.vector_norm(d,dim=-1).clamp_min(1e-12),"logit_mse":delta.square().mean(-1),"max_absolute_logit_difference":delta.abs().amax(-1),
      "kl_dense_sparse":(pd*(log_pd-log_ps)).sum(-1),"kl_sparse_dense":(ps*(log_ps-log_pd)).sum(-1),"js_divergence":.5*((pd*(log_pd-log_m)).sum(-1)+(ps*(log_ps-log_m)).sum(-1)),
      "top1_agreement":(dense_top[...,0]==sparse_top[...,0]).float(),"dense_top1_in_sparse_top5":(dense_top[...,0,None]==sparse_top[...,:5]).any(-1).float(),"sparse_top1_in_dense_top5":(sparse_top[...,0,None]==dense_top[...,:5]).any(-1).float(),"top5_set_overlap":overlap5,"top10_set_overlap":overlap10,"dense_margin":torch.topk(d,2,dim=-1).values.diff(dim=-1).neg().squeeze(-1)}


def logit_metrics(dense:Tensor,sparse:Tensor,token_chunk_size:int=8)->dict[str,Tensor]:
    """Compute vocab-wide metrics in bounded token chunks.

    Qwen's vocabulary is large; chunking prevents simultaneous float32 softmax,
    log-softmax, probability, and mixture tensors for every prompt token.
    """
    dense=dense.reshape(-1,dense.shape[-1]);sparse=sparse.reshape(-1,sparse.shape[-1]);parts={}
    for start in range(0,len(dense),token_chunk_size):
        chunk=_logit_metrics_chunk(dense[start:start+token_chunk_size],sparse[start:start+token_chunk_size])
        for key,value in chunk.items():parts.setdefault(key,[]).append(value)
    return {key:torch.cat(value) for key,value in parts.items()}


def token_nll(logits:Tensor,labels:Tensor)->Tensor:
    return torch.nn.functional.cross_entropy(logits.float(),labels,reduction="none")
