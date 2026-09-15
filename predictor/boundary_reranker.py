"""Locked-core boundary-swap selector with an x-only cascade API."""
from __future__ import annotations
import torch
from torch import nn
from predictor.candidate_reranker import retention_count, ste_topk_count


def boundary_counts(width, lock_retention, final_retention, candidate_retention):
    if not lock_retention < final_retention:
        raise ValueError("lock retention must be less than final retention")
    if not candidate_retention > final_retention:
        raise ValueError("candidate retention must be greater than final retention")
    locked=retention_count(width,lock_retention);final=retention_count(width,final_retention);candidate=retention_count(width,candidate_retention)
    if not locked < final < candidate:raise ValueError("rounded counts must satisfy locked < final < candidate")
    return locked,final,candidate


def stage1_boundary(stage1_scores,lock_retention,final_retention,candidate_retention):
    locked_k,final_k,candidate_k=boundary_counts(stage1_scores.shape[-1],lock_retention,final_retention,candidate_retention);ranked=torch.topk(stage1_scores,candidate_k,-1,sorted=True);return ranked.indices[...,:locked_k],ranked.indices[...,locked_k:],ranked.values[...,locked_k:],final_k-locked_k


class BoundarySwapReranker(nn.Module):
    """Learn an alpha-gated correction for Stage-1 boundary scores only."""
    def __init__(self,input_dim,neuron_count,rerank_dim=16,normalize_stage1_scores=True):
        super().__init__()
        if rerank_dim!=16:raise ValueError("Phase 4.5 uses rerank_dim=16 only")
        self.context=nn.Linear(input_dim,rerank_dim,bias=False);self.neuron_embeddings=nn.Embedding(neuron_count,rerank_dim);self.alpha=nn.Parameter(torch.zeros(()));self.normalize_stage1_scores=normalize_stage1_scores

    def normalized_base(self,scores):
        if not self.normalize_stage1_scores:return scores
        return (scores-scores.mean(-1,keepdim=True))/scores.std(-1,keepdim=True,unbiased=False).clamp_min(1e-6)

    def forward(self,x,boundary_ids,boundary_scores):
        query=torch.nn.functional.silu(self.context(x));delta=torch.einsum("bdr,br->bd",self.neuron_embeddings(boundary_ids),query);return self.normalized_base(boundary_scores)+self.alpha*delta


class BoundarySwapCascade(nn.Module):
    """Freeze Stage 1 and structurally preserve its locked core."""
    def __init__(self,stage1,reranker,lock_retention=.45,final_retention=.5,candidate_retention=.65):
        super().__init__();boundary_counts(reranker.neuron_embeddings.num_embeddings,lock_retention,final_retention,candidate_retention);self.stage1=stage1;self.reranker=reranker;self.lock_retention=lock_retention;self.final_retention=final_retention;self.candidate_retention=candidate_retention;stage1.eval()
        for parameter in stage1.parameters():parameter.requires_grad_(False)

    def state(self,x):
        with torch.no_grad():scores=self.stage1(x);return stage1_boundary(scores,self.lock_retention,self.final_retention,self.candidate_retention)

    def final_indices(self,x,ste=False,temperature=.5):
        locked,boundary,base,needed=self.state(x);scores=self.reranker(x,boundary,base)
        if not ste:return torch.cat((locked,boundary.gather(-1,torch.topk(scores,needed,-1,sorted=False).indices)),-1)
        mask=ste_topk_count(scores,needed,temperature);return locked,boundary,mask

    def forward(self,x):
        indices=self.final_indices(x);full=x.new_full((x.shape[0],self.reranker.neuron_embeddings.num_embeddings),-torch.inf);return full.scatter(-1,indices,0.)


def boundary_mask(locked_ids,boundary_ids,boundary_values,neuron_count):
    mask=boundary_values.new_zeros((boundary_values.shape[0],neuron_count));mask.scatter_(-1,locked_ids,1.);return mask.scatter(-1,boundary_ids,boundary_values)


def boundary_accounting(stage1_macs,stage1_parameters,input_dim,neuron_count,lock_retention,candidate_retention,rerank_dim,dense_ffn_macs):
    locked=retention_count(neuron_count,lock_retention);candidate=retention_count(neuron_count,candidate_retention);boundary=candidate-locked;stage2_macs=input_dim*rerank_dim+boundary*rerank_dim;embedding_parameters=neuron_count*rerank_dim;stage2_parameters=input_dim*rerank_dim+embedding_parameters+1;total_macs=stage1_macs+stage2_macs;total_parameters=stage1_parameters+stage2_parameters;return {"locked_count":locked,"candidate_count":candidate,"boundary_count":boundary,"stage1_macs":stage1_macs,"stage2_macs":stage2_macs,"total_selector_macs":total_macs,"stage1_mac_fraction":stage1_macs/dense_ffn_macs,"stage2_mac_fraction":stage2_macs/dense_ffn_macs,"total_mac_fraction":total_macs/dense_ffn_macs,"stage1_parameters":stage1_parameters,"stage2_parameters":stage2_parameters,"neuron_embedding_parameters":embedding_parameters,"total_parameters":total_parameters,"fp32_bytes":total_parameters*4,"fp16_bf16_bytes":total_parameters*2,"int8_bytes":total_parameters}
