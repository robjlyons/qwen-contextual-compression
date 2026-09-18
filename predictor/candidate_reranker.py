"""Two-stage candidate-only reranking with no FFN information at inference."""
from __future__ import annotations
import math
import torch
from torch import nn
from predictor.output_aware import _StraightThroughMask


def retention_count(width: int, retention: float) -> int:
    if not 0 < retention <= 1:
        raise ValueError("retention must be in (0, 1]")
    return max(1, math.ceil(width * retention))


def candidate_topk(scores, retention):
    return torch.topk(scores, retention_count(scores.shape[-1], retention), dim=-1, sorted=False).indices


def hard_topk_count(scores, k):
    if not 0 < k <= scores.shape[-1]:
        raise ValueError("final k must fit inside the candidate pool")
    ids = torch.topk(scores, k, dim=-1, sorted=False).indices
    return torch.zeros_like(scores).scatter_(-1, ids, 1)


def ste_topk_count(scores, k, temperature=.5):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    hard = hard_topk_count(scores, k)
    threshold = torch.topk(scores, k, dim=-1, sorted=True).values[..., -1:].detach()
    soft = torch.sigmoid((scores - threshold) / temperature)
    return _StraightThroughMask.apply(hard, soft)


class CandidateReranker(nn.Module):
    """Score only Stage-1 candidates from x, their IDs, and their Stage-1 scores."""
    def __init__(self, input_dim, neuron_count, rerank_dim=16, normalize_stage1_scores=True):
        super().__init__()
        if rerank_dim not in (16, 32):
            raise ValueError("rerank_dim must be 16 or 32 in Phase 4.4")
        self.context = nn.Linear(input_dim, rerank_dim, bias=False)
        self.neuron_embeddings = nn.Embedding(neuron_count, rerank_dim)
        self.normalize_stage1_scores = normalize_stage1_scores
        nn.init.zeros_(self.neuron_embeddings.weight)

    def normalized_base(self, candidate_scores):
        if not self.normalize_stage1_scores:
            return candidate_scores
        return (candidate_scores - candidate_scores.mean(-1, keepdim=True)) / candidate_scores.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)

    def forward(self, x, candidate_ids, candidate_scores):
        query = torch.nn.functional.silu(self.context(x))
        embeddings = self.neuron_embeddings(candidate_ids)
        delta = torch.einsum("bcr,br->bc", embeddings, query)
        return self.normalized_base(candidate_scores) + delta


class CandidateCascade(nn.Module):
    """Inference facade: callers supply only normalized pre-FFN hidden state x."""
    def __init__(self, stage1, reranker, candidate_retention=.6, final_retention=.5):
        super().__init__()
        if candidate_retention <= final_retention:
            raise ValueError("candidate retention must be greater than final retention")
        self.stage1, self.reranker = stage1, reranker
        self.candidate_retention, self.final_retention = candidate_retention, final_retention
        self.stage1.eval()
        for parameter in self.stage1.parameters():
            parameter.requires_grad_(False)

    def candidate_state(self, x):
        with torch.no_grad():
            stage1_scores = self.stage1(x)
            ids = candidate_topk(stage1_scores, self.candidate_retention)
            scores = stage1_scores.gather(-1, ids)
        return ids, scores

    def forward(self, x):
        ids, base = self.candidate_state(x)
        reranked = self.reranker(x, ids, base)
        full = reranked.new_full((x.shape[0], self.reranker.neuron_embeddings.num_embeddings), -torch.inf)
        return full.scatter(-1, ids, reranked)

    def final_indices(self, x):
        scores = self(x)
        return torch.topk(scores, retention_count(scores.shape[-1], self.final_retention), -1, sorted=False).indices


def candidate_mask(candidate_ids, candidate_mask_values, neuron_count):
    return candidate_mask_values.new_zeros((*candidate_mask_values.shape[:-1], neuron_count)).scatter(-1, candidate_ids, candidate_mask_values)


def reranker_accounting(stage1_macs, stage1_parameters, input_dim, neuron_count, candidate_retention, rerank_dim, dense_ffn_macs):
    candidates = retention_count(neuron_count, candidate_retention)
    stage2_macs = input_dim * rerank_dim + candidates * rerank_dim
    stage2_parameters = input_dim * rerank_dim + neuron_count * rerank_dim
    total_macs, total_parameters = stage1_macs + stage2_macs, stage1_parameters + stage2_parameters
    return {"candidate_count": candidates, "stage1_macs": stage1_macs, "stage2_macs": stage2_macs, "total_selector_macs": total_macs, "stage1_mac_fraction": stage1_macs/dense_ffn_macs, "stage2_mac_fraction": stage2_macs/dense_ffn_macs, "total_mac_fraction": total_macs/dense_ffn_macs, "stage1_parameters": stage1_parameters, "stage2_parameters": stage2_parameters, "total_parameters": total_parameters, "fp32_bytes": total_parameters*4, "fp16_bf16_bytes": total_parameters*2, "int8_bytes": total_parameters}
