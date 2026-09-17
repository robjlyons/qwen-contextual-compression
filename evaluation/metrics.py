from __future__ import annotations
import torch
from torch import Tensor


def stable_cosine(a: Tensor, b: Tensor, dim: int = -1, eps: float = 1e-12,
                  dtype: torch.dtype = torch.float32) -> Tensor:
    """Cosine with stable accumulation, exact-equality handling, and valid bounds."""
    left, right = a.detach().to(dtype), b.detach().to(dtype)
    dot = (left * right).sum(dim=dim)
    denominator = (torch.linalg.vector_norm(left, dim=dim) *
                   torch.linalg.vector_norm(right, dim=dim)).clamp_min(eps)
    cosine = (dot / denominator).clamp(-1.0, 1.0)
    identical = (a == b).all(dim=dim)
    return torch.where(identical, torch.ones_like(cosine), cosine)


def output_metrics(dense: Tensor, sparse: Tensor, eps: float = 1e-12,
                   cosine_dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
    dense, sparse = dense.float(), sparse.float(); delta = dense - sparse
    return {"cosine_similarity": stable_cosine(dense, sparse, dim=-1, eps=eps, dtype=cosine_dtype),
            "relative_l2": torch.linalg.vector_norm(delta, dim=-1)/(torch.linalg.vector_norm(dense, dim=-1)+eps),
            "mse": delta.square().mean(-1), "max_absolute_error": delta.abs().amax(-1)}
