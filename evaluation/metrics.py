from __future__ import annotations
import torch
from torch import Tensor


def output_metrics(dense: Tensor, sparse: Tensor, eps: float = 1e-12) -> dict[str, Tensor]:
    dense, sparse = dense.float(), sparse.float(); delta = dense - sparse
    return {"cosine_similarity": torch.nn.functional.cosine_similarity(dense, sparse, dim=-1, eps=eps),
            "relative_l2": torch.linalg.vector_norm(delta, dim=-1)/(torch.linalg.vector_norm(dense, dim=-1)+eps),
            "mse": delta.square().mean(-1), "max_absolute_error": delta.abs().amax(-1)}

