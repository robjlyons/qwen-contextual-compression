from __future__ import annotations
import torch
from torch import Tensor


def importance_scores(activations: Tensor, down_weight: Tensor, metric: str) -> Tensor:
    """Score intermediate neurons; last dimensions must agree."""
    if activations.shape[-1] != down_weight.shape[1]:
        raise ValueError("activation width must equal down-projection input width")
    if metric in {"activation", "activation_magnitude"}: return activations.abs()
    norms = torch.linalg.vector_norm(down_weight.float(), dim=0).to(activations.device)
    if metric in {"weighted_activation", "weighted_contribution"}:
        return activations.abs().float() * norms
    if metric in {"exact", "exact_output_norm"}:
        # ||a_j W[:,j]||_2 = sqrt(a_j^2 * sum_i W[i,j]^2), evaluated without a B*N*H temporary.
        squared = activations.float().square() * down_weight.float().square().sum(dim=0)
        return squared.sqrt()
    raise ValueError(f"Unknown importance metric {metric!r}")


def topk_indices(scores: Tensor, k: int) -> Tensor:
    if not 0 < k <= scores.shape[-1]: raise ValueError(f"k must be in [1, {scores.shape[-1]}]")
    return torch.topk(scores, k, dim=-1, sorted=True).indices


def block_topk_indices(scores: Tensor, k: int, block_size: int) -> Tensor:
    """Return complete, non-overlapping blocks, using summed neuron score."""
    if block_size <= 0: raise ValueError("block_size must be positive")
    n = scores.shape[-1]; padded = ((n + block_size - 1) // block_size) * block_size
    block_scores = torch.nn.functional.pad(scores, (0, padded-n)).reshape(*scores.shape[:-1], -1, block_size).sum(-1)
    blocks = max(1, min(block_scores.shape[-1], (k + block_size - 1) // block_size))
    selected = torch.topk(block_scores, blocks, dim=-1).indices
    offsets = torch.arange(block_size, device=scores.device)
    indices = (selected.unsqueeze(-1) * block_size + offsets).flatten(-2)
    return indices.masked_fill(indices >= n, n).sort(dim=-1).values[..., :min(blocks*block_size, n)]


def random_indices(batch: int, n: int, k: int, seed: int, device=None) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.stack([torch.randperm(n, generator=generator)[:k] for _ in range(batch)]).to(device)
