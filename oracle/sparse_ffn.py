from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor


def reconstruct(activations: Tensor, down_weight: Tensor, indices: Tensor,
                bias: Tensor | None = None) -> Tensor:
    """Per-sample W[:, S] @ a[S], without constructing a large selected-weight tensor."""
    selected = torch.zeros_like(activations)
    selected.scatter_(1, indices, activations.gather(1, indices))
    return torch.nn.functional.linear(selected, down_weight, bias)


@dataclass(frozen=True)
class WeightAccounting:
    dense_parameters: int
    selected_parameters: int
    dense_projection_weights: int
    selected_projection_weights: int
    selected_ratio: float


def parameter_accounting(gate_weight: Tensor, up_weight: Tensor, down_weight: Tensor,
                         k: int, bias_parameters: int = 0) -> WeightAccounting:
    n = gate_weight.shape[0]
    if up_weight.shape[0] != n or down_weight.shape[1] != n or not 0 < k <= n:
        raise ValueError("incompatible SwiGLU projection dimensions or k")
    dense_w = gate_weight.numel() + up_weight.numel() + down_weight.numel()
    # k gate rows + k up rows + k down columns; biases are conservatively always accessed.
    selected_w = k * (gate_weight.shape[1] + up_weight.shape[1] + down_weight.shape[0])
    dense = dense_w + bias_parameters; selected = selected_w + bias_parameters
    return WeightAccounting(dense, selected, dense_w, selected_w, selected / dense)


def byte_estimates(parameters: int) -> dict[str, float]:
    return {name: parameters * bits / 8 for name, bits in
            {"bf16": 16, "fp16": 16, "fp8": 8, "int8": 8, "q4": 4, "q3": 3, "q2": 2}.items()}

