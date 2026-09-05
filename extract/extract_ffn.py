"""Structural discovery and an explicit, independently evaluated gated FFN."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import torch
from torch import Tensor, nn


class FFNDiscoveryError(RuntimeError):
    """Raised when a compatible dense gated FFN cannot be located."""


@dataclass(frozen=True)
class LocatedFFN:
    path: str
    layers_path: str
    layer: nn.Module
    module: nn.Module
    gate_proj: nn.Module
    up_proj: nn.Module
    down_proj: nn.Module
    activation: Callable[[Tensor], Tensor]


def _activation(module: nn.Module) -> Callable[[Tensor], Tensor]:
    for name in ("act_fn", "activation_fn", "activation"):
        value = getattr(module, name, None)
        if callable(value):
            return value
    raise FFNDiscoveryError(f"{type(module).__name__} has projections but no callable activation")


def locate_ffn(model: nn.Module, layer_index: int) -> LocatedFFN:
    """Find a ModuleList whose requested layer contains gate/up/down projections."""
    candidates: list[tuple[str, nn.ModuleList, str, nn.Module]] = []
    for layers_path, layers in model.named_modules():
        if not isinstance(layers, nn.ModuleList) or not (0 <= layer_index < len(layers)):
            continue
        layer = layers[layer_index]
        for relative, module in layer.named_modules():
            if all(isinstance(getattr(module, n, None), nn.Module) for n in ("gate_proj", "up_proj", "down_proj")):
                candidates.append((layers_path, layers, relative, module))
    if not candidates:
        paths = [n for n, m in model.named_modules() if isinstance(m, nn.ModuleList)]
        raise FFNDiscoveryError(
            f"Could not locate a gated dense FFN in layer {layer_index}. "
            f"Inspected ModuleLists: {paths or '<none>'}. Expected gate_proj, up_proj, down_proj and act_fn."
        )
    # Prefer the shallowest FFN within the longest plausible language stack.
    candidates.sort(key=lambda x: (-len(x[1]), x[2].count("."), len(x[2])))
    layers_path, layers, relative, module = candidates[0]
    path = ".".join(p for p in (layers_path, str(layer_index), relative) if p)
    return LocatedFFN(path, layers_path, layers[layer_index], module, module.gate_proj,
                      module.up_proj, module.down_proj, _activation(module))


def gated_activations(x: Tensor, ffn: LocatedFFN) -> Tensor:
    return ffn.activation(ffn.gate_proj(x)) * ffn.up_proj(x)


def explicit_dense_ffn(x: Tensor, ffn: LocatedFFN) -> Tensor:
    """Reproduce the discovered implementation without calling its forward method."""
    return ffn.down_proj(gated_activations(x, ffn))


def projection_parameter_count(ffn: LocatedFFN) -> int:
    return sum(p.numel() for m in (ffn.gate_proj, ffn.up_proj, ffn.down_proj) for p in m.parameters())

