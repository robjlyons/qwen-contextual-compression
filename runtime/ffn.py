"""Dense and truly pre-gate-selected PyTorch FFN implementations."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from runtime.weights import validate_weight_shapes


class DenseFFN(nn.Module):
    def __init__(self, gate_weight, up_weight, down_weight):
        super().__init__();validate_weight_shapes({"gate_proj.weight":gate_weight,"up_proj.weight":up_weight,"down_proj.weight":down_weight});self.gate_weight=gate_weight;self.up_weight=up_weight;self.down_weight=down_weight
    def components(self, x):
        gate=F.linear(x,self.gate_weight);up=F.linear(x,self.up_weight);activation=F.silu(gate)*up;return gate,up,activation
    def forward(self,x):
        return F.linear(self.components(x)[2],self.down_weight)


def gather_weights(gate_weight,up_weight,down_weight,selected_ids):
    """Materialize only the selected rows/columns (reference backend)."""
    if selected_ids.ndim != 1:raise ValueError("batch-1 sparse runtime requires one-dimensional selected_ids")
    return gate_weight.index_select(0,selected_ids),up_weight.index_select(0,selected_ids),down_weight.index_select(1,selected_ids)


class StaticPackedFFN(nn.Module):
    """Prepacked fixed-mask runtime ceiling; not valid dynamic inference."""
    def __init__(self,gate_selected,up_selected,down_selected):super().__init__();self.gate=gate_selected;self.up=up_selected;self.down=down_selected
    def components(self,x):
        gate=F.linear(x,self.gate);up=F.linear(x,self.up);activation=F.silu(gate)*up;return gate,up,activation
    def forward(self,x):return F.linear(self.components(x)[2],self.down)


class TorchDynamicSparseFFN(nn.Module):
    """Exact dynamic-gather path that selects before gate/up evaluation."""
    def __init__(self,gate_weight,up_weight,down_weight):super().__init__();validate_weight_shapes({"gate_proj.weight":gate_weight,"up_proj.weight":up_weight,"down_proj.weight":down_weight});self.gate=gate_weight;self.up=up_weight;self.down=down_weight
    def pack(self,selected_ids):return gather_weights(self.gate,self.up,self.down,selected_ids)
    def components(self,x,selected_ids):return StaticPackedFFN(*self.pack(selected_ids)).components(x)
    def forward(self,x,selected_ids):return StaticPackedFFN(*self.pack(selected_ids))(x)


def masked_dense_reference(x,selected_ids,gate_weight,up_weight,down_weight):
    gate=F.linear(x,gate_weight);up=F.linear(x,up_weight);activation=F.silu(gate)*up;mask=torch.zeros_like(activation).scatter_(-1,selected_ids.expand(x.shape[0],-1),1);return F.linear(activation*mask,down_weight)

def neuron_major_down(activation,selected_ids,down_t):
    """Reference for persistent neuron-major [I,H] down storage."""
    if down_t.ndim!=2:raise ValueError("down_t must be [intermediate, hidden]")
    return activation@down_t.index_select(0,selected_ids)
