"""A current-input-only oracle wrapper around an official gated FFN."""
from __future__ import annotations
import math
from typing import Callable
import torch
from torch import Tensor, nn
from evaluation.metrics import output_metrics


class OracleSparseMLP(nn.Module):
    """Select neurons using only ``forward(x)`` from the live sparse trajectory.

    There is intentionally no dense-reference argument or state.  A dense pass is
    used only to warm/precompute down-column norms for offloaded modules.
    """
    actual_oracle_executes_dense_gate_up = True
    runtime_speedup_claimed = False

    def __init__(self, original: nn.Module, layer_idx: int, retention: float = 1.0,
                 telemetry: Callable[[int, dict[str, Tensor]], None] | None = None):
        super().__init__(); self.original=original; self.layer_idx=layer_idx; self.retention=float(retention); self.enabled=False; self.telemetry=telemetry
        self.register_buffer("down_column_norms",None,persistent=False)
        # This hook is registered after Accelerate's alignment hook, so an
        # offloaded down weight is materialized before norms and masking are used.
        self._norm_hook=original.down_proj.register_forward_pre_hook(self._prepare_down)

    def _prepare_down(self,module,args):
        if self.down_column_norms is None:
            if module.weight.device.type=="meta": raise RuntimeError(f"Layer {self.layer_idx} down weight is still meta when norm precomputation runs")
            self.down_column_norms=torch.linalg.vector_norm(module.weight.detach().float(),dim=0).cpu()
        if not self.enabled:return None
        activation=args[0];norms=self.down_column_norms.to(device=activation.device,dtype=torch.float32);scores=activation.detach().abs().float()*norms;k=max(1,min(self.intermediate_size,math.ceil(self.retention*self.intermediate_size)));indices=torch.topk(scores,k,dim=-1,sorted=False).indices;self.last_selected_indices=indices.detach();sparse_activation=torch.zeros_like(activation).scatter_(-1,indices,activation.gather(-1,indices))
        if self.telemetry is not None:
            dense_same_input=torch.nn.functional.linear(activation,module.weight,module.bias);sparse_same_input=torch.nn.functional.linear(sparse_activation,module.weight,module.bias);metrics=output_metrics(dense_same_input.flatten(0,-2),sparse_same_input.flatten(0,-2));incoming=torch.linalg.vector_norm(self._current_input.float(),dim=-1).flatten();self.telemetry(self.layer_idx,{**metrics,"ffn_over_residual":torch.linalg.vector_norm(dense_same_input.float(),dim=-1).flatten()/incoming.clamp_min(1e-12),"ffn_error_over_residual":torch.linalg.vector_norm((dense_same_input-sparse_same_input).float(),dim=-1).flatten()/incoming.clamp_min(1e-12)})
        return (sparse_activation,)

    @property
    def intermediate_size(self)->int: return int(self.original.down_proj.weight.shape[1])

    def set_mode(self,enabled:bool,retention:float|None=None)->None:
        self.enabled=enabled
        if retention is not None:
            if not 0 < retention <= 1: raise ValueError("retention must lie in (0, 1]")
            self.retention=float(retention)

    def forward(self,x:Tensor)->Tensor:
        self._current_input=x
        # Always call the original module so official/Accelerate hooks remain in
        # control. Sparse mode changes only down_proj's activation argument.
        try:return self.original(x)
        finally:self._current_input=None

    def close(self)->None: self._norm_hook.remove()
