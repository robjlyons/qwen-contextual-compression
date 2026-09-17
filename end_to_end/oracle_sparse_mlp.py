"""A current-input-only oracle wrapper around an official gated FFN."""
from __future__ import annotations
import math
from typing import Callable
import torch
from torch import Tensor, nn
from evaluation.metrics import output_metrics


@torch.no_grad()
def compute_down_column_norms_chunked(weight: Tensor, chunk_columns: int = 256) -> Tensor:
    """Compute exact L2 column norms with bounded FP32 temporary storage.

    Only ``out_features * chunk_columns`` values are converted to FP32 at once;
    the complete projection is never copied or moved between devices.
    """
    if weight.ndim != 2:
        raise ValueError("down projection weight must be a matrix")
    if chunk_columns <= 0:
        raise ValueError("chunk_columns must be positive")
    if weight.device.type == "meta":
        raise RuntimeError("down projection is still meta; Accelerate did not materialize it before the pre-hook")
    width = weight.shape[1]
    norms = torch.empty(width, dtype=torch.float32, device="cpu")
    for start in range(0, width, chunk_columns):
        end = min(start + chunk_columns, width)
        chunk = weight.detach()[:, start:end]
        chunk32 = chunk.to(dtype=torch.float32)
        chunk_norms = torch.linalg.vector_norm(chunk32, ord=2, dim=0)
        norms[start:end].copy_(chunk_norms.to(device="cpu"))
        del chunk_norms, chunk32, chunk
    return norms


class OracleSparseMLP(nn.Module):
    """Select neurons using only ``forward(x)`` from the live sparse trajectory.

    There is intentionally no dense-reference argument or state. Norms are
    prepared only on the first genuinely sparse down-projection execution.
    """
    actual_oracle_executes_dense_gate_up = True
    runtime_speedup_claimed = False

    def __init__(self, original: nn.Module, layer_idx: int, retention: float = 1.0,
                 telemetry: Callable[[int, dict[str, Tensor]], None] | None = None,
                 norm_chunk_columns: int = 256):
        super().__init__(); self.original=original; self.layer_idx=layer_idx; self.retention=float(retention); self.enabled=False; self.telemetry=telemetry
        if norm_chunk_columns <= 0: raise ValueError("norm_chunk_columns must be positive")
        self.norm_chunk_columns=norm_chunk_columns;self.norm_computation_count=0;self.norm_preparation_log=None
        self.register_buffer("down_column_norms",None,persistent=False)
        down=original.down_proj;self._accelerate_old_forward=None
        if hasattr(down,"_hf_hook") and hasattr(down,"_old_forward"):
            # Accelerate materialises offloaded/meta weights inside its rewritten
            # forward, after ordinary PyTorch pre-hooks. Intercept _old_forward so
            # norm preparation sees the materialised weight, without moving or
            # duplicating the full projection ourselves.
            self._accelerate_old_forward=down._old_forward
            def intercepted(*args,**kwargs):
                replacement=self._prepare_down(down,args)
                return self._accelerate_old_forward(*(replacement or args),**kwargs)
            self._accelerate_interceptor=intercepted;down._old_forward=intercepted;self._norm_hook=None
        else:self._norm_hook=down.register_forward_pre_hook(self._prepare_down)

    def _prepare_down(self,module,args):
        # Dense/100% execution is a true fast path: no norms, ranking, top-k, or
        # sparse activation are created. This ordering fixes the original OOM.
        if not self.enabled or self.retention >= 1.0:return None
        if self.down_column_norms is None:
            weight=module.weight
            self.down_column_norms=compute_down_column_norms_chunked(weight,self.norm_chunk_columns)
            self.norm_computation_count+=1
            temporary_mib=weight.shape[0]*min(weight.shape[1],self.norm_chunk_columns)*4/2**20
            self.norm_preparation_log={"layer":self.layer_idx,"down_shape":list(weight.shape),"weight_dtype":str(weight.dtype),"weight_device":str(weight.device),"norm_chunk_columns":self.norm_chunk_columns,"estimated_fp32_temp_mib":temporary_mib,"cached_norm_bytes":self.down_column_norms.numel()*self.down_column_norms.element_size()}
            print("Down-column norm preparation: "+", ".join(f"{key}={value}" for key,value in self.norm_preparation_log.items()))
        activation=args[0];norms=self.down_column_norms.to(device=activation.device,dtype=torch.float32,non_blocking=True);scores=activation.detach().abs().float()*norms;k=max(1,min(self.intermediate_size,math.ceil(self.retention*self.intermediate_size)));indices=torch.topk(scores,k,dim=-1,sorted=False).indices;self.last_selected_indices=indices.detach();sparse_activation=torch.zeros_like(activation).scatter_(-1,indices,activation.gather(-1,indices))
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

    def close(self)->None:
        if self._norm_hook is not None:self._norm_hook.remove()
        if self._accelerate_old_forward is not None and self.original.down_proj._old_forward is self._accelerate_interceptor:self.original.down_proj._old_forward=self._accelerate_old_forward
