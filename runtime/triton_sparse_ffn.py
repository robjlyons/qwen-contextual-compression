"""Correctness-first materialization-free Triton indexed FFN backend."""
from __future__ import annotations
import importlib.util

import torch

TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    @triton.jit
    def _indexed_gate_up(x,wg,wu,ids,activation,H:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
        k=tl.program_id(0);neuron=tl.load(ids+k);acc_g=0.0;acc_u=0.0
        for start in range(0,H,BLOCK):
            offsets=start+tl.arange(0,BLOCK);mask=offsets<H;xv=tl.load(x+offsets,mask=mask,other=0.).to(tl.float32);acc_g+=tl.sum(xv*tl.load(wg+neuron*H+offsets,mask=mask,other=0.).to(tl.float32));acc_u+=tl.sum(xv*tl.load(wu+neuron*H+offsets,mask=mask,other=0.).to(tl.float32))
        silu=acc_g*tl.sigmoid(acc_g);tl.store(activation+k,silu*acc_u)

    @triton.jit
    def _indexed_down(activation,wd,ids,out,H:tl.constexpr,I:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
        h=tl.program_id(0);acc=0.0
        for start in range(0,K,BLOCK):
            offsets=start+tl.arange(0,BLOCK);mask=offsets<K;neurons=tl.load(ids+offsets,mask=mask,other=0);a=tl.load(activation+offsets,mask=mask,other=0.).to(tl.float32);w=tl.load(wd+h*I+neurons,mask=mask,other=0.).to(tl.float32);acc+=tl.sum(a*w)
        tl.store(out+h,acc)


def triton_indexed_ffn(x,selected_ids,gate_weight,up_weight,down_weight):
    """Indirectly read selected weights; never creates KxH gathered matrices."""
    if not TRITON_AVAILABLE:raise RuntimeError("Triton is unavailable; PyTorch backends remain usable")
    if not x.is_cuda:raise ValueError("triton_indexed requires CUDA tensors")
    if x.ndim!=2 or x.shape[0]!=1 or selected_ids.ndim!=1:raise ValueError("triton_indexed currently targets batch=1")
    hidden=x.shape[-1];intermediate=gate_weight.shape[0];k=selected_ids.numel();activation=torch.empty(k,device=x.device,dtype=x.dtype);output=torch.empty((1,hidden),device=x.device,dtype=x.dtype);_indexed_gate_up[(k,)](x,gate_weight,up_weight,selected_ids,activation,H=hidden,K=k,BLOCK=256);_indexed_down[(hidden,)](activation,down_weight,selected_ids,output,H=hidden,I=intermediate,K=k,BLOCK=256);return output
