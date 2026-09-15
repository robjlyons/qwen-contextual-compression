"""Frozen, checkpoint-normalized x-only runtime selector."""
from __future__ import annotations

import inspect
import copy
from pathlib import Path

import torch
from torch import nn

from predictor.candidate_reranker import retention_count
from predictor.models import create_predictor


class RuntimeSelector(nn.Module):
    def __init__(self,predictor,mean,std,retention=.5):
        super().__init__();self.predictor=predictor.eval();self.register_buffer("mean",mean);self.register_buffer("std",std.clamp_min(1e-6));self.retention=retention
        for parameter in predictor.parameters():parameter.requires_grad_(False)
    @classmethod
    def from_checkpoint(cls,path:Path,retention=.5,device="cpu"):
        checkpoint=torch.load(path,map_location="cpu",weights_only=False);config=checkpoint["config"];model=create_predictor(config["kind"],config["input_dim"],config["output_dim"],config["latent_dim"]);model.load_state_dict(checkpoint["model"],strict=True);return cls(model,checkpoint["mean"].float(),checkpoint["std"].float(),retention).to(device)
    def normalize(self,x):return (x.float()-self.mean)/self.std
    def scores(self,x):return self.predictor(self.normalize(x))
    def select(self,x):return torch.topk(self.scores(x),retention_count(self.predictor.neurons.out_features,self.retention),-1,sorted=False).indices


def assert_x_only_selector(selector):
    if list(inspect.signature(selector.select).parameters) != ["x"]:raise RuntimeError("Runtime selector inference API must accept only x")


class FoldedRuntimeSelector(RuntimeSelector):
    """Inference copy with normalization folded into the first Linear layer."""
    def __init__(self,predictor,mean,std,retention=.5,dtype=torch.float32):
        predictor=copy.deepcopy(predictor).eval();std=std.float().clamp_min(1e-6);mean=mean.float();source=predictor.encoder;folded=nn.Linear(source.in_features,source.out_features,bias=True,dtype=dtype);weight=source.weight.detach().float()/std[None,:];bias=-(weight@mean)
        with torch.no_grad():folded.weight.copy_(weight.to(dtype));folded.bias.copy_(bias.to(dtype))
        predictor.encoder=folded;predictor.to(dtype=dtype);nn.Module.__init__(self);self.predictor=predictor;self.register_buffer("mean",mean.new_zeros(mean.shape,dtype=dtype));self.register_buffer("std",std.new_ones(std.shape,dtype=dtype));self.retention=retention;self.inference_dtype=dtype
        for parameter in predictor.parameters():parameter.requires_grad_(False)
    def normalize(self,x):return x.to(self.inference_dtype)


class CastRuntimeSelector(RuntimeSelector):
    def __init__(self,predictor,mean,std,retention,dtype):super().__init__(predictor,mean.to(dtype),std.to(dtype),retention);self.inference_dtype=dtype
    def normalize(self,x):return (x.to(self.inference_dtype)-self.mean)/self.std


def selector_variants(selector):
    """Return independent variants; never mutates checkpoint-owned parameters."""
    return {"baseline_fp32":selector,"folded_norm_fp32":FoldedRuntimeSelector(selector.predictor,selector.mean,selector.std,selector.retention,torch.float32),"fp16_predictor":CastRuntimeSelector(copy.deepcopy(selector.predictor).half(),selector.mean,selector.std,selector.retention,torch.float16),"fp16_folded_norm":FoldedRuntimeSelector(selector.predictor,selector.mean,selector.std,selector.retention,torch.float16)}


def selected_set_comparison(reference,candidate,width):
    ref=torch.zeros((len(reference),width),device=reference.device,dtype=torch.bool).scatter_(-1,reference,True);other=torch.zeros_like(ref).scatter_(-1,candidate,True);intersection=(ref&other).sum(-1).float();changed=(ref^other).sum(-1).float();return {"set_equality_fraction":float((changed==0).float().mean()),"jaccard":float((intersection/(ref|other).sum(-1)).mean()),"mean_changed_neurons":float(changed.mean()),"p95_changed_neurons":float(changed.quantile(.95))}
