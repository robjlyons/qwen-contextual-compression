"""Frozen, checkpoint-normalized x-only runtime selector."""
from __future__ import annotations

import inspect
from pathlib import Path

import torch
from torch import nn

from predictor.candidate_reranker import retention_count
from predictor.models import create_predictor


class RuntimeSelector(nn.Module):
    def __init__(self,predictor,mean,std,retention=.5):
        super().__init__();self.predictor=predictor.eval();self.register_buffer("mean",mean.float());self.register_buffer("std",std.float().clamp_min(1e-6));self.retention=retention
        for parameter in predictor.parameters():parameter.requires_grad_(False)
    @classmethod
    def from_checkpoint(cls,path:Path,retention=.5,device="cpu"):
        checkpoint=torch.load(path,map_location="cpu",weights_only=False);config=checkpoint["config"];model=create_predictor(config["kind"],config["input_dim"],config["output_dim"],config["latent_dim"]);model.load_state_dict(checkpoint["model"],strict=True);return cls(model,checkpoint["mean"],checkpoint["std"],retention).to(device)
    def normalize(self,x):return (x.float()-self.mean)/self.std
    def scores(self,x):return self.predictor(self.normalize(x))
    def select(self,x):return torch.topk(self.scores(x),retention_count(self.predictor.neurons.out_features,self.retention),-1,sorted=False).indices


def assert_x_only_selector(selector):
    if list(inspect.signature(selector.select).parameters) != ["x"]:raise RuntimeError("Runtime selector inference API must accept only x")
