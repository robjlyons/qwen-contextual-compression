import torch
from torch import nn
from extract.extract_ffn import locate_ffn

class TinyFFN(nn.Module):
 def __init__(self,h=4,n=8,bias=False):
  super().__init__(); self.gate_proj=nn.Linear(h,n,bias=bias); self.up_proj=nn.Linear(h,n,bias=bias); self.down_proj=nn.Linear(n,h,bias=bias); self.act_fn=nn.SiLU()
 def forward(self,x): return self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))
class Layer(nn.Module):
 def __init__(self): super().__init__(); self.mlp=TinyFFN()
class Model(nn.Module):
 def __init__(self): super().__init__(); self.backbone=nn.Module(); self.backbone.layers=nn.ModuleList([Layer()])
 def forward(self,x): return self.backbone.layers[0].mlp(x)

