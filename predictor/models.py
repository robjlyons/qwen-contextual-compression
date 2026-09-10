import torch
from torch import nn
class FactorizedPredictor(nn.Module):
 def __init__(self,input_dim,output_dim,latent_dim=128,dropout=0.):super().__init__();self.encoder=nn.Linear(input_dim,latent_dim,bias=False);self.dropout=nn.Dropout(dropout);self.neurons=nn.Linear(latent_dim,output_dim)
 def forward(self,x):return self.neurons(self.dropout(self.encoder(x)))
class LowRankMLP(nn.Module):
 def __init__(self,input_dim,output_dim,latent_dim=256,dropout=0.,activation="silu"):
  super().__init__();act=nn.SiLU if activation=="silu" else nn.GELU;self.net=nn.Sequential(nn.Linear(input_dim,latent_dim),act(),nn.Dropout(dropout),nn.Linear(latent_dim,latent_dim),act(),nn.Linear(latent_dim,output_dim))
 def forward(self,x):return self.net(x)
class StaticHot:
 def fit(self,scores):self.scores=scores.mean(0);return self
 def __call__(self,x):return self.scores.expand(*x.shape[:-1],-1)
