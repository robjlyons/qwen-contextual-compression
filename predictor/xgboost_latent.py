import torch
class TargetSVD:
 def __init__(self,rank=64):self.rank=rank
 def fit(self,train_targets):
  self.mean=train_targets.mean(0);_,_,v=torch.pca_lowrank(train_targets-self.mean,q=self.rank);self.components=v.T;return self
 def encode(self,target):return (target-self.mean)@self.components.T
 def decode(self,latent):return latent@self.components+self.mean
def fit_xgboost_latent(train_x,train_targets,rank=64,**kwargs):
 from xgboost import XGBRegressor
 from sklearn.multioutput import MultiOutputRegressor
 codec=TargetSVD(rank).fit(train_targets);model=MultiOutputRegressor(XGBRegressor(**kwargs)).fit(train_x.cpu().numpy(),codec.encode(train_targets).cpu().numpy());return model,codec
