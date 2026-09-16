import torch
from torch import nn
from conftest import TinyFFN
from end_to_end.model_patcher import OracleMLPPatcher
from end_to_end.oracle_sparse_mlp import OracleSparseMLP

class Layer(nn.Module):
 def __init__(self):super().__init__();self.norm=nn.Identity();self.mlp=TinyFFN()
 def forward(self,x):return x+self.mlp(x)
class Stack(nn.Module):
 def __init__(self):super().__init__();self.layers=nn.ModuleList([Layer(),Layer()]);self.head=nn.Linear(4,6)
 def forward(self,x):
  for layer in self.layers:x=layer(x)
  return self.head(x)

def test_patcher_modifies_only_mlps_and_restore_recovers_modules():
 model=Stack();norms=[layer.norm for layer in model.layers];original=[layer.mlp for layer in model.layers];patcher=OracleMLPPatcher(model)
 assert all(isinstance(layer.mlp,OracleSparseMLP) for layer in model.layers);assert [layer.norm for layer in model.layers]==norms
 patcher.restore();assert [layer.mlp for layer in model.layers]==original

def test_full_wrapper_end_to_end_and_sparse_state_propagates():
 torch.manual_seed(3);model=Stack();x=torch.randn(2,4);patcher=OracleMLPPatcher(model);patcher.dense();dense=model(x);seen=[];handle=model.layers[1].register_forward_pre_hook(lambda _m,args:seen.append(args[0].detach().clone()))
 patcher.apply_schedule([.25,1.]);sparse=model(x);handle.remove();assert not torch.allclose(dense,sparse)
 first_input=x+model.layers[0].mlp(x);torch.testing.assert_close(seen[-1],first_input)
 patcher.apply_schedule([1.,1.]);torch.testing.assert_close(model(x),dense);patcher.restore()

