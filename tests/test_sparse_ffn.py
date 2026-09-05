import torch
from conftest import Model
from extract.extract_ffn import locate_ffn,explicit_dense_ffn,gated_activations
from oracle.sparse_ffn import reconstruct,parameter_accounting
from oracle.importance import topk_indices
def test_explicit_matches_original():
 m=Model(); x=torch.randn(5,4); f=locate_ffn(m,0)
 assert f.path=="backbone.layers.0.mlp"; assert torch.allclose(m(x),explicit_dense_ffn(x,f))
def test_full_and_partial_sparse():
 m=Model(); f=locate_ffn(m,0); x=torch.randn(3,4); a=gated_activations(x,f)
 full=reconstruct(a,f.down_proj.weight,topk_indices(a.abs(),8),f.down_proj.bias)
 assert full.shape==(3,4) and torch.allclose(full,explicit_dense_ffn(x,f),atol=1e-6)
 assert reconstruct(a,f.down_proj.weight,topk_indices(a.abs(),3)).shape==(3,4)
def test_parameter_accounting():
 m=Model(); f=locate_ffn(m,0); q=parameter_accounting(f.gate_proj.weight,f.up_proj.weight,f.down_proj.weight,2)
 assert q.dense_projection_weights==96 and q.selected_projection_weights==24 and q.selected_ratio==.25
def test_hook_captures_input_not_output():
 m=Model(); seen=[]; h=m.backbone.layers[0].mlp.register_forward_pre_hook(lambda mod,args:seen.append(args[0].clone()))
 x=torch.randn(2,4); y=m(x); h.remove(); assert torch.equal(seen[0],x) and seen[0].shape!=y.unsqueeze(0).shape

