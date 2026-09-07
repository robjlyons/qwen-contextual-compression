import inspect
import torch
from conftest import TinyFFN
from end_to_end.oracle_sparse_mlp import OracleSparseMLP
from oracle.importance import importance_scores

def _warm(wrapper,x):
 wrapper.set_mode(False);wrapper(x);wrapper.set_mode(True)

def test_full_retention_matches_original():
 torch.manual_seed(1);mlp=TinyFFN();wrapper=OracleSparseMLP(mlp,0,1.);x=torch.randn(3,4);wrapper.set_mode(False);dense=wrapper(x);wrapper.set_mode(True);torch.testing.assert_close(wrapper(x),dense)

def test_topk_and_column_norm_importance_match_existing_oracle():
 torch.manual_seed(2);mlp=TinyFFN();wrapper=OracleSparseMLP(mlp,0,.25);x=torch.randn(2,4);_warm(wrapper,x);wrapper(x);a=mlp.act_fn(mlp.gate_proj(x))*mlp.up_proj(x);expected=torch.topk(importance_scores(a,mlp.down_proj.weight,"weighted_activation"),2,dim=-1).indices
 assert all(set(left.tolist())==set(right.tolist()) for left,right in zip(wrapper.last_selected_indices,expected))

def test_selector_api_has_only_current_forward_input_and_responds_to_it():
 assert list(inspect.signature(OracleSparseMLP.forward).parameters)==["self","x"]
 mlp=TinyFFN();wrapper=OracleSparseMLP(mlp,0,.25);x=torch.randn(1,4);_warm(wrapper,x);wrapper(x);first=wrapper.last_selected_indices.clone();wrapper(-x*3);second=wrapper.last_selected_indices
 assert first.shape==second.shape

