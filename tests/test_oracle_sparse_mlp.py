import inspect
import torch
from conftest import TinyFFN
from end_to_end.oracle_sparse_mlp import OracleSparseMLP,compute_down_column_norms_chunked
from oracle.importance import importance_scores

def _warm(wrapper,x):
 wrapper.set_mode(False);wrapper(x);wrapper.set_mode(True)

def test_full_retention_matches_original():
 torch.manual_seed(1);mlp=TinyFFN();wrapper=OracleSparseMLP(mlp,0,1.);x=torch.randn(3,4);wrapper.set_mode(False);dense=wrapper(x);wrapper.set_mode(True);torch.testing.assert_close(wrapper(x),dense)
 assert wrapper.norm_computation_count==0 and wrapper.down_column_norms is None

@torch.no_grad()
def test_chunked_norm_matches_fp32_reference_for_dtypes_and_chunks():
 torch.manual_seed(8);base=torch.randn(97,521)
 for dtype in (torch.float16,torch.bfloat16,torch.float32):
  weight=base.to(dtype);reference=torch.linalg.vector_norm(weight.float(),dim=0)
  for columns in (17,64,128,256,512,521):
   torch.testing.assert_close(compute_down_column_norms_chunked(weight,columns),reference,rtol=1e-6,atol=1e-6)

def test_norms_are_lazy_and_cached():
 mlp=TinyFFN();wrapper=OracleSparseMLP(mlp,0,1.,norm_chunk_columns=2);x=torch.randn(2,4);wrapper.set_mode(True,1.);wrapper(x)
 assert wrapper.norm_computation_count==0
 wrapper.set_mode(True,.75);wrapper(x);assert wrapper.norm_computation_count==1
 wrapper(x);assert wrapper.norm_computation_count==1

def test_accelerate_rewritten_forward_is_intercepted_after_materialisation_step():
 mlp=TinyFFN();down=mlp.down_proj;old_forward=down.forward;down._old_forward=old_forward;down._hf_hook=object();events=[]
 def accelerate_forward(*args,**kwargs):
  events.append("materialised");return down._old_forward(*args,**kwargs)
 down.forward=accelerate_forward;wrapper=OracleSparseMLP(mlp,0,.75,norm_chunk_columns=2);x=torch.randn(1,4);wrapper.set_mode(True);wrapper(x)
 assert events==["materialised"] and wrapper.norm_computation_count==1
 wrapper.close();assert down._old_forward is old_forward

def test_topk_and_column_norm_importance_match_existing_oracle():
 torch.manual_seed(2);mlp=TinyFFN();wrapper=OracleSparseMLP(mlp,0,.25);x=torch.randn(2,4);_warm(wrapper,x);wrapper(x);a=mlp.act_fn(mlp.gate_proj(x))*mlp.up_proj(x);expected=torch.topk(importance_scores(a,mlp.down_proj.weight,"weighted_activation"),2,dim=-1).indices
 assert all(set(left.tolist())==set(right.tolist()) for left,right in zip(wrapper.last_selected_indices,expected))

def test_selector_api_has_only_current_forward_input_and_responds_to_it():
 assert list(inspect.signature(OracleSparseMLP.forward).parameters)==["self","x"]
 mlp=TinyFFN();wrapper=OracleSparseMLP(mlp,0,.25);x=torch.randn(1,4);_warm(wrapper,x);wrapper(x);first=wrapper.last_selected_indices.clone();wrapper(-x*3);second=wrapper.last_selected_indices
 assert first.shape==second.shape
