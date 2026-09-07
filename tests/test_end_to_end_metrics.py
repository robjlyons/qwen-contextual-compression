import torch
from end_to_end.hidden_state_metrics import tensor_metrics
from end_to_end.streaming_metrics import logit_metrics,token_nll
from end_to_end.evaluation_runner import identity_metrics,identity_validation_checks,should_run_schedule
from evaluation.metrics import output_metrics,stable_cosine

def test_top_token_metrics_and_stable_kl():
 dense=torch.tensor([[10.,9.,8.,7.,6.,5.,4.,3.,2.,1.]])
 same=logit_metrics(dense,dense);assert same["top1_agreement"].item()==1 and same["dense_top1_in_sparse_top5"].item()==1
 assert same["top5_set_overlap"].item()==1 and same["top10_set_overlap"].item()==1 and torch.isfinite(same["kl_dense_sparse"]).all() and same["kl_dense_sparse"].abs().max()<1e-6

def test_top1_disagreement_but_top5_containment():
 dense=torch.tensor([[10.,9.,8.,7.,6.,5.,4.,3.,2.,1.]]);sparse=dense.clone();sparse[0,1]=11
 metrics=logit_metrics(dense,sparse);assert metrics["top1_agreement"].item()==0 and metrics["dense_top1_in_sparse_top5"].item()==1

def test_hidden_state_metrics_identity():
 hidden=torch.randn(2,3,4);metrics=tensor_metrics(hidden,hidden)
 assert torch.allclose(metrics["cosine_similarity"],torch.ones(6),atol=1e-6) and metrics["relative_l2"].max()==0

def test_nll_is_finite_for_large_logits():
 logits=torch.tensor([[10000.,9999.,-10000.]]);assert torch.isfinite(token_nll(logits,torch.tensor([0]))).all()

def test_schedule_resume_uses_completed_csv(tmp_path):
 target=tmp_path/"measured_moderate.csv";assert should_run_schedule(target);target.write_text("complete\n");assert not should_run_schedule(target);assert should_run_schedule(target,force=True)

def test_identical_low_precision_cosines_are_exactly_one():
 for dtype in (torch.float16,torch.bfloat16,torch.float32):
  value=torch.linspace(-3,3,4096,dtype=dtype).repeat(2,1)
  cosine=stable_cosine(value,value,dtype=torch.float64);metrics=output_metrics(value,value,cosine_dtype=torch.float64)
  assert torch.equal(cosine,torch.ones_like(cosine))
  assert metrics["max_absolute_error"].max()==0 and metrics["relative_l2"].max()==0

def test_identity_diagnostics_distinguish_tiny_and_clear_changes():
 dense=torch.tensor([[1.,2.,3.,4.]]);tiny=dense.clone();tiny[0,-1]+=5e-7;different=dense.clone();different[0,-1]+=1e-2
 tiny_result=identity_metrics(dense,tiny);different_result=identity_metrics(dense,different)
 assert tiny_result["cosine_mean"]<1 and tiny_result["relative_l2_max"]>0 and tiny_result["max_abs_diff"]>0 and tiny_result["allclose"]
 assert different_result["cosine_mean"]<1 and different_result["relative_l2_max"]>0 and not different_result["allclose"]

def test_cosine_is_always_clamped_to_valid_range():
 value=torch.randn(8,2048,dtype=torch.float16)
 cosine=stable_cosine(value,value.clone(),dtype=torch.float32)
 assert bool(((cosine>=-1)&(cosine<=1)).all())

def test_identity_validator_ignores_cosine_but_rejects_real_difference():
 exact=identity_metrics(torch.ones(2,4),torch.ones(2,4));checks=identity_validation_checks(exact,0.,1.,0.,0.,0)
 assert all(check["passed"] for check in checks.values())
 changed=identity_metrics(torch.ones(2,4),torch.zeros(2,4));checks=identity_validation_checks(changed,1.,0.,1.,1.,0)
 assert not all(check["passed"] for check in checks.values())
