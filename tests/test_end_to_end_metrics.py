import torch
from end_to_end.hidden_state_metrics import tensor_metrics
from end_to_end.streaming_metrics import logit_metrics,token_nll
from end_to_end.evaluation_runner import should_run_schedule

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
