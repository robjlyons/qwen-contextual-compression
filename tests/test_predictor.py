import inspect,torch
from torch import nn
from predictor.accounting import predictor_accounting
from predictor.confidence import fallback_retention
from predictor.datasets import prompt_split
from predictor.losses import distribution_ce
from predictor.metrics import reconstruct_metrics,select_topk,selection_metrics
from predictor.models import FactorizedPredictor,StaticHot
from predictor.targets import oracle_scores
from predictor.xgboost_latent import TargetSVD
from extract.extract_ffn import locate_ffn
from conftest import Model
def test_target_formula_and_predictor_antileakage():
 model=Model();f=locate_ffn(model,0);x=torch.randn(3,4);a=f.activation(f.gate_proj(x))*f.up_proj(x);expected=a.abs()*torch.linalg.vector_norm(f.down_proj.weight.float(),dim=0)
 torch.testing.assert_close(oracle_scores(x,f),expected);assert list(inspect.signature(FactorizedPredictor.forward).parameters)==["self","x"]
def test_prompt_split_no_leakage():
 ids=[1,1,2,2,3,3,4,4,5,5,6,6];s=prompt_split(ids);sets=[{ids[i] for i in s[k]} for k in ("train","validation","test")];assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
def test_factorized_shape_k_and_parameter_accounting():
 m=FactorizedPredictor(5120,17408,64);x=torch.randn(2,5120);scores=m(x);assert scores.shape==(2,17408) and select_topk(scores,.3).shape[-1]==5223
 a=predictor_accounting(m,5120,17408,.5);assert a["parameters"]==sum(p.numel() for p in m.parameters()) and a["macs_per_token"]==5120*64+64*17408
def test_weighted_mass_penalises_missing_large_neuron():
 target=torch.tensor([[100.,10.,1.,0.]]);good=selection_metrics(torch.tensor([[9.,8.,0.,0.]]),target,.5);bad=selection_metrics(torch.tensor([[0.,9.,8.,0.]]),target,.5);assert good["captured_mass"]>bad["captured_mass"]
def test_static_is_deterministic_and_retention_needs_no_retraining():
 target=torch.tensor([[3.,1.,2.],[3.,0.,1.]]);m=StaticHot().fit(target);x=torch.zeros(1,2);assert torch.equal(m(x),m(x)) and select_topk(m(x),.34).shape[-1]==2 and select_topk(m(x),.67).shape[-1]==3
def test_reconstruction_and_distribution_loss():
 a=torch.randn(2,5);w=torch.randn(3,5);idx=torch.tensor([[0,2],[1,4]]);metrics=reconstruct_metrics(a,w,idx);manual=torch.nn.functional.linear(torch.zeros_like(a).scatter_(1,idx,a.gather(1,idx)),w);dense=torch.nn.functional.linear(a,w);torch.testing.assert_close(metrics["mse"],(dense-manual).square().mean(-1));assert torch.isfinite(distribution_ce(torch.randn(2,5),a.abs()))
def test_train_only_svd_and_fallback_average():
 train=torch.randn(20,10);validation=torch.randn(5,10)+100;codec=TargetSVD(3).fit(train);torch.testing.assert_close(codec.mean,train.mean(0));r=fallback_retention(torch.tensor([0.,1.]),.5,.1,.5);assert float(r.mean())==torch.tensor(.55).item()
