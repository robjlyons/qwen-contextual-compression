import inspect,json,pytest,torch
from torch import nn
from predictor.accounting import predictor_accounting
from predictor.confidence import fallback_retention
from predictor.datasets import prompt_split
from predictor.losses import distribution_ce
from predictor.metrics import reconstruct_metrics,select_topk,selection_metrics
from predictor.models import FactorizedPredictor,StaticHot
from predictor.targets import oracle_scores
from predictor.xgboost_latent import TargetSVD
from predictor.evaluation import normalise_inputs,prepare_evaluation
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

def _evaluation_fixture(tmp_path):
 target=tmp_path/"layer_000/targets";run=tmp_path/"layer_000/factorized_d3";target.mkdir(parents=True);run.mkdir();inputs=torch.tensor([[1.,2.,3.,4.],[2.,4.,6.,8.],[3.,6.,9.,12.],[4.,8.,12.,16.]])
 raw=torch.tensor([[9.,1.,0.,0.,0.,0.],[8.,1.,0.,0.,0.,0.],[0.,9.,1.,0.,0.,0.],[0.,8.,1.,0.,0.,0.]])
 torch.save({"inputs":inputs,"scores":raw.half(),"raw_scores":raw.half(),"gated_activations":torch.randn(4,6)},target/"targets.pt");torch.save({"weight":torch.randn(2,6),"bias":None},target/"down_projection.pt");(run.parent/"splits.json").write_text(json.dumps({"train":[0,1],"validation":[],"test":[2,3]}));model=FactorizedPredictor(4,6,3);torch.save({"model":model.state_dict(),"mean":torch.tensor([1.,2.,3.,4.]),"std":torch.tensor([0.,2.,3.,4.]),"config":{"kind":"factorized","latent_dim":3}},run/"best.pt");return target,run

def test_cpu_checkpoint_and_dataset_are_prepared_on_cpu_without_leakage(tmp_path):
 target,run=_evaluation_fixture(tmp_path);data,model,tensors,diagnostics=prepare_evaluation(target,run,"cpu");assert all(value.device.type=="cpu" for value in tensors.values() if isinstance(value,torch.Tensor));assert torch.isfinite(tensors["x"]).all();torch.testing.assert_close(tensors["static"],data["raw_scores"][:2].float().mean(0));assert diagnostics["test_samples"]==2

def test_normalisation_uses_checkpoint_statistics_and_clamps_zero_std():
 x=torch.tensor([[2.,6.]]);normalised,mean,std=normalise_inputs(x,torch.tensor([1.,2.]),torch.tensor([0.,2.]),"cpu");torch.testing.assert_close(normalised,torch.tensor([[1e6,2.]]));assert torch.isfinite(normalised).all()

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_cpu_artifacts_move_explicitly_to_cuda(tmp_path):
 target,run=_evaluation_fixture(tmp_path);_data,model,tensors,_diagnostics=prepare_evaluation(target,run,"cuda:0");assert next(model.parameters()).is_cuda and all(value.is_cuda for value in tensors.values() if isinstance(value,torch.Tensor))

def test_missing_evaluation_file_has_clear_path(tmp_path):
 with pytest.raises(FileNotFoundError,match="targets.pt"):prepare_evaluation(tmp_path,tmp_path/"run","cpu")
