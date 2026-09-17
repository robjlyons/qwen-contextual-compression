import inspect,pytest,torch
from predictor.models import FactorizedPredictor
from predictor.output_aware import hard_topk_mask,ste_topk_mask,output_cosine_loss,output_relative_loss,reconstruct,ensure_dense_output_cache
def test_hard_and_ste_forward_are_exact_k():
 scores=torch.randn(4,11,requires_grad=True);hard=hard_topk_mask(scores,.5);ste=ste_topk_mask(scores,.5,.5,True);assert torch.equal(ste.detach(),hard) and bool((hard.sum(-1)==6).all())
def test_ste_has_gradient_and_temperature_changes_only_backward():
 scores=torch.randn(3,9,requires_grad=True);a=ste_topk_mask(scores,.5,1.);a.sum().backward();g1=scores.grad.clone();scores.grad.zero_();b=ste_topk_mask(scores,.5,.1);b.sum().backward();assert g1.abs().sum()>0 and scores.grad.abs().sum()>0 and not torch.allclose(g1,scores.grad) and torch.equal(a.detach(),b.detach())
def test_output_losses_zero_for_identity_and_grow_for_error():
 dense=torch.randn(5,7);same=dense.clone();bad=dense+2;assert output_cosine_loss(same,dense).abs()<1e-6 and output_relative_loss(same,dense)==0 and output_relative_loss(bad,dense)>0
def test_training_reconstruction_uses_supervision_not_predictor_api():
 model=FactorizedPredictor(4,6,3);assert list(inspect.signature(model.forward).parameters)==["x"];x=torch.randn(2,4);gated=torch.randn(2,6);weight=torch.randn(5,6);mask=ste_topk_mask(model(x),.5);loss=output_relative_loss(reconstruct(gated,mask,weight),torch.randn(2,5));loss.backward();assert model.encoder.weight.grad.abs().sum()>0
def test_dense_cache_matches_explicit_projection_and_reuses_file(tmp_path):
 torch.save({"gated_activations":torch.randn(8,6)},tmp_path/"targets.pt");down={"weight":torch.randn(4,6),"bias":torch.randn(4)};torch.save(down,tmp_path/"down_projection.pt");path=ensure_dense_output_cache(tmp_path,"cpu",3);cached=torch.load(path,weights_only=True);source=torch.load(tmp_path/"targets.pt",weights_only=True);torch.testing.assert_close(cached.float(),torch.nn.functional.linear(source["gated_activations"],down["weight"],down["bias"]),rtol=2e-3,atol=2e-3);assert ensure_dense_output_cache(tmp_path)==path
@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_ste_cuda():
 scores=torch.randn(2,20,device="cuda",requires_grad=True);ste_topk_mask(scores,.5).sum().backward();assert scores.grad.abs().sum()>0
