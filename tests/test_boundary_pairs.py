import inspect
import torch
from predictor.boundary_pairs import add_gains,pair_gain_matrix,pairwise_swap_loss,remove_gains,shortlist_pairs
from predictor.boundary_reranker import BoundarySwapCascade,BoundarySwapReranker
from predictor.models import ResidualFactorizedPredictor

def test_gain_formulas_match_brute_force():
 g=torch.Generator().manual_seed(7);e=torch.randn(5,generator=g);selected=torch.randn(3,5,generator=g);rejected=torch.randn(4,5,generator=g);expected=torch.empty(3,4)
 for i in range(3):
  for j in range(4):expected[i,j]=e.square().sum()-(e-(rejected[j]-selected[i])).square().sum()
 actual=pair_gain_matrix(e,selected,rejected);assert actual.shape==(3,4);torch.testing.assert_close(actual,expected);torch.testing.assert_close(add_gains(e,rejected),torch.stack([e.square().sum()-(e-v).square().sum() for v in rejected]));torch.testing.assert_close(remove_gains(e,selected),torch.stack([e.square().sum()-(e+v).square().sum() for v in selected]))

def test_shortlists_are_bounded_and_pairs_have_correct_sign_and_sets():
 g=torch.Generator().manual_seed(11);selected=torch.tensor([2,3,4,5]);rejected=torch.tensor([6,7,8,9,10]);result=shortlist_pairs(torch.randn(6,generator=g),selected,rejected,torch.randn(12,generator=g),torch.randn(6,12,generator=g),2,3,4,4)
 assert len(result["gain"])<=8;assert set(result["selected_ids"].tolist())<=set(selected.tolist());assert set(result["rejected_ids"].tolist())<=set(rejected.tolist());assert (result["gain"][result["label"]]>0).all();assert (result["gain"][~result["label"]]<=0).all()

def test_pairwise_loss_pushes_margins_in_the_correct_direction():
 selected=torch.tensor([0.,0.],requires_grad=True);rejected=torch.tensor([0.,0.],requires_grad=True);loss=pairwise_swap_loss(selected,rejected,torch.tensor([True,False]),torch.ones(2));loss.backward();assert rejected.grad[0]<0 and selected.grad[0]>0;assert rejected.grad[1]>0 and selected.grad[1]<0

def test_inference_api_has_no_training_only_inputs_and_stage1_is_frozen():
 stage1=ResidualFactorizedPredictor(4,20,32);cascade=BoundarySwapCascade(stage1,BoundarySwapReranker(4,20),.45,.5,.65);assert list(inspect.signature(cascade.forward).parameters)==["x"];assert all(not p.requires_grad for p in stage1.parameters());x=torch.randn(2,4);expected=torch.topk(stage1(x),10,-1,sorted=False).indices;actual=cascade.final_indices(x);torch.testing.assert_close(torch.sort(actual).values,torch.sort(expected).values)
