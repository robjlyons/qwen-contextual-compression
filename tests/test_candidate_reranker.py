import inspect
import pytest
import torch
from predictor.candidate_analysis import analyze_candidate_retention,recommend_candidate_retention
from predictor.candidate_reranker import CandidateCascade,CandidateReranker,candidate_mask,candidate_topk,hard_topk_count,reranker_accounting,retention_count,ste_topk_count
from predictor.losses import distribution_ce
from predictor.models import ResidualFactorizedPredictor


def test_candidate_and_final_counts_and_retention_validation():
 scores=torch.randn(3,11);ids=candidate_topk(scores,.6);assert ids.shape==(3,7);assert retention_count(11,.5)==6;assert bool((hard_topk_count(scores.gather(-1,ids),6).sum(-1)==6).all())
 with pytest.raises(ValueError,match="greater"):CandidateCascade(ResidualFactorizedPredictor(4,11,32),CandidateReranker(4,11,16),.5,.5)


def test_stage1_is_frozen_non_candidates_excluded_and_inference_is_x_only():
 stage1=ResidualFactorizedPredictor(4,10,32);reranker=CandidateReranker(4,10,16);cascade=CandidateCascade(stage1,reranker,.6,.5);x=torch.randn(2,4);full=cascade(x);candidates,_=cascade.candidate_state(x);present=torch.zeros_like(full,dtype=torch.bool).scatter_(-1,candidates,True)
 assert all(not p.requires_grad for p in stage1.parameters());assert torch.isneginf(full.masked_select(~present)).all();assert list(inspect.signature(cascade.forward).parameters)==["x"];assert cascade.final_indices(x).shape[-1]==5


@pytest.mark.parametrize("dimension",[16,32])
def test_zero_initialized_reranker_preserves_order_and_shapes(dimension):
 reranker=CandidateReranker(5,13,dimension);x=torch.randn(3,5);ids=torch.stack([torch.randperm(13)[:8] for _ in range(3)]);base=torch.randn(3,8);out=reranker(x,ids,base);expected=reranker.normalized_base(base);torch.testing.assert_close(out,expected);assert out.shape==(3,8)


def test_embedding_gather_and_batched_scoring_match_loop():
 reranker=CandidateReranker(4,9,16,False);torch.nn.init.normal_(reranker.neuron_embeddings.weight);x=torch.randn(2,4);ids=torch.tensor([[1,3,5],[2,4,8]]);base=torch.randn(2,3);batched=reranker(x,ids,base);q=torch.nn.functional.silu(reranker.context(x));loop=torch.stack([base[b]+torch.stack([(reranker.neuron_embeddings(ids[b,j])*q[b]).sum() for j in range(3)]) for b in range(2)]);torch.testing.assert_close(batched,loop)


def test_candidate_distribution_ce_and_ste_are_restricted_and_exact():
 scores=torch.randn(2,7,requires_grad=True);target=torch.rand(2,7);assert torch.isfinite(distribution_ce(scores,target));mask=ste_topk_count(scores,5,.5);assert torch.equal(mask.detach(),hard_topk_count(scores,5));mask.sum().backward();assert scores.grad.abs().sum()>0;ids=torch.tensor([[0,2,4,6,8,9,11],[1,3,5,7,8,10,11]]);full=candidate_mask(ids,mask.detach(),12);assert bool((full.sum(-1)==5).all()) and not bool(full[:,ids.new_tensor([1])].all())


def test_reranker_accounting_exact_and_under_one_percent():
 result=reranker_accounting(722944,740480,5120,17408,.6,16,267386880);assert result["candidate_count"]==10445;assert result["stage2_macs"]==5120*16+10445*16==249040;assert result["total_selector_macs"]==971984;assert result["total_mac_fraction"]==971984/267386880<.01;assert result["stage2_parameters"]==5120*16+17408*16


def test_restricted_oracle_ceiling_uses_candidates_only():
 stage=torch.tensor([[9.,8.,7.,6.,5.,4.]]);oracle=torch.tensor([[1.,2.,3.,4.,5.,6.]]);activations=torch.randn(1,6);weight=torch.randn(3,6);result=analyze_candidate_retention(stage,oracle,activations,weight,.75,.5);assert result["candidate_retention"]==.75;assert 0<=result["oracle_top50_coverage"]["mean"]<=1


def test_candidate_choice_uses_ceiling_not_test_metrics():
 rows=[{"candidate_retention":.55,"restricted_candidate_oracle":{"ffn_cosine":.9900}},{"candidate_retention":.60,"restricted_candidate_oracle":{"ffn_cosine":.9950}},{"candidate_retention":.65,"restricted_candidate_oracle":{"ffn_cosine":.9952}}];assert recommend_candidate_retention(rows)==.60
