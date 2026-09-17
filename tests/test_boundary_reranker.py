import inspect,json,pytest,torch
from predictor.boundary_analysis import analyze_boundary_configuration,recommend_boundary
from predictor.boundary_reranker import BoundarySwapCascade,BoundarySwapReranker,boundary_accounting,boundary_counts,boundary_mask,stage1_boundary
from predictor.models import ResidualFactorizedPredictor
from predictor.models import architecture_metadata
from predictor.boundary_training import train_boundary_reranker

def test_boundary_counts_and_invalid_fractions():
 assert boundary_counts(17408,.45,.5,.65)==(7834,8704,11316)
 with pytest.raises(ValueError,match="lock"):boundary_counts(10,.5,.5,.7)
 with pytest.raises(ValueError,match="candidate"):boundary_counts(10,.4,.5,.5)

def test_locked_and_candidate_semantics_and_zero_alpha_identity():
 stage1=ResidualFactorizedPredictor(4,20,32).eval();reranker=BoundarySwapReranker(4,20);cascade=BoundarySwapCascade(stage1,reranker,.4,.5,.7);x=torch.randn(3,4);locked,boundary,_,needed=cascade.state(x);final=cascade.final_indices(x);stage_final=torch.topk(stage1(x),10,-1,sorted=False).indices;final_set=torch.zeros(3,20,dtype=torch.bool).scatter_(-1,final,True);locked_set=final_set.gather(-1,locked);candidate=torch.cat((locked,boundary),-1);candidate_set=torch.zeros_like(final_set).scatter_(-1,candidate,True)
 assert needed==2 and final.shape[-1]==10 and locked_set.all() and not (final_set&~candidate_set).any();assert torch.equal(torch.sort(final).values,torch.sort(stage_final).values);assert all(not p.requires_grad for p in stage1.parameters());assert list(inspect.signature(cascade.forward).parameters)==["x"]

def test_boundary_ste_gradients_do_not_touch_stage1():
 stage1=ResidualFactorizedPredictor(4,20,32);reranker=BoundarySwapReranker(4,20);cascade=BoundarySwapCascade(stage1,reranker,.4,.5,.7);locked,boundary,soft=cascade.final_indices(torch.randn(2,4),ste=True);mask=boundary_mask(locked,boundary,soft,20);mask.sum().backward();assert all(p.grad is None for p in stage1.parameters());assert reranker.alpha.grad is not None

def test_gradfix_initialization_and_gradient_sequence():
 reranker=BoundarySwapReranker(4,20,boundary_init="zero_embedding_alpha_one");assert torch.equal(reranker.neuron_embeddings.weight,torch.zeros_like(reranker.neuron_embeddings.weight));assert reranker.alpha.item()==1.;x=torch.randn(2,4);ids=torch.stack([torch.randperm(20)[:8] for _ in range(2)]);base=torch.randn(2,8);normalized=reranker.normalized_base(base);scores=reranker(x,ids,base);torch.testing.assert_close(scores,normalized);loss=(scores*torch.arange(8.)).sum();loss.backward();assert reranker.neuron_embeddings.weight.grad.norm()>0;assert reranker.context.weight.grad.norm()==0
 with torch.no_grad():reranker.neuron_embeddings.weight.normal_(0,.01)
 reranker.zero_grad();reranker(x,ids,base).square().sum().backward();assert reranker.context.weight.grad.norm()>0 and reranker.alpha.grad is not None

def test_old_zero_alpha_mode_remains_explicitly_supported():
 old=BoundarySwapReranker(4,20,boundary_init="zero_alpha");assert old.alpha.item()==0.;assert old.boundary_init=="zero_alpha"

def test_boundary_accounting_uses_boundary_band():
 result=boundary_accounting(722944,740480,5120,17408,.45,.65,16,267386880);assert result["boundary_count"]==11316-7834==3482;assert result["stage2_macs"]==5120*16+3482*16==137632;assert result["total_selector_macs"]==860576;assert result["total_mac_fraction"]<.005;assert result["neuron_embedding_parameters"]==17408*16

def test_boundary_oracle_matches_expected_bruteforce_set():
 stage=torch.tensor([[9.,8.,7.,6.,5.,4.,3.,2.,1.,0.]]);oracle=torch.tensor([[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]]);a=torch.randn(1,10);w=torch.randn(3,10);result=analyze_boundary_configuration(stage,oracle,a,w,.3,.7,.5);locked,boundary,_,needed=stage1_boundary(stage,.3,.5,.7);chosen=boundary.gather(-1,torch.topk(oracle.gather(-1,boundary),needed,-1).indices);assert torch.equal(torch.sort(torch.cat((locked,chosen),-1)).values,torch.tensor([[0,1,2,5,6]]));assert result["selected_from_boundary"]==2

def test_validation_boundary_choice_prefers_restrictive_eligible():
 rows=[{"lock_retention":.4,"candidate_retention":.6,"boundary_restricted_oracle":{"ffn_cosine":.990}},{"lock_retention":.45,"candidate_retention":.6,"boundary_restricted_oracle":{"ffn_cosine":.9895}},{"lock_retention":.45,"candidate_retention":.65,"boundary_restricted_oracle":{"ffn_cosine":.9902}}];chosen=recommend_boundary(rows,.001);assert chosen["lock_retention"]==.45 and chosen["candidate_retention"]==.6

def test_cpu_boundary_training_and_checkpoint_metadata(tmp_path):
 layer=tmp_path/"layer_000";target=layer/"targets";stage_dir=layer/"stage1";target.mkdir(parents=True);stage_dir.mkdir();g=torch.Generator().manual_seed(4);inputs=torch.randn(12,4,generator=g);raw=torch.rand(12,10,generator=g);gated=torch.randn(12,10,generator=g);torch.save({"inputs":inputs,"scores":raw/raw.sum(-1,keepdim=True),"raw_scores":raw,"gated_activations":gated},target/"targets.pt");torch.save({"weight":torch.randn(3,10,generator=g),"bias":None},target/"down_projection.pt");(target/"sample_metadata.jsonl").write_text("".join(json.dumps({"prompt_ids":i})+"\n" for i in range(12)));(layer/"splits.json").write_text(json.dumps({"train":list(range(8)),"validation":[8,9],"test":[10,11]}));stage=ResidualFactorizedPredictor(4,10,32);torch.save({"model":stage.state_dict(),"mean":inputs[:8].mean(0),"std":inputs[:8].std(0),"config":architecture_metadata(stage,"residual_factorized",4,10,32)},stage_dir/"best.pt")
 result=train_boundary_reranker(tmp_path,0,"stage1","boundary",.3,.7,.5,device="cpu",epochs=1,batch_size=4);checkpoint=torch.load(layer/"boundary"/"best.pt",map_location="cpu",weights_only=False);assert checkpoint["config"]["lock_retention"]==.3 and checkpoint["config"]["candidate_retention"]==.7 and checkpoint["config"]["final_retention"]==.5;assert checkpoint["config"]["boundary_init"]=="zero_embedding_alpha_one";assert result["initial_validation"]["selection_mismatch_count"]==0;assert (layer/"boundary"/"gradient_diagnostics.json").is_file()
