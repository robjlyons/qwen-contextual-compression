import json
from pathlib import Path
import pytest,torch
import runtime
from predictor.models import ResidualFactorizedPredictor
from runtime.cuda_indexed import CPP_SOURCE,CUDA_SOURCE,cuda_build_diagnostic,cuda_indexed_ffn,validate_selected_ids
from runtime.phase5c_benchmark import benchmark_orders
from runtime.ffn import neuron_major_down
from runtime.selector import RuntimeSelector,selected_set_comparison,selector_variants
from runtime.temporal import neuron_payload,recover_sequence_metadata,simulate_lru_cache,temporal_mask_statistics
from runtime.weights import resolve_ffn_tensor_names

def test_runtime_is_packaged_and_importable():
 assert runtime.RuntimeSelector is RuntimeSelector;assert "runtime*" in Path("pyproject.toml").read_text()

def test_main_language_model_wins_over_mtp_and_ambiguity_stays_strict():
 names={}
 for role in ("gate_proj.weight","up_proj.weight","down_proj.weight"):
  names[f"model.language_model.layers.0.mlp.{role}"]="main";names[f"mtp.layers.0.mlp.{role}"]="mtp"
 resolved=resolve_ffn_tensor_names(names,0);assert all(name.startswith("model.language_model") for name in resolved.values())
 names["model.language_model.layers.0.mlp.extra.gate_proj.weight"]="other"
 with pytest.raises(RuntimeError,match="Expected one"):resolve_ffn_tensor_names(names,0)

def test_folded_normalization_fp32_and_fp16_shapes_and_sets():
 torch.manual_seed(2);model=ResidualFactorizedPredictor(7,20,32).eval();mean=torch.randn(7);std=torch.rand(7)+.2;base=RuntimeSelector(model,mean,std,.5);variants=selector_variants(base);x=torch.randn(4,7);torch.testing.assert_close(variants["folded_norm_fp32"].scores(x),base.scores(x),rtol=1e-5,atol=1e-5);assert variants["fp16_folded_norm"].scores(x).shape==(4,20);ids=base.select(x);folded=variants["folded_norm_fp32"].select(x);assert ids.shape[-1]==10;assert selected_set_comparison(ids,folded,20)["set_equality_fraction"]==1.;index_sorted=torch.sort(ids).values;assert selected_set_comparison(ids,index_sorted,20)["set_equality_fraction"]==1.;assert 0<=selected_set_comparison(ids,variants["fp16_predictor"].select(x),20)["jaccard"]<=1

def test_transposed_down_matches_original():
 torch.manual_seed(3);down=torch.randn(5,12);ids=torch.tensor([1,4,7]);activation=torch.randn(2,3);expected=torch.nn.functional.linear(activation,down[:,ids]);torch.testing.assert_close(neuron_major_down(activation,ids,down.T.contiguous()),expected)

def test_cuda_backend_is_materialization_free_and_skip_is_diagnostic():
 assert "index_select" not in CUDA_SOURCE;assert "torch/extension.h" not in CUDA_SOURCE;assert "<ATen/ATen.h>" in CUDA_SOURCE and "at::Tensor" in CUDA_SOURCE;assert "torch/extension.h" in CPP_SOURCE;assert "getCurrentCUDAStream" in CUDA_SOURCE and "C10_CUDA_KERNEL_LAUNCH_CHECK" in CUDA_SOURCE;diagnostic=cuda_build_diagnostic();assert "available" in diagnostic

def test_cuda_down_shape_check_uses_braced_torch_check_branches():
 assert 'if(t){TORCH_CHECK' in CUDA_SOURCE;assert '}else{TORCH_CHECK' in CUDA_SOURCE;assert 'if(t)TORCH_CHECK' not in CUDA_SOURCE

def test_id_validation_and_interleaving_are_deterministic():
 validate_selected_ids(torch.tensor([0,2,4]),5,3,True)
 with pytest.raises(ValueError,match="expected"):validate_selected_ids(torch.tensor([0,2]),5,3)
 with pytest.raises(ValueError,match="range"):validate_selected_ids(torch.tensor([0,5]),5,2,True)
 first=benchmark_orders(["dense","cuda","static"],7,42);assert first==benchmark_orders(["dense","cuda","static"],7,42);assert len(first)==7 and all(sorted(row)==["cuda","dense","static"] for row in first)

@pytest.mark.skipif(not cuda_build_diagnostic()["available"],reason="CUDA extension toolchain unavailable")
def test_cuda_indexed_original_and_transposed_match_reference():
 torch.manual_seed(8);x=torch.randn(1,32,device="cuda",dtype=torch.float16);gate=torch.randn(16,32,device="cuda",dtype=torch.float16);up=torch.randn_like(gate);down=torch.randn(32,16,device="cuda",dtype=torch.float16);ids=torch.tensor([1,3,7,9,12,15],device="cuda");activation=torch.nn.functional.silu(torch.nn.functional.linear(x,gate[ids]))*torch.nn.functional.linear(x,up[ids]);expected=torch.nn.functional.linear(activation,down[:,ids]);torch.testing.assert_close(cuda_indexed_ffn(x,ids,gate,up,down),expected,rtol=.03,atol=.03);torch.testing.assert_close(cuda_indexed_ffn(x,ids,gate,up,down.T.contiguous(),True),expected,rtol=.03,atol=.03)

@pytest.mark.skipif(not cuda_build_diagnostic()["available"],reason="CUDA extension toolchain unavailable")
def test_cuda_warp8_and_current_stream():
 torch.manual_seed(9);x=torch.randn(1,32,device="cuda",dtype=torch.float16);gate=torch.randn(17,32,device="cuda",dtype=torch.float16);up=torch.randn_like(gate);down=torch.randn(32,17,device="cuda",dtype=torch.float16);ids=torch.tensor([1,3,7,9,12,15],device="cuda");stream=torch.cuda.Stream()
 with torch.cuda.stream(stream):reference=cuda_indexed_ffn(x,ids,gate,up,down,variant="reference");optimized=cuda_indexed_ffn(x,ids,gate,up,down,variant="warp8")
 stream.synchronize();torch.testing.assert_close(optimized,reference,rtol=.03,atol=.03)

def metadata():return [{"sample":0,"prompt_id":"a","token_position":0},{"sample":1,"prompt_id":"a","token_position":1},{"sample":2,"prompt_id":"b","token_position":0},{"sample":3,"prompt_id":"b","token_position":2}]

def test_temporal_pairs_never_cross_prompt_or_position_gaps():
 masks=torch.tensor([[0,1],[1,2],[0,2],[1,2]]);result=temporal_mask_statistics(masks,metadata());assert result["pair_count"]==1;assert result["pair_metrics"]["intersection"]["mean"]==1

def test_metadata_requires_prompt_and_position(tmp_path):
 path=tmp_path/"meta.jsonl";path.write_text(json.dumps({"prompt_id":"a"})+"\n");assert recover_sequence_metadata(path) is None;assert recover_sequence_metadata(tmp_path/"missing") is None

@pytest.mark.parametrize("capacity",[.5,.55,.6,.65,.75,1.])
def test_lru_cache_accounting_capacity_and_cold_start(capacity):
 masks=torch.tensor([[0,1],[1,2],[0,2],[1,2]]);result=simulate_lru_cache(masks,metadata(),capacity,4);assert result["maximum_observed_cache_size"]<=result["capacity_neurons"];assert result["cold_start"]["tokens"]==3;assert result["steady_state"]["tokens"]==1;json.dumps(result)

def test_lru_exact_hit_miss_and_refill_accounting():
 result=simulate_lru_cache(torch.tensor([[0,1],[1,2],[0,2],[1,2]]),metadata(),.75,4);first,second=result["events"][:2];assert (first["hits"],first["misses"])==(0,2);assert (second["hits"],second["misses"])==(1,1);assert second["bytes_transferred"]==neuron_payload()

def test_neuron_payload_bytes():
 assert neuron_payload()==30720;assert neuron_payload(bits=4)==7680
