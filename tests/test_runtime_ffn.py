import inspect,json
from pathlib import Path
import pytest,torch
from safetensors.torch import save_file
from predictor.models import ResidualFactorizedPredictor,architecture_metadata
from runtime.diagnostics import comparison,index_locality,weight_payload
from runtime.ffn import DenseFFN,StaticPackedFFN,TorchDynamicSparseFFN,masked_dense_reference
from runtime.selector import RuntimeSelector
from runtime.triton_sparse_ffn import TRITON_AVAILABLE,triton_indexed_ffn
from runtime.weights import extract_ffn_layer_weights,load_runtime_weights,resolve_ffn_tensor_names,validate_weight_shapes

def tensors():
 g=torch.Generator().manual_seed(4);return torch.randn(12,5,generator=g),torch.randn(12,5,generator=g),torch.randn(5,12,generator=g),torch.randn(1,5,generator=g)

def test_shapes_dense_and_sparse_reference_equivalence():
 gate,up,down,x=tensors();assert validate_weight_shapes({"gate_proj.weight":gate,"up_proj.weight":up,"down_proj.weight":down})==(5,12);ids=torch.tensor([1,3,4,8,9,11]);dense=DenseFFN(gate,up,down);dynamic=TorchDynamicSparseFFN(gate,up,down);dg,du,da=dense.components(x);sg,su,sa=dynamic.components(x,ids);torch.testing.assert_close(sg,dg[:,ids]);torch.testing.assert_close(su,du[:,ids]);torch.testing.assert_close(sa,da[:,ids]);torch.testing.assert_close(dynamic(x,ids),masked_dense_reference(x,ids,gate,up,down));packed=StaticPackedFFN(*dynamic.pack(ids));torch.testing.assert_close(packed(x),dynamic(x,ids))

def test_runtime_selector_normalizes_and_accepts_only_x():
 model=ResidualFactorizedPredictor(5,12,32).eval();mean=torch.randn(5);std=torch.rand(5)+.2;selector=RuntimeSelector(model,mean,std,.5);x=torch.randn(2,5);torch.testing.assert_close(selector.normalize(x),(x.float()-mean)/std);expected=torch.topk(model((x-mean)/std),6,-1,sorted=False).indices;actual=selector.select(x);torch.testing.assert_close(torch.sort(actual).values,torch.sort(expected).values);assert list(inspect.signature(selector.select).parameters)==["x"] and actual.shape==(2,6)

def test_selective_local_extraction_reads_only_layer_ffn(tmp_path):
 names={"model.layers.0.mlp.gate_proj.weight":"a.safetensors","model.layers.0.mlp.up_proj.weight":"b.safetensors","model.layers.0.mlp.down_proj.weight":"b.safetensors","model.layers.1.mlp.gate_proj.weight":"other.safetensors"};gate,up,down,_=tensors();save_file({"model.layers.0.mlp.gate_proj.weight":gate},tmp_path/"a.safetensors");save_file({"model.layers.0.mlp.up_proj.weight":up,"model.layers.0.mlp.down_proj.weight":down},tmp_path/"b.safetensors");(tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map":names}));output=tmp_path/"runtime"/"layer.safetensors";result=extract_ffn_layer_weights(output,layer=0,model_dir=tmp_path);weights,metadata=load_runtime_weights(output);assert set(weights)=={"gate_proj.weight","up_proj.weight","down_proj.weight"};assert metadata["hidden_size"]=="5" and result["metadata"]["intermediate_size"]=="12";assert resolve_ffn_tensor_names(names,0)["gate_proj.weight"].startswith("model.layers.0")

def test_payload_locality_comparison_and_json_serialization():
 payload=weight_payload(5,12,6);assert payload["sparse_weight_read"]["bytes"]*2==payload["dense_weight_read"]["bytes"];locality=index_locality(torch.tensor([[0,1,3,7],[2,3,4,8]]));assert locality["samples"]==2;json.dumps({"payload":payload,"locality":locality,"comparison":comparison(torch.ones(1,3),torch.ones(1,3))})

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_fp16_cuda_dynamic_smoke():
 gate,up,down,x=(value.half().cuda() for value in tensors());TorchDynamicSparseFFN(gate,up,down)(x,torch.tensor([1,3,4,8,9,11],device="cuda"))

@pytest.mark.skipif(not (torch.cuda.is_available() and TRITON_AVAILABLE),reason="CUDA/Triton unavailable")
def test_triton_matches_dynamic():
 gate,up,down,x=(value.half().cuda() for value in tensors());ids=torch.tensor([1,3,4,8,9,11],device="cuda");reference=TorchDynamicSparseFFN(gate,up,down)(x,ids);torch.testing.assert_close(triton_indexed_ffn(x,ids,gate,up,down),reference,rtol=.02,atol=.02);torch.testing.assert_close(triton_indexed_ffn(x,ids,gate,up,down.T.contiguous(),True),reference,rtol=.02,atol=.02)
