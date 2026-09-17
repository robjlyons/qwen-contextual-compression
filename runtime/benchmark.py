"""Cross-platform single-layer Phase 5A benchmark and artifact writer."""
from __future__ import annotations
import json,time
from datetime import datetime,timezone
from pathlib import Path
import torch

from predictor.candidate_reranker import retention_count
from runtime.diagnostics import comparison,distribution_metrics,hardware_metadata,index_locality,summarize_times,weight_payload
from runtime.ffn import DenseFFN,StaticPackedFFN,TorchDynamicSparseFFN
from runtime.selector import RuntimeSelector,assert_x_only_selector
from runtime.triton_sparse_ffn import TRITON_AVAILABLE,triton_indexed_ffn
from runtime.weights import load_runtime_weights

def _events(operation,warmup,iterations):
    for index in range(warmup):operation(index)
    torch.cuda.synchronize();values=[]
    for index in range(iterations):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();operation(index);end.record();end.synchronize();values.append(start.elapsed_time(end))
    return summarize_times(values)

def _memory(operation):
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();before_alloc=torch.cuda.memory_allocated();before_reserved=torch.cuda.memory_reserved();operation();torch.cuda.synchronize();return {"allocated_before":before_alloc,"reserved_before":before_reserved,"peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),"temporary_allocated_delta":torch.cuda.max_memory_allocated()-before_alloc}

def _host_transfer(path,selected_ids,device,warmup,iterations):
    cpu,_=load_runtime_weights(path,"cpu");cpu={name:value.pin_memory() for name,value in cpu.items()};selected_ids=selected_ids.cpu();packed=(cpu["gate_proj.weight"].index_select(0,selected_ids).contiguous().pin_memory(),cpu["up_proj.weight"].index_select(0,selected_ids).contiguous().pin_memory(),cpu["down_proj.weight"].index_select(1,selected_ids).contiguous().pin_memory());dense_bytes=sum(value.numel()*value.element_size() for value in cpu.values());sparse_bytes=sum(value.numel()*value.element_size() for value in packed);dense_time=_events(lambda i:tuple(value.to(device,non_blocking=True) for value in cpu.values()),warmup,iterations);sparse_time=_events(lambda i:tuple(value.to(device,non_blocking=True) for value in packed),warmup,iterations)
    for timing,size in ((dense_time,dense_bytes),(sparse_time,sparse_bytes)):timing["bytes_transferred"]=size;timing["effective_GBps"]=(size/1e9)/(timing["mean_ms"]/1e3)
    return {"warning":"analysis-only CPU-pinned H2D transfer; not the final architecture","dense_full_layer":dense_time,"selected_neuron_payload":sparse_time}

def verify_target_compatibility(inputs,stored_gated,dense_cache,dense_model,atol=.02):
    with torch.inference_mode():
        _,_,gated=dense_model.components(inputs);dense=dense_model(inputs)
    gated_result=comparison(stored_gated,gated);dense_result=comparison(dense_cache,dense)
    if gated_result["relative_l2"]>atol or dense_result["relative_l2"]>atol:raise RuntimeError(f"Runtime checkpoint does not match saved targets: gated={gated_result}, dense={dense_result}")
    return {"gated_activations":gated_result,"dense_outputs":dense_result}

def run_benchmark(results_dir:Path,runtime_weights:Path,layer:int,stage1_run:str,retention=.5,device="cuda",dtype="fp16",warmup=20,iterations=200,profile=False,benchmark_host_transfer=False):
    if layer!=0:raise ValueError("Phase 5A is restricted to layer 0")
    if retention!=.5:raise ValueError("Phase 5A retention is fixed at 0.50")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():raise RuntimeError("Phase 5A latency benchmark requires CUDA")
    layer_dir=Path(results_dir)/f"layer_{layer:03d}";target_dir=layer_dir/"targets";weights,weight_meta=load_runtime_weights(runtime_weights,device);cast={"fp16":torch.float16,"bf16":torch.bfloat16}[dtype];gate=weights["gate_proj.weight"].to(cast);up=weights["up_proj.weight"].to(cast);down=weights["down_proj.weight"].to(cast);dense_model=DenseFFN(gate,up,down);dynamic=TorchDynamicSparseFFN(gate,up,down);selector=RuntimeSelector.from_checkpoint(layer_dir/stage1_run/"best.pt",retention,device);assert_x_only_selector(selector)
    data=torch.load(target_dir/"targets.pt",map_location="cpu",weights_only=True);splits=json.loads((layer_dir/"splits.json").read_text());validation=torch.tensor(splits["validation"]);inputs=data["inputs"].index_select(0,validation).to(device=device,dtype=cast);stored_gated=data["gated_activations"].index_select(0,validation).to(device);dense_cache=torch.load(target_dir/"dense_ffn_outputs.pt",map_location="cpu",weights_only=True).index_select(0,validation).to(device)
    compatibility=verify_target_compatibility(inputs,stored_gated,dense_cache,dense_model);selected=selector.select(inputs);k=retention_count(gate.shape[0],retention)
    if selected.shape[-1]!=k:raise RuntimeError("Runtime selector returned an incorrect Top-K count")
    with torch.inference_mode():
        dense_outputs=dense_model(inputs);sparse_outputs=torch.cat([dynamic(inputs[i:i+1],selected[i]) for i in range(len(inputs))]);quality=distribution_metrics(dense_outputs,sparse_outputs)
        stored_selected=torch.topk(selector.scores(inputs),k,-1,sorted=False).indices;quality["selector_set_mismatches"]=int((torch.sort(selected).values!=torch.sort(stored_selected).values).any(-1).sum())
        packed=StaticPackedFFN(*dynamic.pack(selected[0]));reference=dynamic(inputs[:1],selected[0]);dense_gate,dense_up,dense_activation=dense_model.components(inputs[:1]);sparse_gate,sparse_up,sparse_activation=dynamic.components(inputs[:1],selected[0]);backend_checks={"static_packed":comparison(reference,packed(inputs[:1])),"selected_gate":comparison(dense_gate.index_select(-1,selected[0]),sparse_gate),"selected_up":comparison(dense_up.index_select(-1,selected[0]),sparse_up),"selected_activation":comparison(dense_activation.index_select(-1,selected[0]),sparse_activation)}
        if TRITON_AVAILABLE:backend_checks["triton_indexed"]=comparison(reference,triton_indexed_ffn(inputs[:1],selected[0],gate,up,down))
    states=[inputs[i:i+1] for i in range(len(inputs))];ids=[selected[i] for i in range(len(inputs))];dense_components=dense_model.components(states[0]);packed_weights=dynamic.pack(ids[0]);packed_components=packed.components(states[0]);score_states=[selector.scores(state) for state in states];normalized_states=[selector.normalize(state) for state in states]
    dense_op=lambda i:dense_model(states[i%len(states)]);dynamic_ffn_op=lambda i:dynamic(states[i%len(states)],ids[i%len(ids)]);dynamic_op=lambda i:dynamic(states[i%len(states)],selector.select(states[i%len(states)])[0]);static_op=lambda i:packed(states[0]);triton_op=lambda i:triton_indexed_ffn(states[i%len(states)],selector.select(states[i%len(states)])[0],gate,up,down)
    dense_breakdown={"gate_up":_events(lambda i:(torch.nn.functional.linear(states[0],gate),torch.nn.functional.linear(states[0],up)),warmup,iterations),"activation":_events(lambda i:torch.nn.functional.silu(dense_components[0])*dense_components[1],warmup,iterations),"down":_events(lambda i:torch.nn.functional.linear(dense_components[2],down),warmup,iterations)}
    sparse_breakdown={"gather":_events(lambda i:dynamic.pack(ids[0]),warmup,iterations),"gate_up":_events(lambda i:(torch.nn.functional.linear(states[0],packed_weights[0]),torch.nn.functional.linear(states[0],packed_weights[1])),warmup,iterations),"activation":_events(lambda i:torch.nn.functional.silu(packed_components[0])*packed_components[1],warmup,iterations),"down":_events(lambda i:torch.nn.functional.linear(packed_components[2],packed_weights[2]),warmup,iterations)}
    backends={"dense_gpu":{"timing":_events(dense_op,warmup,iterations),"breakdown":dense_breakdown,"memory":_memory(lambda:dense_op(0)),"retention":1.},"sparse_torch_dynamic":{"timing":_events(dynamic_op,warmup,iterations),"ffn_only_timing":_events(dynamic_ffn_op,warmup,iterations),"breakdown":sparse_breakdown,"memory":_memory(lambda:dynamic_op(0)),"retention":retention},"sparse_static_packed":{"timing":_events(static_op,warmup,iterations),"memory":_memory(lambda:static_op(0)),"retention":retention,"warning":"runtime ceiling / diagnostic only"}}
    if TRITON_AVAILABLE:backends["triton_indexed"]={"timing":_events(triton_op,warmup,iterations),"memory":_memory(lambda:triton_op(0)),"retention":retention,"materialized_gather":False}
    else:backends["triton_indexed"]={"skipped":"Triton unavailable"}
    selector_parts={"total":_events(lambda i:selector.select(states[i%len(states)]),warmup,iterations),"normalization":_events(lambda i:selector.normalize(states[i%len(states)]),warmup,iterations),"predictor":_events(lambda i:selector.predictor(normalized_states[i%len(states)]),warmup,iterations),"topk":_events(lambda i:torch.topk(score_states[i%len(states)],k,-1,sorted=False),warmup,iterations)}
    for value in selector_parts.values():value["mean_us"]=1000*value["mean_ms"]
    dense_ms=backends["dense_gpu"]["timing"]["mean_ms"]
    for value in backends.values():
        if "timing" in value:value["speedup_vs_dense"]=dense_ms/value["timing"]["mean_ms"]
    payload=weight_payload(gate.shape[1],gate.shape[0],k,gate.element_size());host_transfer=_host_transfer(runtime_weights,selected[0],device,warmup,iterations) if benchmark_host_transfer else {"skipped":"not requested"};hardware=hardware_metadata();hardware["free_token_reference"]={"model":"Qwen3.6-27B-NVFP4","gpu":"RTX 3070 Ti 8GB","tokens_per_second":.5,"comparable_to_phase5a":False};timestamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");out=Path(results_dir).parent/"runtime"/f"phase5_layer{layer:03d}_{timestamp}";out.mkdir(parents=True)
    config={"scope":"single layer-0 FFN; not model tokens/s","layer":layer,"selector_checkpoint":str(layer_dir/stage1_run/"best.pt"),"runtime_weights":str(runtime_weights),"runtime_weight_metadata":weight_meta,"retention":retention,"dtype":dtype,"warmup":warmup,"iterations":iterations,"profile_requested":profile,"host_transfer_requested":benchmark_host_transfer,"full_model_vram_claim":False};benchmark={"backends":backends,"selector":selector_parts,"theoretical_payload":payload,"host_transfer":host_transfer,"index_locality":index_locality(selected),"backend_comparison":backend_checks};
    if profile:
        activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True) as profiler:
            with torch.autograd.profiler.record_function("selector"):profile_ids=selector.select(states[0])[0]
            with torch.autograd.profiler.record_function("sparse_dynamic_gather_gate_up_activation_down"):dynamic(states[0],profile_ids)
        profiler.export_chrome_trace(str(out/"trace.json"))
    for name,value in (("benchmark.json",benchmark),("quality.json",{"compatibility":compatibility,"runtime_sparse":quality}),("hardware.json",hardware),("config.json",config)):(out/name).write_text(json.dumps(value,indent=2)+"\n")
    warning="This is one Qwen3.8-27B layer-0 FFN, not full-model token throughput or VRAM savings. GPU-resident dense weights remain allocated."
    dynamic_speedup=backends["sparse_torch_dynamic"]["speedup_vs_dense"];static_speedup=backends["sparse_static_packed"]["speedup_vs_dense"]
    if static_speedup>1.2 and dynamic_speedup<=1.05:interpretation="Static packing is faster but dynamic execution is not: gather/index overhead is the leading bottleneck."
    elif dynamic_speedup>1.05:interpretation="The measured dynamic PyTorch cascade is faster than dense for this single layer and hardware."
    else:interpretation="Neither measured 50% sparse path clearly beats dense; profile fixed kernel, reduction, and indexing overhead before changing retention."
    rows=["| Backend | Mean ms | p95 ms | FFN/s | Speedup |","|---|---:|---:|---:|---:|"]+[f'| {name} | {value["timing"]["mean_ms"]:.4f} | {value["timing"]["p95_ms"]:.4f} | {value["timing"]["ffn_evaluations_per_second"]:.2f} | {value["speedup_vs_dense"]:.3f}x |' for name,value in backends.items() if "timing" in value]
    report="# Phase 5A single-layer report\n\n"+warning+"\n\n## Hardware\n```json\n"+json.dumps(hardware,indent=2)+"\n```\n\n## Quality and checkpoint compatibility\n```json\n"+json.dumps({"compatibility":compatibility,"runtime_sparse":quality,"backend_checks":backend_checks},indent=2)+"\n```\n\n## Timing\n"+"\n".join(rows)+"\n\n## Selector timing and locality\n```json\n"+json.dumps({"selector":selector_parts,"locality":benchmark["index_locality"]},indent=2)+"\n```\n\n## Weight payload and memory warning\n```json\n"+json.dumps(payload,indent=2)+"\n```\n\n## Interpretation\n"+interpretation+"\n"
    (out/"report.md").write_text(report)
    return {"output":str(out),"benchmark":benchmark,"quality":quality,"compatibility":compatibility,"hardware":hardware,"warning":warning,"interpretation":interpretation}
