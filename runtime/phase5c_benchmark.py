"""Interleaved, repeated Phase 5C single-layer benchmark."""
from __future__ import annotations
import json,random,subprocess
from datetime import datetime,timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from runtime.benchmark import _memory
from runtime.cuda_indexed import VARIANTS,cuda_indexed_ffn,cuda_indexed_parts,load_cuda_indexed,validate_selected_ids
from runtime.cuda_selector import CudaSelector,load_cuda_selector
from runtime.diagnostics import comparison,distribution_metrics,hardware_metadata
from runtime.ffn import DenseFFN,StaticPackedFFN,TorchDynamicSparseFFN
from runtime.selector import RuntimeSelector,selected_set_comparison,selector_variants
from runtime.weights import load_runtime_weights


def _samples(operation,warmup,iterations):
    for i in range(warmup):operation(i)
    torch.cuda.synchronize();values=[]
    for i in range(iterations):
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();operation(i);end.record();end.synchronize();values.append(begin.elapsed_time(end))
    return values

def _summary(values,round_means):
    x=torch.tensor(values);return {"mean_ms":float(x.mean()),"median_ms":float(x.median()),"p05_ms":float(x.quantile(.05)),"p50_ms":float(x.quantile(.5)),"p95_ms":float(x.quantile(.95)),"p99_ms":float(x.quantile(.99)),"std_ms":float(x.std(unbiased=False)),"min_ms":float(x.min()),"max_ms":float(x.max()),"round_means_ms":round_means}

def benchmark_orders(names,rounds,seed):
    orders=[]
    for round_index in range(rounds):
        order=list(names);random.Random(seed+round_index).shuffle(order);orders.append(order)
    return orders

def interleaved_benchmark(operations,warmup=20,iterations=100,rounds=7,seed=42):
    all_values={name:[] for name in operations};round_means={name:[] for name in operations};orders=[]
    for round_index,order in enumerate(benchmark_orders(operations,rounds,seed)):
        orders.append(order)
        for name in order:
            values=_samples(operations[name],warmup if round_index==0 else 0,iterations);all_values[name].extend(values);round_means[name].append(sum(values)/len(values))
    return {name:_summary(all_values[name],round_means[name]) for name in operations},orders

def selector_components(selector,x,warmup,iterations):
    p=selector.predictor;z=p.encoder(x);res=F.silu(z);update2=p.residual_fc2(p.dropout(F.silu(p.residual_fc1(res))));norm=p.layer_norm(res+update2);scores=p.neurons(norm);k=scores.shape[-1]//2
    operations={"folded_encoder":lambda i:p.encoder(x),"latent_residual":lambda i:p.residual_fc2(p.dropout(F.silu(p.residual_fc1(res))))+res,"layer_norm":lambda i:p.layer_norm(res+update2),"score_head":lambda i:p.neurons(norm),"topk":lambda i:torch.topk(scores,k,-1,sorted=False).indices,"total":lambda i:selector.select(x)}
    return {name:_summary(values,[]) for name,values in ((name,_samples(op,warmup,iterations)) for name,op in operations.items())}

def run_phase5c(results_dir:Path,runtime_weights:Path,stage1_run:str,layer=0,warmup=20,iterations=100,rounds=7,seed=42,device="cuda"):
    if layer!=0:raise ValueError("Phase 5C is layer 0 only")
    if not torch.cuda.is_available():raise RuntimeError("Phase 5C requires CUDA")
    extension,diagnostic=load_cuda_indexed()
    if extension is None:
        if diagnostic.get("compiler_claimed_available"):raise RuntimeError(diagnostic["reason"])
        raise RuntimeError(f'CUDA indexed backend unsupported: {diagnostic["reason"]}')
    layer_dir=Path(results_dir)/"layer_000";data=torch.load(layer_dir/"targets"/"targets.pt",map_location="cpu",weights_only=True);splits=json.loads((layer_dir/"splits.json").read_text());sample_ids=torch.tensor(splits["validation"]);states=data["inputs"].index_select(0,sample_ids).to(device);weights,_=load_runtime_weights(runtime_weights,device);gate,up,down=(weights[name] for name in ("gate_proj.weight","up_proj.weight","down_proj.weight"));dense=DenseFFN(gate,up,down);dynamic=TorchDynamicSparseFFN(gate,up,down);base=RuntimeSelector.from_checkpoint(layer_dir/stage1_run/"best.pt",.5,device);selectors={name:value.to(device).eval() for name,value in selector_variants(base).items()};selector=selectors["fp16_folded_norm"];cuda_selector_extension,cuda_selector_status=load_cuda_selector();cuda_selector=CudaSelector(selector) if cuda_selector_extension is not None else None
    if cuda_selector is None and cuda_selector_status.get("compiler_claimed_available"):raise RuntimeError(cuda_selector_status["reason"])
    xs=[row[None] for row in states];ids=[selector.select(x)[0].contiguous() for x in xs]
    for selected in ids:validate_selected_ids(selected,gate.shape[0],gate.shape[0]//2,check_bounds=True)
    fixed_x=xs[0].to(gate.dtype);fixed_ids=ids[0];packed=StaticPackedFFN(*dynamic.pack(fixed_ids))
    fixed_ops={"dense":lambda i:dense(fixed_x),"static_packed":lambda i:packed(fixed_x)}
    for variant in VARIANTS:fixed_ops[f"cuda_{variant}"]=lambda i,v=variant:cuda_indexed_ffn(fixed_x,fixed_ids,gate,up,down,variant=v)
    fixed,orders_fixed=interleaved_benchmark(fixed_ops,warmup,iterations,rounds,seed)
    dynamic_ops={"dense":lambda i:dense(xs[i%len(xs)].to(gate.dtype)),"selector_only":lambda i:selector.select(xs[i%len(xs)]),"cuda_reference_ffn":lambda i:cuda_indexed_ffn(xs[i%len(xs)].to(gate.dtype),ids[i%len(ids)],gate,up,down,variant="reference"),"cuda_warp8_ffn":lambda i:cuda_indexed_ffn(xs[i%len(xs)].to(gate.dtype),ids[i%len(ids)],gate,up,down,variant="warp8"),"selector_cuda_reference":lambda i:cuda_indexed_ffn(xs[i%len(xs)].to(gate.dtype),selector.select(xs[i%len(xs)])[0],gate,up,down,variant="reference"),"selector_cuda_warp8":lambda i:cuda_indexed_ffn(xs[i%len(xs)].to(gate.dtype),selector.select(xs[i%len(xs)])[0],gate,up,down,variant="warp8")}
    if cuda_selector is not None:
        dynamic_ops["cuda_selector_only"]=lambda i:cuda_selector.select(xs[i%len(xs)])
        dynamic_ops["cuda_selector_cuda_reference"]=lambda i:cuda_indexed_ffn(xs[i%len(xs)].to(gate.dtype),cuda_selector.select(xs[i%len(xs)])[0],gate,up,down,variant="reference")
        dynamic_ops["cuda_selector_cuda_warp8"]=lambda i:cuda_indexed_ffn(xs[i%len(xs)].to(gate.dtype),cuda_selector.select(xs[i%len(xs)])[0],gate,up,down,variant="warp8")
    dynamic_results,orders_dynamic=interleaved_benchmark(dynamic_ops,warmup,iterations,rounds,seed+100)
    cuda_profile={}
    for variant in VARIANTS:
        activation,_=cuda_indexed_parts(fixed_x,fixed_ids,gate,up,down,variant=variant);gate_values=_samples(lambda i,v=variant:extension.gate_up(fixed_x,gate,up,fixed_ids,VARIANTS[v]),warmup,iterations);down_values=_samples(lambda i,v=variant,a=activation:extension.down(a,down,fixed_ids,gate.shape[1],gate.shape[0],False,VARIANTS[v]),warmup,iterations);cuda_profile[variant]={"gate_up_activation":_summary(gate_values,[]),"down":_summary(down_values,[]),"total":fixed[f"cuda_{variant}"],"kernel":{"threads_per_block":256,"warps_per_cta":8 if variant=="warp8" else 8,"original_down":True}}
    baseline_scores=selectors["baseline_fp32"].scores(states);baseline_ids=selectors["baseline_fp32"].select(states);dense_quality_reference=dense(states.to(gate.dtype));selector_quality={}
    if cuda_selector is not None:selectors["cuda_selector"]=cuda_selector
    for name,value in selectors.items():
        scores=value.scores(states);selected=value.select(states);sparse_outputs=torch.cat([dynamic(states[i:i+1].to(gate.dtype),selected[i]) for i in range(len(states))]);selector_quality[name]={"score_cosine":float(F.cosine_similarity(baseline_scores.float(),scores.float(),dim=-1).mean()),**selected_set_comparison(baseline_ids,selected,scores.shape[-1]),"sparse_output":distribution_metrics(dense_quality_reference,sparse_outputs)}
    cuda_quality={name:comparison(dynamic(fixed_x,fixed_ids),cuda_indexed_ffn(fixed_x,fixed_ids,gate,up,down,variant=name)) for name in VARIANTS};selector_profile=selector_components(selector,xs[0],warmup,iterations);memory={name:_memory(lambda op=op:op(0)) for name,op in dynamic_ops.items()};hardware=hardware_metadata();hardware["cuda_toolkit"]=str(diagnostic.get("cuda_home"));hardware["git_commit"]=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip();hardware["compute_capability"]=torch.cuda.get_device_capability()
    dense_ms=dynamic_results["dense"]["mean_ms"];candidate_paths=[name for name in ("selector_cuda_reference","selector_cuda_warp8","cuda_selector_cuda_reference","cuda_selector_cuda_warp8") if name in dynamic_results];best_name=min(candidate_paths,key=lambda n:dynamic_results[n]["mean_ms"]);best=dynamic_results[best_name]["mean_ms"]
    config={"scope":"single Qwen3.8 layer-0 FFN; not full-model tokens/s","H":gate.shape[1],"I":gate.shape[0],"K":fixed_ids.numel(),"dtype":str(gate.dtype),"batch":1,"warmup":warmup,"iterations_per_round":iterations,"rounds":rounds,"seed":seed,"fixed_backend_orders":orders_fixed,"dynamic_backend_orders":orders_dynamic,"primary_down_layout":"original [H,I]"};success={"compute_side":min(fixed["cuda_reference"]["mean_ms"],fixed["cuda_warp8"]["mean_ms"])<fixed["dense"]["mean_ms"],"dynamic_break_even":best<=dense_ms,"speedup_1_10":best<=dense_ms/1.1,"speedup_1_20":best<=dense_ms/1.2,"best_path":best_name,"speedup_vs_dense":dense_ms/best,"break_even_selector_budget_ms":dense_ms-min(fixed["cuda_reference"]["mean_ms"],fixed["cuda_warp8"]["mean_ms"])}
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");out=Path(results_dir).parent/"runtime"/f"phase5c_{stamp}";out.mkdir(parents=True)
    outputs={"config.json":config,"hardware.json":hardware,"benchmark_fixed_state.json":fixed,"benchmark_dynamic.json":dynamic_results,"selector_profile.json":selector_profile,"cuda_profile.json":cuda_profile,"selector_quality.json":selector_quality,"cuda_quality.json":cuda_quality,"memory.json":memory}
    for name,value in outputs.items():(out/name).write_text(json.dumps(value,indent=2)+"\n")
    cuda_rows=["| CUDA variant | Gate/up ms | Down ms | Total FFN ms | Temp MiB | RelL2 |","|---|---:|---:|---:|---:|---:|"]
    for name in VARIANTS:cuda_rows.append(f'| {name} | {cuda_profile[name]["gate_up_activation"]["mean_ms"]:.4f} | {cuda_profile[name]["down"]["mean_ms"]:.4f} | {fixed[f"cuda_{name}"]["mean_ms"]:.4f} | {memory[f"cuda_{name}_ffn"]["temporary_allocated_delta"]/2**20:.3f} | {cuda_quality[name]["relative_l2"]:.7f} |')
    selector_rows=["| Selector | Encoder ms | Latent ms | LayerNorm ms | Head ms | Select ms | Total ms | Jaccard |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    selector_rows.append(f'| fp16_folded_norm | {selector_profile["folded_encoder"]["mean_ms"]:.4f} | {selector_profile["latent_residual"]["mean_ms"]:.4f} | {selector_profile["layer_norm"]["mean_ms"]:.4f} | {selector_profile["score_head"]["mean_ms"]:.4f} | {selector_profile["topk"]["mean_ms"]:.4f} | {selector_profile["total"]["mean_ms"]:.4f} | {selector_quality["fp16_folded_norm"]["jaccard"]:.6f} |')
    e2e_rows=["| Path | Total ms | Speedup vs dense | Temp MiB |","|---|---:|---:|---:|"]+[f'| {name} | {row["mean_ms"]:.4f} | {dense_ms/row["mean_ms"]:.3f}x | {memory[name]["temporary_allocated_delta"]/2**20:.3f} |' for name,row in dynamic_results.items() if name in memory]
    report="# Phase 5C single-layer report\n\nThis is not full-model tokens/s or a full-model VRAM result. Static packing is a diagnostic ceiling only.\n\n## Outcome\n```json\n"+json.dumps(success,indent=2)+"\n```\n\n## CUDA kernels\n"+"\n".join(cuda_rows)+"\n\n## Selector\n"+"\n".join(selector_rows)+"\n\n## End to end\n"+"\n".join(e2e_rows)+"\n"
    (out/"report.md").write_text(report)
    return {"output":str(out),"success":success,"fixed":fixed,"dynamic":dynamic_results}
