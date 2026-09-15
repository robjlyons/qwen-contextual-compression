"""Track-B kernel-only and selector-plus-indexed benchmark orchestration."""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
import torch
from runtime.benchmark import _events,_memory
from runtime.cuda_indexed import cuda_build_diagnostic,cuda_indexed_ffn,load_cuda_indexed
from runtime.diagnostics import comparison,hardware_metadata
from runtime.ffn import DenseFFN,StaticPackedFFN,TorchDynamicSparseFFN
from runtime.selector import RuntimeSelector,selector_variants
from runtime.triton_sparse_ffn import TRITON_AVAILABLE,triton_indexed_ffn
from runtime.weights import load_runtime_weights

def _row(selector_ms,ffn,total,memory,dense_ms,extra=None):
    result={"selector_ms":selector_ms,"ffn_ms":ffn["mean_ms"],"total_ms":total["mean_ms"],"temporary_MiB":memory["temporary_allocated_delta"]/2**20,"speedup_vs_dense":dense_ms/total["mean_ms"],"ffn_timing":ffn,"total_timing":total,"memory":memory};result.update(extra or {});return result

def run_indexed_benchmark(results_dir:Path,runtime_weights:Path,layer:int,stage1_run:str,selector_variant="fp16_folded_norm",id_order="unsorted",warmup=20,iterations=200,device="cuda"):
    if layer!=0:raise ValueError("Phase 5B remains layer 0 only")
    if not torch.cuda.is_available():raise RuntimeError("indexed benchmark requires CUDA")
    layer_dir=Path(results_dir)/f"layer_{layer:03d}";data=torch.load(layer_dir/"targets"/"targets.pt",map_location="cpu",weights_only=True);splits=json.loads((layer_dir/"splits.json").read_text());samples=torch.tensor(splits["validation"]);x=data["inputs"].index_select(0,samples).to(device);base=RuntimeSelector.from_checkpoint(layer_dir/stage1_run/"best.pt",.5,device);selector=selector_variants(base)[selector_variant].to(device).eval();weights,_=load_runtime_weights(runtime_weights,device);gate,up,down=weights["gate_proj.weight"],weights["up_proj.weight"],weights["down_proj.weight"];down_t=down.T.contiguous();dense=DenseFFN(gate,up,down);dynamic=TorchDynamicSparseFFN(gate,up,down);states=[x[i:i+1] for i in range(len(x))]
    def choose(state):
        scores=selector.scores(state);ids=torch.topk(scores,gate.shape[0]//2,-1,sorted=id_order=="score_sorted").indices[0]
        return torch.sort(ids).values if id_order=="index_sorted" else ids
    ids=[choose(state) for state in states];packed=StaticPackedFFN(*dynamic.pack(ids[0]));dense_timing=_events(lambda i:dense(states[i%len(states)].to(gate.dtype)),warmup,iterations);selector_timing=_events(lambda i:choose(states[i%len(states)]),warmup,iterations);rows={"dense":_row(0.,dense_timing,dense_timing,_memory(lambda:dense(states[0].to(gate.dtype))),dense_timing["mean_ms"])}
    ffn=lambda i:dynamic(states[i%len(states)].to(gate.dtype),ids[i%len(ids)]);total=lambda i:dynamic(states[i%len(states)].to(gate.dtype),choose(states[i%len(states)]));rows["torch_dynamic"]=_row(selector_timing["mean_ms"],_events(ffn,warmup,iterations),_events(total,warmup,iterations),_memory(lambda:total(0)),dense_timing["mean_ms"],{"materializes_selected_weights":True})
    static_timing=_events(lambda i:packed(states[0].to(gate.dtype)),warmup,iterations);rows["static_packed"]=_row(0.,static_timing,static_timing,_memory(lambda:packed(states[0].to(gate.dtype))),dense_timing["mean_ms"],{"diagnostic_only":True})
    reference=dynamic(states[0].to(gate.dtype),ids[0]);quality={"static_packed":comparison(reference,packed(states[0].to(gate.dtype)))}
    if TRITON_AVAILABLE:
        for name,layout,is_t in (("triton_original_down",down,False),("triton_transposed_down",down_t,True)):
            op=lambda i,d=layout,t=is_t:triton_indexed_ffn(states[i%len(states)].to(gate.dtype),ids[i%len(ids)],gate,up,d,t);complete=lambda i,d=layout,t=is_t:triton_indexed_ffn(states[i%len(states)].to(gate.dtype),choose(states[i%len(states)]),gate,up,d,t);rows[name]=_row(selector_timing["mean_ms"],_events(op,warmup,iterations),_events(complete,warmup,iterations),_memory(lambda:complete(0)),dense_timing["mean_ms"],{"materializes_selected_weights":False});quality[name]=comparison(reference,op(0))
    else:rows["triton_indexed"]={"skipped":"Triton unavailable"}
    extension,diagnostic=load_cuda_indexed()
    if extension is not None:
        for name,layout,is_t in (("cuda_original_down",down,False),("cuda_transposed_down",down_t,True)):
            op=lambda i,d=layout,t=is_t:cuda_indexed_ffn(states[i%len(states)].to(gate.dtype),ids[i%len(ids)],gate,up,d,t);complete=lambda i,d=layout,t=is_t:cuda_indexed_ffn(states[i%len(states)].to(gate.dtype),choose(states[i%len(states)]),gate,up,d,t);rows[name]=_row(selector_timing["mean_ms"],_events(op,warmup,iterations),_events(complete,warmup,iterations),_memory(lambda:complete(0)),dense_timing["mean_ms"],{"materializes_selected_weights":False});quality[name]=comparison(reference,op(0))
    else:rows["cuda_indexed"]={"skipped":diagnostic["reason"]}
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");out=Path(results_dir).parent/"runtime"/f"phase5b_indexed_{id_order}_{stamp}";out.mkdir(parents=True);config={"layer":layer,"retention":.5,"stage1_run":stage1_run,"selector_variant":selector_variant,"id_order":id_order,"id_sort_cost_included":True,"down_transpose_preprocessing":"one-time model load"};(out/"config.json").write_text(json.dumps(config,indent=2)+"\n");(out/"hardware.json").write_text(json.dumps(hardware_metadata(),indent=2)+"\n");(out/"indexed_benchmark.json").write_text(json.dumps(rows,indent=2)+"\n");(out/"quality.json").write_text(json.dumps(quality,indent=2)+"\n");table=["| Backend | Selector ms | FFN ms | Total ms | Temp MiB | Speedup |","|---|---:|---:|---:|---:|---:|"]+[f'| {name} | {row["selector_ms"]:.4f} | {row["ffn_ms"]:.4f} | {row["total_ms"]:.4f} | {row["temporary_MiB"]:.3f} | {row["speedup_vs_dense"]:.3f}x |' for name,row in rows.items() if "total_ms" in row];(out/"report.md").write_text("# Phase 5B indexed runtime track\n\nSingle layer-0 FFN only; no model tokens/s or full-model VRAM claim. Sorting cost is included in total latency.\n\n"+"\n".join(table)+"\n");return {"output":str(out),"backends":rows,"quality":quality}
