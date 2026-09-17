"""Track-A selector variants with mask and sparse-output quality gates."""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from runtime.benchmark import _events
from runtime.diagnostics import distribution_metrics,hardware_metadata
from runtime.ffn import DenseFFN,TorchDynamicSparseFFN
from runtime.selector import RuntimeSelector,selected_set_comparison,selector_variants
from runtime.weights import load_runtime_weights

def run_selector_benchmark(results_dir:Path,runtime_weights:Path,layer:int,stage1_run:str,retention=.5,warmup=20,iterations=200,device="cuda",compile_variant=False):
    if layer!=0 or retention!=.5:raise ValueError("Phase 5B Track A is fixed to layer 0 at 50% retention")
    if not torch.cuda.is_available():raise RuntimeError("selector latency benchmarking requires CUDA")
    layer_dir=Path(results_dir)/f"layer_{layer:03d}";data=torch.load(layer_dir/"targets"/"targets.pt",map_location="cpu",weights_only=True);splits=json.loads((layer_dir/"splits.json").read_text());ids=torch.tensor(splits["validation"]);x=data["inputs"].index_select(0,ids).to(device);base=RuntimeSelector.from_checkpoint(layer_dir/stage1_run/"best.pt",retention,device);variants={name:value.to(device).eval() for name,value in selector_variants(base).items()};weights,_=load_runtime_weights(runtime_weights,device);dense=DenseFFN(weights["gate_proj.weight"],weights["up_proj.weight"],weights["down_proj.weight"]);sparse=TorchDynamicSparseFFN(weights["gate_proj.weight"],weights["up_proj.weight"],weights["down_proj.weight"])
    baseline_scores=variants["baseline_fp32"].scores(x);baseline_ids=variants["baseline_fp32"].select(x);dense_outputs=dense(x.to(weights["gate_proj.weight"].dtype));rows={}
    for name,variant in variants.items():
        with torch.inference_mode():
            scores=variant.scores(x);selected=variant.select(x);outputs=torch.cat([sparse(x[i:i+1].to(weights["gate_proj.weight"].dtype),selected[i]) for i in range(len(x))]);score_cosine=float(F.cosine_similarity(baseline_scores.float(),scores.float(),dim=-1).mean());quality=distribution_metrics(dense_outputs,outputs)
        states=[x[i:i+1] for i in range(len(x))];timing=_events(lambda i:variant.select(states[i%len(states)]),warmup,iterations);rows[name]={"timing":timing,"score_cosine_vs_baseline":score_cosine,"selection":selected_set_comparison(baseline_ids,selected,scores.shape[-1]),"sparse_quality":quality}
    rows["compiled"]={"skipped":"not requested"}
    if compile_variant:
        try:
            compiled=torch.compile(variants["fp16_folded_norm"],fullgraph=True);states=[x[i:i+1] for i in range(len(x))];timing=_events(lambda i:torch.topk(compiled.scores(states[i%len(states)]),baseline_ids.shape[-1],-1,sorted=False).indices,warmup,iterations);rows["compiled"]={"timing":timing,"source_variant":"fp16_folded_norm"}
        except Exception as error:
            rows["compiled"]={"skipped":f"torch.compile unsupported: {error}"}
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");out=Path(results_dir).parent/"runtime"/f"phase5b_selector_{stamp}";out.mkdir(parents=True);config={"layer":layer,"retention":retention,"stage1_run":stage1_run,"variants":list(variants),"automatic_default_selection":False};quality={name:{key:value for key,value in row.items() if key in ("score_cosine_vs_baseline","selection","sparse_quality")} for name,row in rows.items()};(out/"config.json").write_text(json.dumps(config,indent=2)+"\n");(out/"hardware.json").write_text(json.dumps(hardware_metadata(),indent=2)+"\n");(out/"selector_benchmark.json").write_text(json.dumps(rows,indent=2)+"\n");(out/"quality.json").write_text(json.dumps(quality,indent=2)+"\n");table=["| Variant | Mean ms | Score cosine | Set equality | Changed |","|---|---:|---:|---:|---:|"]+[f'| {name} | {row["timing"]["mean_ms"]:.4f} | {row["score_cosine_vs_baseline"]:.7f} | {row["selection"]["set_equality_fraction"]:.4f} | {row["selection"]["mean_changed_neurons"]:.2f} |' for name,row in rows.items() if "timing" in row and "selection" in row];(out/"report.md").write_text("# Phase 5B selector track\n\nNo fastest variant is automatically promoted; inspect mask and sparse quality.\n\n"+"\n".join(table)+"\n");return {"output":str(out),"variants":rows}
