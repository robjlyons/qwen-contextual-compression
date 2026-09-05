"""Aligned, resumable capture of genuine MLP inputs from one official forward."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json, os, random, resource, time
from pathlib import Path
from typing import Iterable
import torch
from extract.extract_ffn import locate_ffn


BUILTIN_CORPUS = [
 {"category":"prose","source":"builtin","text":"Morning light crossed the harbour while commuters filled the narrow streets."},
 {"category":"prose","source":"builtin","text":"Explain how a community garden can improve an urban neighbourhood."},
 {"category":"qa","source":"builtin","text":"What physical process causes rainbows, and why are their colours ordered?"},
 {"category":"qa","source":"builtin","text":"Compare renewable and non-renewable energy sources with two examples of each."},
 {"category":"reasoning","source":"builtin","text":"All red boxes are heavy. Some heavy objects are fragile. What follows logically, and what does not?"},
 {"category":"reasoning","source":"builtin","text":"Plan three dependency-aware steps for migrating a small database with minimal downtime."},
 {"category":"mathematics","source":"builtin","text":"Solve 4x - 9 = 35, verify the answer, and explain each algebraic step."},
 {"category":"mathematics","source":"builtin","text":"A tank is three fifths full and contains 180 litres. Find its total capacity."},
 {"category":"code","source":"builtin","text":"Python: debug `def mean(xs): return sum(xs) / len(x)` for an empty input."},
 {"category":"code","source":"builtin","text":"JavaScript: write a function that groups records by a string key without mutating input."},
 {"category":"code","source":"builtin","text":"C++: explain RAII and show how unique_ptr prevents a resource leak."},
 {"category":"code","source":"builtin","text":"Shell: safely find all .log files older than seven days without deleting them."},
 {"category":"technical","source":"builtin","text":"Describe the difference between latency, throughput, memory bandwidth, and compute utilisation."},
 {"category":"technical","source":"builtin","text":"Explain why feedback control can oscillate when gain and delay are both large."},
 {"category":"dialogue","source":"builtin","text":"User: I do not understand the last step.\nAssistant: Let us restate it with a concrete example."},
 {"category":"dialogue","source":"builtin","text":"User: Give a detailed answer, then summarise it in one sentence.\nAssistant:"},
]


def load_corpus(path: Path | None = None, dataset: str | None = None,
                dataset_split: str = "train", text_field: str = "text",
                category_field: str = "category") -> list[dict]:
    if dataset:
        from datasets import load_dataset
        rows=[]
        for item in load_dataset(dataset,split=dataset_split):
            rows.append({"text":str(item[text_field]),"category":str(item.get(category_field,"unknown")),"source":dataset})
        return rows
    if path is None: return [dict(row) for row in BUILTIN_CORPUS]
    if path.suffix.lower()==".jsonl":
        rows=[]
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            item=json.loads(line); rows.append({"text":str(item[text_field]),"category":str(item.get(category_field,"unknown")),"source":str(item.get("source",path.name))})
        return rows
    return [{"text":line,"category":"unknown","source":path.name} for line in path.read_text().splitlines() if line.strip()]


def mixer_type(layer: torch.nn.Module, config=None, layer_index: int | None=None) -> str:
    """Record the official layer mixer without treating neuron IDs across layers as shared."""
    if config is not None and layer_index is not None:
        layer_types=getattr(config,"layer_types",None)
        if layer_types and layer_index<len(layer_types): return str(layer_types[layer_index])
    for name,module in layer.named_children():
        lowered=f"{name} {type(module).__name__}".lower()
        if "delta" in lowered or "linear_attn" in lowered or "linearattention" in lowered: return "gated_deltanet"
        if "attn" in lowered or "attention" in lowered: return "full_attention"
    return "unknown"


def _atomic_json(path: Path, value: dict) -> None:
    temporary=path.with_suffix(path.suffix+".tmp"); temporary.write_text(json.dumps(value,indent=2)+"\n"); os.replace(temporary,path)


def validate_layer_sample_alignment(activation_root:Path,layers:list[int])->int:
    """Fail if any corresponding layer shard contains different stable sample IDs."""
    references={path.name:torch.load(path,map_location="cpu",weights_only=True)["sample_ids"] for path in sorted((activation_root/f"layer_{layers[0]:03d}").glob("shard_*.pt"))}
    for layer in layers[1:]:
        paths={path.name:path for path in (activation_root/f"layer_{layer:03d}").glob("shard_*.pt")}
        if set(paths)!=set(references): raise RuntimeError(f"Layer {layer} shard set does not align with layer {layers[0]}")
        for name,expected in references.items():
            observed=torch.load(paths[name],map_location="cpu",weights_only=True)["sample_ids"]
            if not torch.equal(observed,expected): raise RuntimeError(f"Sample IDs differ for layer {layer}, {name}")
    return sum(len(value) for value in references.values())


def capture_multilayer(model,tokenizer,layers:list[int],corpus:list[dict],output_dir:Path,
                       max_tokens_per_layer:int=2000,activation_dtype=torch.float16,
                       exclude_special:bool=True,seed:int=42,revision:str|None=None) -> dict:
    """Run complete sequences normally and capture identical content-token IDs at every MLP."""
    started=time.perf_counter(); output_dir.mkdir(parents=True,exist_ok=True); activation_root=output_dir/"activations"; activation_root.mkdir(exist_ok=True)
    state_path=output_dir/"capture_state.json"; state=json.loads(state_path.read_text()) if state_path.exists() else {"completed_prompts":[],"samples":0,"shards":0}
    completed=set(state["completed_prompts"]); sample_count=int(state["samples"]); shard_index=int(state["shards"])
    located={layer:locate_ffn(model,layer) for layer in layers}; captured={}; handles=[]
    backbone_path=located[layers[0]].layers_path.rsplit(".layers",1)[0]
    backbone=model.get_submodule(backbone_path) if backbone_path else model
    def make_hook(layer):
        def hook(_module,args): captured[layer]=args[0].detach().cpu()
        return hook
    for layer,ffn in located.items(): handles.append(ffn.module.register_forward_pre_hook(make_hook(layer)))
    samples_path=output_dir/"samples.jsonl"
    if samples_path.exists():
        committed=samples_path.read_text().splitlines()[:sample_count]
        samples_path.write_text("\n".join(committed)+("\n" if committed else ""))
    sample_file=samples_path.open("a")
    try:
      with torch.inference_mode():
       prompt_order=list(range(len(corpus))); random.Random(seed).shuffle(prompt_order)
       for prompt_id in prompt_order:
        item=corpus[prompt_id]
        if prompt_id in completed or sample_count>=max_tokens_per_layer: continue
        encoded=tokenizer(item["text"],return_tensors="pt",truncation=True)
        ids=encoded["input_ids"][0]; attention=encoded.get("attention_mask",torch.ones_like(encoded["input_ids"]))[0].bool(); special=torch.zeros_like(ids,dtype=torch.bool)
        if tokenizer.all_special_ids:
            special=torch.isin(ids,torch.tensor(tokenizer.all_special_ids))
        keep=attention & (~special if exclude_special else torch.ones_like(special))
        positions=torch.arange(len(ids))[keep]; remaining=max_tokens_per_layer-sample_count; positions=positions[:remaining]
        if not len(positions): completed.add(prompt_id); continue
        device=model.get_input_embeddings().weight.device; captured.clear(); backbone(**{k:v.to(device) for k,v in encoded.items()},use_cache=False)
        if set(captured)!=set(layers): raise RuntimeError(f"Expected hooks {layers}, observed {sorted(captured)}")
        token_ids=ids[positions]; sample_ids=torch.arange(sample_count,sample_count+len(positions),dtype=torch.int64)
        payload_common={"sample_ids":sample_ids,"token_ids":token_ids,"prompt_ids":torch.full_like(sample_ids,prompt_id),"token_positions":positions.to(torch.int64)}
        for layer in layers:
            directory=activation_root/f"layer_{layer:03d}"; directory.mkdir(exist_ok=True)
            payload={"inputs":captured[layer][0,positions].to(activation_dtype),**payload_common}
            target=directory/f"shard_{shard_index:05d}.pt"; temporary=target.with_suffix(".pt.tmp"); torch.save(payload,temporary); os.replace(temporary,target)
        for local,(sample_id,position,token_id) in enumerate(zip(sample_ids.tolist(),positions.tolist(),token_ids.tolist())):
            sample_file.write(json.dumps({"sample_id":sample_id,"prompt_id":prompt_id,"token_position":position,"token_id":token_id,
              "decoded_token":tokenizer.decode([token_id]),"category":item["category"],"source":item["source"],"special_token":bool(special[position])})+"\n")
        sample_file.flush(); sample_count+=len(positions); shard_index+=1; completed.add(prompt_id)
        state={"completed_prompts":sorted(completed),"samples":sample_count,"shards":shard_index}; _atomic_json(state_path,state)
    finally:
      sample_file.close()
      for handle in handles: handle.remove()
    aligned_samples=validate_layer_sample_alignment(activation_root,layers)
    if aligned_samples!=sample_count: raise RuntimeError(f"Aligned shard count {aligned_samples} != capture state {sample_count}")
    elapsed=time.perf_counter()-started; peak_gpu=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    metadata={"model":getattr(model.config,"_name_or_path","unknown"),"revision":revision,"layers":layers,"layer_count":len(model.get_submodule(located[layers[0]].layers_path)),
      "mixers":{str(layer):mixer_type(located[layer].layer,model.config,layer) for layer in layers},"samples_per_layer":sample_count,
      "activation_dtype":str(activation_dtype),"hidden_sizes":{str(layer):int(located[layer].gate_proj.weight.shape[1]) for layer in layers},
      "intermediate_sizes":{str(layer):int(located[layer].gate_proj.weight.shape[0]) for layer in layers},"seed":seed,
      "corpus_categories":{category:sum(item["category"]==category for item in corpus) for category in sorted({item["category"] for item in corpus})},
      "corpus_sources":sorted({item["source"] for item in corpus}),
      "capture_wall_seconds":elapsed,"samples_per_second":sample_count/max(elapsed,1e-9),"peak_gpu_vram_bytes":peak_gpu,
      "peak_system_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"disk_bytes":sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file()),
      "hardware":{"cuda":torch.cuda.get_device_name() if torch.cuda.is_available() else None,"device_map":getattr(model,"hf_device_map",None)},
      "timestamp":datetime.now(timezone.utc).isoformat(),"completed_prompts":len(completed)}
    _atomic_json(output_dir/"metadata.json",metadata); return metadata
