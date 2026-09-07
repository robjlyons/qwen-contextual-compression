"""Streaming dense-vs-sparse teacher-forced and deterministic generation runner."""
from __future__ import annotations
import json,time
from pathlib import Path
import pandas as pd
import torch
from end_to_end.hidden_state_metrics import LayerStateMonitor
from end_to_end.model_patcher import OracleMLPPatcher
from end_to_end.schedules import active_parameter_accounting
from end_to_end.streaming_metrics import logit_metrics,token_nll
from extract.extract_ffn import locate_ffn


def should_run_schedule(path:Path,force:bool=False)->bool:
    """A schedule CSV is the atomic completion marker for resume."""
    return force or not path.is_file()


def _cuda_memory(reset_peak:bool=False)->dict:
    result={}
    if not torch.cuda.is_available():return result
    for index in range(torch.cuda.device_count()):
        if reset_peak:torch.cuda.reset_peak_memory_stats(index)
        result[f"cuda:{index}"]={"allocated_mib":torch.cuda.memory_allocated(index)/2**20,"reserved_mib":torch.cuda.memory_reserved(index)/2**20,"peak_allocated_mib":torch.cuda.max_memory_allocated(index)/2**20}
    return result


def validate_wrapper(model,tokenizer,patcher:OracleMLPPatcher,text:str)->dict:
    encoded=tokenizer(text,return_tensors="pt");device=model.get_input_embeddings().weight.device;encoded={k:v.to(device) for k,v in encoded.items()}
    mode={"value":"dense"};ffn_dense={};ffn_sparse={};handles=[]
    for index,wrapper in patcher.wrappers.items():
        def capture(_module,_args,output,layer=index):
            target=ffn_dense if mode["value"]=="dense" else ffn_sparse;target[layer]=output.detach().cpu()
        handles.append(wrapper.register_forward_hook(capture))
    patcher.dense();memory={"before_model_forward":_cuda_memory(reset_peak=True)}
    with torch.inference_mode():dense=model(**encoded,output_hidden_states=True,use_cache=False)
    memory["dense_forward"]=_cuda_memory();dense_logits=dense.logits.detach().cpu();dense_hidden=[state.detach().cpu() for state in dense.hidden_states];del dense;_cuda_memory(reset_peak=True)
    mode["value"]="sparse";patcher.apply_schedule([1.]*len(patcher.wrappers));norms_before=patcher.norm_computation_count
    with torch.inference_mode():sparse=model(**encoded,output_hidden_states=True,use_cache=False)
    memory["sparse_100_percent_forward"]=_cuda_memory();memory["norm_preparation"]={"computed_layers":patcher.norm_computation_count-norms_before,"peak_memory":{}}
    for handle in handles:handle.remove()
    sparse_logits=sparse.logits.detach().cpu();metrics=logit_metrics(dense_logits.flatten(0,-2),sparse_logits.flatten(0,-2)); hidden=[]
    for layer,(left,right) in enumerate(zip(dense_hidden,sparse.hidden_states)):
        right=right.detach().cpu();cosine=torch.nn.functional.cosine_similarity(left.float(),right.float(),dim=-1);relative=torch.linalg.vector_norm((left-right).float(),dim=-1)/torch.linalg.vector_norm(left.float(),dim=-1).clamp_min(1e-12);hidden.append({"layer":layer,"mean_cosine":float(cosine.mean()),"max_relative_l2":float(relative.max())})
    ffn=[]
    for layer in sorted(ffn_dense):
        left,right=ffn_dense[layer].float(),ffn_sparse[layer].float();ffn.append({"layer":layer,"mean_cosine":float(torch.nn.functional.cosine_similarity(left,right,dim=-1).mean()),"max_absolute_error":float((left-right).abs().max())})
    result={"tokens":int(dense_logits.shape[-2]),"logit_cosine_mean":float(metrics["logit_cosine"].mean()),"logit_relative_l2_max":float(metrics["logit_relative_l2"].max()),"max_absolute_logit_difference":float(metrics["max_absolute_logit_difference"].max()),"kl_dense_sparse_mean":float(metrics["kl_dense_sparse"].mean()),"top1_agreement":float(metrics["top1_agreement"].mean()),"down_column_norm_computations":patcher.norm_computation_count-norms_before,"gpu_memory":memory,"ffn_outputs":ffn,"hidden_states":hidden,
      "passed":bool(metrics["logit_cosine"].mean()>=.99999 and metrics["logit_relative_l2"].max()<=1e-3 and metrics["top1_agreement"].mean()==1)}
    if not result["passed"]:raise RuntimeError(f"100% sparse wrapper validation failed; sparse schedules are blocked: {result}")
    return result


def _position_bucket(position:int)->str:
    return "0-31" if position<32 else "32-127" if position<128 else "128-511" if position<512 else "512+"


def evaluate_schedules(model,tokenizer,corpus:list[dict],schedules:dict[str,list[float]],output_dir:Path,
                       max_eval_tokens:int=250,force:bool=False,norm_chunk_columns:int=256)->dict:
    output_dir.mkdir(parents=True,exist_ok=True);teacher=output_dir/"teacher_forced";teacher.mkdir(exist_ok=True)
    local_rows=[]
    def telemetry(layer,metrics):
        length=len(next(iter(metrics.values())))
        for token in range(length):local_rows.append({"layer":layer,"token_offset":token,**{key:float(value[token].cpu()) for key,value in metrics.items()}})
    patcher=OracleMLPPatcher(model,telemetry,norm_chunk_columns);first=locate_ffn(model,0);layers=model.get_submodule(first.layers_path);monitor=LayerStateMonitor(layers)
    ffn_parameters=[sum(parameter.numel() for parameter in wrapper.original.parameters()) for wrapper in patcher.wrappers.values()];total_parameters=sum(parameter.numel() for parameter in model.parameters())
    expansions={name:{"schedule":schedule,**active_parameter_accounting(schedule,ffn_parameters,total_parameters),"actual_oracle_executes_dense_gate_up":True,"runtime_speedup_claimed":False} for name,schedule in schedules.items()}
    (output_dir/"schedule_expansions.json").write_text(json.dumps(expansions,indent=2)+"\n")
    validation_path=output_dir/"wrapper_validation.json"
    if validation_path.exists() and not force:validation=json.loads(validation_path.read_text())
    else:validation=validate_wrapper(model,tokenizer,patcher,corpus[0]["text"]);validation_path.write_text(json.dumps(validation,indent=2)+"\n")
    if not validation.get("passed"):raise RuntimeError("Saved wrapper validation did not pass")
    dense_baseline=teacher/"dense_baseline.csv";run_metadata={}
    try:
      for schedule_name,schedule in schedules.items():
        target=dense_baseline if schedule_name=="dense" else teacher/f"{schedule_name}.csv"
        if not should_run_schedule(target,force):run_metadata[schedule_name]={"status":"skipped_complete"};continue
        started=time.perf_counter();rows=[];hidden_rows=[];ffn_rows=[];evaluated=0;dense_rows=[]
        for prompt_id,item in enumerate(corpus):
          if evaluated>=max_eval_tokens:break
          encoded=tokenizer(item["text"],return_tensors="pt",truncation=True);usable=max(0,encoded["input_ids"].shape[1]-1);take=min(usable,max_eval_tokens-evaluated)
          if take<=0:continue
          # Include one following label and evaluate identical positions in both modes.
          encoded={key:value[:,:take+1].to(model.get_input_embeddings().weight.device) for key,value in encoded.items()}
          monitor.start_dense();patcher.dense()
          with torch.inference_mode():dense=model(**encoded,use_cache=False).logits[:,:-1]
          dense_cpu=dense.detach().cpu();labels=encoded["input_ids"][:,1:].detach().cpu();local_rows.clear();monitor.start_sparse();patcher.apply_schedule(schedule)
          with torch.inference_mode():sparse=model(**encoded,use_cache=False).logits[:,:-1]
          sparse_cpu=sparse.detach().cpu();metrics=logit_metrics(dense_cpu.flatten(0,-2),sparse_cpu.flatten(0,-2));dense_nll=token_nll(dense_cpu.flatten(0,-2),labels.flatten());sparse_nll=token_nll(sparse_cpu.flatten(0,-2),labels.flatten())
          for position in range(take):
            record={"schedule":schedule_name,"prompt_id":prompt_id,"token_position":position,"position_bucket":_position_bucket(position),"category":item.get("category","unknown"),"source":item.get("source","unknown"),"label_token_id":int(labels[0,position]),"dense_nll":float(dense_nll[position]),"sparse_nll":float(sparse_nll[position])}
            record.update({key:float(value[position]) for key,value in metrics.items()});rows.append(record)
            dense_rows.append({**record,"schedule":"dense","sparse_nll":record["dense_nll"],"logit_cosine":1.,"logit_relative_l2":0.,"logit_mse":0.,"max_absolute_logit_difference":0.,"kl_dense_sparse":0.,"kl_sparse_dense":0.,"js_divergence":0.,"top1_agreement":1.,"dense_top1_in_sparse_top5":1.,"sparse_top1_in_dense_top5":1.,"top5_set_overlap":1.,"top10_set_overlap":1.})
          for row in monitor.rows:row.update({"schedule":schedule_name,"prompt_id":prompt_id});hidden_rows.extend(monitor.rows)
          for row in local_rows:row.update({"schedule":schedule_name,"prompt_id":prompt_id});ffn_rows.extend(local_rows)
          evaluated+=take;del dense,sparse,dense_cpu,sparse_cpu
        pd.DataFrame(rows).to_csv(target,index=False);pd.DataFrame(hidden_rows).to_csv(output_dir/f"hidden_state_drift_{schedule_name}.csv",index=False);pd.DataFrame(ffn_rows).to_csv(output_dir/f"ffn_local_error_{schedule_name}.csv",index=False)
        if schedule_name!="dense" and (not dense_baseline.exists() or force):pd.DataFrame(dense_rows).to_csv(dense_baseline,index=False)
        run_metadata[schedule_name]={"status":"complete","tokens":evaluated,"wall_seconds":time.perf_counter()-started}
    finally:monitor.close();patcher.restore()
    (output_dir/"run_metadata.json").write_text(json.dumps(run_metadata,indent=2)+"\n");return run_metadata


def generation_pairs(model,tokenizer,corpus,schedules,output_dir,max_prompts=20,max_new_tokens=64):
    patcher=OracleMLPPatcher(model);rows=[]
    try:
      for prompt_id,item in enumerate(corpus[:max_prompts]):
        encoded=tokenizer(item["text"],return_tensors="pt").to(model.get_input_embeddings().weight.device);patcher.dense()
        with torch.inference_mode():dense=model.generate(**encoded,max_new_tokens=max_new_tokens,do_sample=False)
        dense_new=dense[0,encoded.input_ids.shape[1]:].tolist()
        for name,schedule in schedules.items():
          patcher.apply_schedule(schedule)
          with torch.inference_mode():sparse=model.generate(**encoded,max_new_tokens=max_new_tokens,do_sample=False)
          sparse_new=sparse[0,encoded.input_ids.shape[1]:].tolist();limit=min(len(dense_new),len(sparse_new));divergence=next((i for i in range(limit) if dense_new[i]!=sparse_new[i]),limit if len(dense_new)==len(sparse_new) else limit)
          rows.append({"prompt_id":prompt_id,"prompt":item["text"],"schedule":name,"dense_generation":tokenizer.decode(dense_new),"sparse_generation":tokenizer.decode(sparse_new),"first_divergence_position":divergence,"common_prefix_length":divergence,"dense_tokens":len(dense_new),"sparse_tokens":len(sparse_new),"exact_sequence_match":dense_new==sparse_new})
    finally:patcher.restore()
    directory=output_dir/"generation";directory.mkdir(exist_ok=True);frame=pd.DataFrame(rows);frame.to_json(directory/"generation_pairs.jsonl",orient="records",lines=True);frame.groupby("schedule",as_index=False).agg(prompts=("prompt_id","count"),exact_sequence_match=("exact_sequence_match","mean"),first_divergence_median=("first_divergence_position","median")).to_csv(directory/"generation_summary.csv",index=False);return frame
