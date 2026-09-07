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
from evaluation.metrics import output_metrics


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


def identity_metrics(dense:torch.Tensor,sparse:torch.Tensor,atol:float=1e-6,rtol:float=1e-6)->dict:
    """Strict identity diagnostics; cosine is FP64 and never decides pass/fail."""
    flattened_dense=dense.flatten(0,-2);flattened_sparse=sparse.flatten(0,-2)
    metrics=output_metrics(flattened_dense,flattened_sparse,cosine_dtype=torch.float64)
    delta=(dense-sparse).float()
    return {"cosine_mean":float(metrics["cosine_similarity"].mean()),"relative_l2_max":float(metrics["relative_l2"].max()),"mse_mean":float(metrics["mse"].mean()),"max_abs_diff":float(metrics["max_absolute_error"].max()),"mean_abs_diff":float(delta.abs().mean()),"exact_equal":bool(torch.equal(dense,sparse)),"allclose":bool(torch.allclose(dense,sparse,atol=atol,rtol=rtol))}


def identity_validation_checks(logits:dict,kl:float,top1:float,max_hidden:float,
                               max_ffn:float,norm_count:int)->dict:
    """Return individually inspectable checks; cosine is intentionally absent."""
    return {"logit_max_abs":{"value":logits["max_abs_diff"],"threshold":1e-6,"passed":logits["max_abs_diff"]<=1e-6},"logit_relative_l2":{"value":logits["relative_l2_max"],"threshold":1e-6,"passed":logits["relative_l2_max"]<=1e-6},"logit_allclose":{"value":logits["allclose"],"threshold":True,"passed":logits["allclose"]},"kl":{"value":kl,"threshold":1e-7,"passed":kl<=1e-7},"top1_agreement":{"value":top1,"threshold":1.0,"passed":top1==1.0},"hidden_max_abs":{"value":max_hidden,"threshold":1e-6,"passed":max_hidden<=1e-6},"ffn_max_abs":{"value":max_ffn,"threshold":1e-6,"passed":max_ffn<=1e-6},"norm_computations":{"value":norm_count,"threshold":0,"passed":norm_count==0}}


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
    sparse_logits=sparse.logits.detach().cpu();metrics=logit_metrics(dense_logits.flatten(0,-2),sparse_logits.flatten(0,-2));logit_identity=identity_metrics(dense_logits,sparse_logits); hidden=[]
    for layer,(left,right) in enumerate(zip(dense_hidden,sparse.hidden_states)):
        right=right.detach().cpu();hidden.append({"layer":layer,**identity_metrics(left,right)})
    ffn=[]
    for layer in sorted(ffn_dense):
        ffn.append({"layer":layer,**identity_metrics(ffn_dense[layer],ffn_sparse[layer])})
    norm_count=patcher.norm_computation_count-norms_before;kl=abs(float(metrics["kl_dense_sparse"].mean()));top1=float(metrics["top1_agreement"].mean());max_hidden=max(item["max_abs_diff"] for item in hidden);max_ffn=max(item["max_abs_diff"] for item in ffn)
    checks=identity_validation_checks(logit_identity,kl,top1,max_hidden,max_ffn,norm_count)
    failed=[name for name,check in checks.items() if not check["passed"]]
    result={"tokens":int(dense_logits.shape[-2]),"logit_cosine_mean":logit_identity["cosine_mean"],"logit_relative_l2_max":logit_identity["relative_l2_max"],"max_absolute_logit_difference":logit_identity["max_abs_diff"],"mean_absolute_logit_difference":logit_identity["mean_abs_diff"],"logits_exact_equal":logit_identity["exact_equal"],"logits_allclose":logit_identity["allclose"],"kl_dense_sparse_mean":float(metrics["kl_dense_sparse"].mean()),"top1_agreement":top1,"down_column_norm_computations":norm_count,"gpu_memory":memory,"checks":checks,"failed_checks":failed,"ffn_outputs":ffn,"hidden_states":hidden,"passed":not failed}
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
