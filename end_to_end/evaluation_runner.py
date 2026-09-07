"""Streaming dense-vs-sparse teacher-forced and deterministic generation runner."""
from __future__ import annotations
import gc,json,os,time
from pathlib import Path
import pandas as pd
import torch
import psutil
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


def _rss_mib()->float:return psutil.Process(os.getpid()).memory_info().rss/2**20


def _append_rows(path:Path,rows:list[dict])->None:
    if not rows:return
    if any(isinstance(value,torch.Tensor) for row in rows for value in row.values()):
        raise TypeError("compact result rows must not retain tensors")
    pd.DataFrame(rows).to_csv(path,mode="a",header=not path.exists(),index=False)


def _atomic_json(path:Path,value:dict)->None:
    temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(json.dumps(value,indent=2)+"\n");temporary.replace(path)


def _discard_incomplete_prompt_rows(paths:list[Path],completed_prompts:int)->None:
    """Remove any rows flushed for a prompt whose checkpoint was not committed."""
    for path in paths:
        if not path.exists() or path.suffix!=".csv":continue
        frame=pd.read_csv(path)
        if "prompt_id" in frame:frame=frame[frame.prompt_id<completed_prompts]
        frame.to_csv(path,index=False)


def bounded_prompt_inputs(tokenizer,corpus:list[dict],max_eval_tokens:int,
                          max_prompt_tokens:int,start_prompt:int=0,
                          already_evaluated:int=0):
    """Yield only sequences that fit the remaining target-token budget."""
    remaining=max_eval_tokens-already_evaluated
    for prompt_id,item in enumerate(corpus):
        if prompt_id<start_prompt:continue
        if remaining<=0:break
        encoded=tokenizer(item["text"],return_tensors="pt",truncation=True,
                          max_length=min(max_prompt_tokens,remaining+1))
        take=min(max(0,encoded["input_ids"].shape[1]-1),remaining)
        if take<=0:continue
        yield prompt_id,item,{key:value[:,:take+1] for key,value in encoded.items()},take
        remaining-=take


def evaluate_schedules(model,tokenizer,corpus:list[dict],schedules:dict[str,list[float]],output_dir:Path,
                       max_eval_tokens:int=250,force:bool=False,norm_chunk_columns:int=256,
                       max_prompt_tokens:int=128)->dict:
    """Evaluate one bounded prompt at a time and persist compact rows immediately."""
    if max_eval_tokens<=0:raise ValueError("max_eval_tokens must be positive")
    if max_prompt_tokens<2:raise ValueError("max_prompt_tokens must be at least 2")
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
    else:
        validation_text=tokenizer.decode(tokenizer(corpus[0]["text"],truncation=True,max_length=max_prompt_tokens)["input_ids"])
        validation=validate_wrapper(model,tokenizer,patcher,validation_text);validation_path.write_text(json.dumps(validation,indent=2)+"\n")
    if not validation.get("passed"):raise RuntimeError("Saved wrapper validation did not pass")
    names=[name for name in schedules if name!="dense"];dense_baseline=teacher/"dense_baseline.csv";progress_path=output_dir/"progress.json";memory_path=output_dir/"memory_telemetry.csv"
    artifacts=[dense_baseline,memory_path,*[teacher/f"{name}.csv" for name in names],*[output_dir/f"hidden_state_drift_{name}.csv" for name in names],*[output_dir/f"ffn_local_error_{name}.csv" for name in names]]
    if force:
        for path in artifacts:
            if path.exists():path.unlink()
        if progress_path.exists():progress_path.unlink()
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {"completed_prompts":0,"evaluated_tokens":0,"schedules":names,"max_eval_tokens":max_eval_tokens,"max_prompt_tokens":max_prompt_tokens}
    if progress["schedules"]!=names or progress["max_eval_tokens"]!=max_eval_tokens or progress["max_prompt_tokens"]!=max_prompt_tokens:raise ValueError("Resume configuration differs; use --force or a new output directory")
    if progress_path.exists():_discard_incomplete_prompt_rows(artifacts,int(progress["completed_prompts"]))
    evaluated=int(progress["evaluated_tokens"]);started=time.perf_counter();peak_rss=_rss_mib();baseline_rss=peak_rss
    try:
        batches=bounded_prompt_inputs(tokenizer,corpus,max_eval_tokens,max_prompt_tokens,int(progress["completed_prompts"]),evaluated)
        for prompt_id,item,encoded,take in batches:
          rss_before=_rss_mib();encoded={key:value.to(model.get_input_embeddings().weight.device) for key,value in encoded.items()}
          monitor.start_dense();patcher.dense()
          with torch.inference_mode():dense=model(**encoded,use_cache=False).logits[:,:-1]
          dense_cpu=dense.detach().cpu();del dense;labels=encoded["input_ids"][:,1:].detach().cpu();dense_nll=token_nll(dense_cpu.flatten(0,-2),labels.flatten());rss_dense=_rss_mib();dense_rows=[];truncated=encoded["input_ids"].shape[1]>=max_prompt_tokens
          for position in range(take):dense_rows.append({"schedule":"dense","prompt_id":prompt_id,"token_position":position,"position_bucket":_position_bucket(position),"category":item.get("category","unknown"),"source":item.get("source","unknown"),"label_token_id":int(labels[0,position]),"prompt_truncated":truncated,"forward_sequence_tokens":take+1,"dense_nll":float(dense_nll[position]),"sparse_nll":float(dense_nll[position]),"logit_cosine":1.,"logit_relative_l2":0.,"logit_mse":0.,"max_absolute_logit_difference":0.,"kl_dense_sparse":0.,"kl_sparse_dense":0.,"js_divergence":0.,"top1_agreement":1.,"dense_top1_in_sparse_top5":1.,"sparse_top1_in_dense_top5":1.,"top5_set_overlap":1.,"top10_set_overlap":1.,"dense_margin":float(torch.topk(dense_cpu[0,position].float(),2).values.diff().neg())})
          _append_rows(dense_baseline,dense_rows)
          sparse_peaks={}
          for schedule_name in names:
            local_rows.clear();monitor.start_sparse();patcher.apply_schedule(schedules[schedule_name])
            with torch.inference_mode():sparse=model(**encoded,use_cache=False).logits[:,:-1]
            sparse_cpu=sparse.detach().cpu();metrics=logit_metrics(dense_cpu.flatten(0,-2),sparse_cpu.flatten(0,-2));sparse_nll=token_nll(sparse_cpu.flatten(0,-2),labels.flatten());rows=[]
            for position in range(take):
              record={"schedule":schedule_name,"prompt_id":prompt_id,"token_position":position,"position_bucket":_position_bucket(position),"category":item.get("category","unknown"),"source":item.get("source","unknown"),"label_token_id":int(labels[0,position]),"prompt_truncated":truncated,"forward_sequence_tokens":take+1,"dense_nll":float(dense_nll[position]),"sparse_nll":float(sparse_nll[position])};record.update({key:float(value[position]) for key,value in metrics.items()});rows.append(record)
            _append_rows(teacher/f"{schedule_name}.csv",rows)
            for row in monitor.rows:row.update({"schedule":schedule_name,"prompt_id":prompt_id})
            _append_rows(output_dir/f"hidden_state_drift_{schedule_name}.csv",monitor.rows)
            for row in local_rows:row.update({"schedule":schedule_name,"prompt_id":prompt_id})
            _append_rows(output_dir/f"ffn_local_error_{schedule_name}.csv",local_rows);sparse_peaks[schedule_name]=_rss_mib();monitor.rows.clear();del sparse,sparse_cpu,metrics,sparse_nll,rows
          evaluated+=take;monitor.release_prompt();del dense_cpu,dense_nll,dense_rows,labels,encoded;gc.collect()
          if torch.cuda.is_available():torch.cuda.empty_cache()
          rss_cleanup=_rss_mib();peak_rss=max(peak_rss,rss_before,rss_dense,rss_cleanup,*sparse_peaks.values());memory_record={"prompt_id":prompt_id,"evaluated_tokens":take,"total_evaluated_tokens":evaluated,"rss_before_prompt_mib":rss_before,"rss_after_dense_mib":rss_dense,"rss_after_sparse_mib":sparse_peaks,"rss_after_cleanup_mib":rss_cleanup,"post_model_baseline_rss_mib":baseline_rss,"peak_rss_mib":peak_rss};_append_rows(memory_path,[memory_record])
          print(json.dumps(memory_record));progress.update({"completed_prompts":prompt_id+1,"evaluated_tokens":evaluated,"peak_rss_mib":peak_rss});_atomic_json(progress_path,progress)
    finally:monitor.close();patcher.restore()
    run_metadata={name:{"status":"complete" if evaluated>=max_eval_tokens else "corpus_exhausted","tokens":evaluated,"wall_seconds":time.perf_counter()-started} for name in ["dense",*names]};run_metadata["memory"]={"post_model_baseline_rss_mib":baseline_rss,"peak_rss_mib":peak_rss};_atomic_json(output_dir/"run_metadata.json",run_metadata);return run_metadata


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
