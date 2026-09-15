"""Training and evaluation for the frozen Stage-1 candidate cascade."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import torch
from predictor.candidate_analysis import load_stage1,reconstruction_summary
from predictor.candidate_reranker import CandidateCascade,CandidateReranker,candidate_mask,hard_topk_count,reranker_accounting,retention_count,ste_topk_count
from predictor.losses import distribution_ce
from predictor.metrics import reconstruct_metrics
from predictor.output_aware import ensure_dense_output_cache,output_loss,reconstruct
from predictor.accounting import predictor_accounting
from predictor.reproducibility import epoch_permutation,set_global_seed,split_metadata


def _reranker_checkpoint_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hard_candidate_indices(candidate_ids,candidate_scores,final_k):
    return candidate_ids.gather(-1,torch.topk(candidate_scores,final_k,-1,sorted=False).indices)


def _validate(cascade,x,oracle,gated,dense,weight,bias,ids,batch_size,output_aware):
    loss_sum=0.;sample_count=0;cos=[];rel=[]
    with torch.inference_mode():
        for batch in ids.split(batch_size):
            candidates,base=cascade.candidate_state(x[batch]);scores=cascade.reranker(x[batch],candidates,base);target=oracle[batch].gather(-1,candidates);loss_sum+=float(distribution_ce(scores,target))*len(batch);sample_count+=len(batch);final=_hard_candidate_indices(candidates,scores,retention_count(oracle.shape[-1],cascade.final_retention))
            if output_aware:
                metrics=reconstruct_metrics(gated[batch],weight,final,bias);cos.append(metrics["cosine_similarity"].cpu());rel.append(metrics["relative_l2"].cpu())
    validation_ce=loss_sum/sample_count
    if not output_aware:return {"validation_distribution_ce":validation_ce,"score":-validation_ce}
    cosine,relative=torch.cat(cos),torch.cat(rel);return {"validation_distribution_ce":validation_ce,"mean_cosine":float(cosine.mean()),"p01_cosine":float(cosine.quantile(.01)),"p05_cosine":float(cosine.quantile(.05)),"relative_l2":float(relative.mean()),"p95_relative_l2":float(relative.quantile(.95)),"p99_relative_l2":float(relative.quantile(.99)),"score":float(cosine.mean()-.1*relative.mean())}


def train_candidate_reranker(results_dir:Path,layer:int,stage1_run:str,run_name:str,candidate_retention=.6,final_retention=.5,rerank_dim=16,loss="distribution_ce",device="cpu",epochs=50,patience=8,batch_size=16,lr=None,weight_decay=1e-4,ranking_weight=.1,cosine_weight=1.,relative_weight=.25,temperature=0.5,normalize_stage1_scores=True,seed=42,init_checkpoint=None):
    if candidate_retention<=final_retention:raise ValueError("candidate retention must be greater than final retention")
    set_global_seed(seed);device=torch.device(device);layer_dir=Path(results_dir)/f"layer_{layer:03d}";target_dir=layer_dir/"targets";run_dir=layer_dir/run_name
    if run_dir.exists() and any(run_dir.iterdir()):raise FileExistsError(f"Refusing to overwrite candidate reranker run: {run_dir}")
    run_dir.mkdir(parents=True);stage1,stage1_checkpoint,data,stage1_hash=load_stage1(target_dir,layer_dir/stage1_run,device);splits=json.loads((layer_dir/"splits.json").read_text());tr=torch.tensor(splits["train"]);vi=torch.tensor(splits["validation"]);mean=stage1_checkpoint["mean"].float();std=stage1_checkpoint["std"].float().clamp_min(1e-6);x=((data["inputs"].float()-mean)/std).to(device);oracle=data["raw_scores"].float().to(device);gated=data["gated_activations"].to(device);reranker=CandidateReranker(x.shape[-1],oracle.shape[-1],rerank_dim,normalize_stage1_scores).to(device)
    if init_checkpoint:
        source=torch.load(init_checkpoint,map_location="cpu",weights_only=False);expected=(candidate_retention,final_retention,rerank_dim,stage1_hash);actual=(source["config"]["candidate_retention"],source["config"]["final_retention"],source["config"]["rerank_dim"],source["config"]["stage1_checkpoint_sha256"])
        if actual!=expected:raise ValueError(f"Candidate reranker checkpoint mismatch: expected {expected}, found {actual}")
        reranker.load_state_dict(source["reranker"],strict=True)
    cascade=CandidateCascade(stage1,reranker,candidate_retention,final_retention);effective_lr=lr if lr is not None else (5e-5 if init_checkpoint else 1e-3);optimizer=torch.optim.AdamW(reranker.parameters(),lr=effective_lr,weight_decay=weight_decay);output_aware=loss=="output_hybrid_rank";dense=weight=bias=None
    if output_aware:
        dense=torch.load(ensure_dense_output_cache(target_dir,device,batch_size),map_location="cpu",weights_only=True).to(device);down=torch.load(target_dir/"down_projection.pt",map_location="cpu",weights_only=True);weight=down["weight"].float().to(device);bias=None if down["bias"] is None else down["bias"].float().to(device)
    initial=_validate(cascade,x,oracle,gated,dense,weight,bias,vi,batch_size,output_aware);history=[{"epoch":-1,**initial}];best=None;wait=0;started=time.perf_counter();final_k=retention_count(oracle.shape[-1],final_retention)
    for epoch in range(epochs):
        reranker.train();total=0.
        for ids in epoch_permutation(tr,seed,epoch).split(batch_size):
            candidates,base=cascade.candidate_state(x[ids]);scores=reranker(x[ids],candidates,base);target=oracle[ids].gather(-1,candidates);objective=distribution_ce(scores,target)
            if output_aware:
                soft=ste_topk_count(scores,final_k,temperature);full_mask=candidate_mask(candidates,soft,oracle.shape[-1]);sparse=reconstruct(gated[ids],full_mask,weight,bias);objective=output_loss(sparse,dense[ids],"output_hybrid",cosine_weight,relative_weight)+ranking_weight*distribution_ce(scores,target)
            optimizer.zero_grad();objective.backward();optimizer.step();total+=float(objective.detach())*len(ids)
        reranker.eval();validation=_validate(cascade,x,oracle,gated,dense,weight,bias,vi,batch_size,output_aware);history.append({"epoch":epoch,"train_loss":total/len(tr),**validation})
        config={"kind":"candidate_reranker","stage1_architecture":stage1_checkpoint["config"]["kind"],"stage1_run":stage1_run,"stage1_checkpoint":str(layer_dir/stage1_run/"best.pt"),"stage1_checkpoint_sha256":stage1_hash,"candidate_retention":candidate_retention,"final_retention":final_retention,"rerank_dim":rerank_dim,"normalize_stage1_scores":normalize_stage1_scores,"loss":loss,"ranking_weight":ranking_weight,"cosine_weight":cosine_weight,"relative_weight":relative_weight,"seed":seed,"normalization_source":"stage1_checkpoint","init_checkpoint":str(init_checkpoint) if init_checkpoint else None,**split_metadata(splits)};state={"reranker":reranker.state_dict(),"config":config,"epoch":epoch,"validation":validation}
        torch.save(state,run_dir/"latest.pt")
        if best is None or validation["score"]>best["score"]:best=validation;wait=0;torch.save(state,run_dir/"best.pt")
        else:wait+=1
        if wait>=patience:break
    (run_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n");(run_dir/"config.json").write_text(json.dumps(config,indent=2)+"\n");return {"initial_validation":initial,"best_validation":best,"epochs":len(history)-1,"seconds":time.perf_counter()-started}


def evaluate_candidate_reranker(results_dir:Path,layer:int,run_name:str,device="cpu",split_name="test"):
    layer_dir=Path(results_dir)/f"layer_{layer:03d}";run_dir=layer_dir/run_name;path=run_dir/"best.pt"
    if not path.is_file():raise FileNotFoundError(f"Missing candidate reranker checkpoint: {path}")
    checkpoint=torch.load(path,map_location="cpu",weights_only=False);config=checkpoint["config"];target_dir=layer_dir/"targets";stage1,stage1_checkpoint,data,stage1_hash=load_stage1(target_dir,layer_dir/config["stage1_run"],device)
    if stage1_hash!=config["stage1_checkpoint_sha256"]:raise ValueError("Referenced Stage-1 checkpoint hash has changed")
    reranker=CandidateReranker(data["inputs"].shape[-1],data["raw_scores"].shape[-1],config["rerank_dim"],config["normalize_stage1_scores"]);reranker.load_state_dict(checkpoint["reranker"],strict=True);reranker.to(device).eval();cascade=CandidateCascade(stage1,reranker,config["candidate_retention"],config["final_retention"]);splits=json.loads((layer_dir/"splits.json").read_text());ids=torch.tensor(splits[split_name]);mean=stage1_checkpoint["mean"].float();std=stage1_checkpoint["std"].float().clamp_min(1e-6);x=((data["inputs"].index_select(0,ids).float()-mean)/std).to(device);oracle=data["raw_scores"].index_select(0,ids).float().to(device);gated=data["gated_activations"].index_select(0,ids).to(device);down=torch.load(target_dir/"down_projection.pt",map_location="cpu",weights_only=True);weight=down["weight"].to(device);bias=None if down["bias"] is None else down["bias"].to(device);final_k=retention_count(oracle.shape[-1],config["final_retention"])
    with torch.inference_mode():
        candidates,base=cascade.candidate_state(x);reranked=reranker(x,candidates,base);final=_hard_candidate_indices(candidates,reranked,final_k);stage1_final=torch.topk(stage1(x),final_k,-1,sorted=False).indices;metrics=reconstruct_metrics(gated,weight,final,bias);stage1_metrics=reconstruct_metrics(gated,weight,stage1_final,bias);summary=reconstruction_summary(metrics);stage1_summary=reconstruction_summary(stage1_metrics);oracle_final=torch.topk(oracle,final_k,-1,sorted=False).indices;candidate_bool=torch.zeros_like(oracle,dtype=torch.bool).scatter_(-1,candidates,True);final_bool=torch.zeros_like(oracle,dtype=torch.bool).scatter_(-1,final,True);oracle_present=candidate_bool.gather(-1,oracle_final);oracle_selected=final_bool.gather(-1,oracle_final);missed=(~oracle_selected);omission=((~oracle_present)&missed).sum().float()/missed.sum().clamp_min(1);ranking=(oracle_present&missed).sum().float()/missed.sum().clamp_min(1);mass=oracle.gather(-1,final).sum(-1)/oracle.sum(-1).clamp_min(1e-12);summary["captured_mass"]=float(mass.mean());summary["oracle_top50_recall"]=float(oracle_selected.float().mean());intersection=oracle_selected.sum(-1).float();summary["oracle_jaccard"]=float((intersection/(2*final_k-intersection)).mean());delta_cos=metrics["cosine_similarity"]-stage1_metrics["cosine_similarity"];delta_l2=metrics["relative_l2"]-stage1_metrics["relative_l2"];summary.update(fraction_cosine_improved=float((delta_cos>0).float().mean()),fraction_cosine_worsened=float((delta_cos<0).float().mean()),fraction_l2_improved=float((delta_l2<0).float().mean()),median_cosine_delta=float(delta_cos.median()),median_l2_delta=float(delta_l2.median()),candidate_omission_share=float(omission),candidate_ranking_share=float(ranking))
    stage1_accounting=predictor_accounting(stage1,x.shape[-1],oracle.shape[-1],config["final_retention"]);accounting=reranker_accounting(stage1_accounting["macs_per_token"],stage1_accounting["parameters"],x.shape[-1],oracle.shape[-1],config["candidate_retention"],config["rerank_dim"],stage1_accounting["dense_ffn_macs"]);result={"selection_split":split_name,"config":config,"metrics":summary,"stage1_metrics":stage1_summary,"accounting":accounting};(run_dir/"evaluation.json").write_text(json.dumps(result,indent=2)+"\n");return result
