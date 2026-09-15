"""Direct output-aware training and evaluation of boundary swaps."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
from predictor.accounting import predictor_accounting
from predictor.boundary_reranker import BoundarySwapCascade,BoundarySwapReranker,boundary_accounting,boundary_mask,retention_count
from predictor.candidate_analysis import load_stage1,reconstruction_summary
from predictor.losses import distribution_ce
from predictor.metrics import reconstruct_metrics
from predictor.output_aware import ensure_dense_output_cache,output_loss,reconstruct
from predictor.reproducibility import epoch_permutation,set_global_seed,split_metadata


def _validation(cascade,x,gated,dense,weight,bias,ids,batch_size):
    cosines=[];relatives=[]
    with torch.inference_mode():
        for batch in ids.split(batch_size):
            final=cascade.final_indices(x[batch]);metrics=reconstruct_metrics(gated[batch],weight,final,bias);cosines.append(metrics["cosine_similarity"].cpu());relatives.append(metrics["relative_l2"].cpu())
    cosine,relative=torch.cat(cosines),torch.cat(relatives);return {"mean_cosine":float(cosine.mean()),"median_cosine":float(cosine.median()),"p05_cosine":float(cosine.quantile(.05)),"p01_cosine":float(cosine.quantile(.01)),"relative_l2":float(relative.mean()),"median_relative_l2":float(relative.median()),"p95_relative_l2":float(relative.quantile(.95)),"p99_relative_l2":float(relative.quantile(.99)),"score":float(cosine.mean()-.1*relative.mean())}


def train_boundary_reranker(results_dir:Path,layer:int,stage1_run:str,run_name:str,lock_retention:float,candidate_retention:float,final_retention=.5,rerank_dim=16,loss="output_hybrid",ranking_weight=0.,device="cpu",epochs=50,patience=8,batch_size=16,lr=5e-5,weight_decay=1e-4,cosine_weight=1.,relative_weight=.25,temperature=.5,normalize_stage1_scores=True,seed=42,init_checkpoint=None):
    if loss!="output_hybrid":raise ValueError("Phase 4.5 trains with output_hybrid only")
    if ranking_weight not in (0.,.02):raise ValueError("boundary ranking weight must be 0 or 0.02")
    set_global_seed(seed);device=torch.device(device);layer_dir=Path(results_dir)/f"layer_{layer:03d}";target_dir=layer_dir/"targets";run_dir=layer_dir/run_name
    if run_dir.exists() and any(run_dir.iterdir()):raise FileExistsError(f"Refusing to overwrite boundary run: {run_dir}")
    run_dir.mkdir(parents=True);stage1,stage1_checkpoint,data,stage1_hash=load_stage1(target_dir,layer_dir/stage1_run,device);splits=json.loads((layer_dir/"splits.json").read_text());tr=torch.tensor(splits["train"]);vi=torch.tensor(splits["validation"]);mean=stage1_checkpoint["mean"].float();std=stage1_checkpoint["std"].float().clamp_min(1e-6);x=((data["inputs"].float()-mean)/std).to(device);oracle=data["raw_scores"].float().to(device);gated=data["gated_activations"].to(device);reranker=BoundarySwapReranker(x.shape[-1],oracle.shape[-1],rerank_dim,normalize_stage1_scores).to(device);cascade=BoundarySwapCascade(stage1,reranker,lock_retention,final_retention,candidate_retention)
    if init_checkpoint:
        source=torch.load(init_checkpoint,map_location="cpu",weights_only=False);keys=("lock_retention","candidate_retention","final_retention","rerank_dim","stage1_checkpoint_sha256");expected=(lock_retention,candidate_retention,final_retention,rerank_dim,stage1_hash);actual=tuple(source["config"][key] for key in keys)
        if actual!=expected:raise ValueError(f"Boundary checkpoint mismatch: expected {expected}, found {actual}")
        reranker.load_state_dict(source["reranker"],strict=True)
    dense=torch.load(ensure_dense_output_cache(target_dir,device,batch_size),map_location="cpu",weights_only=True).to(device);down=torch.load(target_dir/"down_projection.pt",map_location="cpu",weights_only=True);weight=down["weight"].float().to(device);bias=None if down["bias"] is None else down["bias"].float().to(device);optimizer=torch.optim.AdamW(reranker.parameters(),lr=lr,weight_decay=weight_decay);initial=_validation(cascade,x,gated,dense,weight,bias,vi,batch_size);history=[{"epoch":-1,"alpha":float(reranker.alpha.detach()),**initial}];best=None;wait=0;started=time.perf_counter()
    for epoch in range(epochs):
        reranker.train();total=0.
        for ids in epoch_permutation(tr,seed,epoch).split(batch_size):
            locked,boundary,soft=cascade.final_indices(x[ids],ste=True,temperature=temperature);mask=boundary_mask(locked,boundary,soft,oracle.shape[-1]);sparse=reconstruct(gated[ids],mask,weight,bias);objective=output_loss(sparse,dense[ids],"output_hybrid",cosine_weight,relative_weight)
            if ranking_weight:objective=objective+ranking_weight*distribution_ce(reranker(x[ids],boundary,stage1(x[ids]).gather(-1,boundary)),oracle[ids].gather(-1,boundary))
            optimizer.zero_grad();objective.backward();optimizer.step();total+=float(objective.detach())*len(ids)
        reranker.eval();validation=_validation(cascade,x,gated,dense,weight,bias,vi,batch_size);history.append({"epoch":epoch,"train_loss":total/len(tr),"alpha":float(reranker.alpha.detach()),**validation});config={"kind":"boundary_swap_reranker","stage1_architecture":stage1_checkpoint["config"]["kind"],"stage1_run":stage1_run,"stage1_checkpoint_sha256":stage1_hash,"lock_retention":lock_retention,"candidate_retention":candidate_retention,"final_retention":final_retention,"rerank_dim":rerank_dim,"normalize_stage1_scores":normalize_stage1_scores,"loss":loss,"ranking_weight":ranking_weight,"seed":seed,"normalization_source":"stage1_checkpoint",**split_metadata(splits)};state={"reranker":reranker.state_dict(),"config":config,"epoch":epoch,"validation":validation}
        torch.save(state,run_dir/"latest.pt")
        if best is None or validation["score"]>best["score"]:best=validation;wait=0;torch.save(state,run_dir/"best.pt")
        else:wait+=1
        if wait>=patience:break
    (run_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n");(run_dir/"config.json").write_text(json.dumps(config,indent=2)+"\n");return {"initial_validation":initial,"best_validation":best,"learned_alpha":float(reranker.alpha.detach()),"epochs":len(history)-1,"seconds":time.perf_counter()-started}


def evaluate_boundary_reranker(results_dir:Path,layer:int,run_name:str,device="cpu",split_name="test"):
    layer_dir=Path(results_dir)/f"layer_{layer:03d}";run_dir=layer_dir/run_name;path=run_dir/"best.pt"
    if not path.is_file():raise FileNotFoundError(f"Missing boundary checkpoint: {path}")
    checkpoint=torch.load(path,map_location="cpu",weights_only=False);config=checkpoint["config"];target_dir=layer_dir/"targets";stage1,stage1_checkpoint,data,stage1_hash=load_stage1(target_dir,layer_dir/config["stage1_run"],device)
    if stage1_hash!=config["stage1_checkpoint_sha256"]:raise ValueError("Referenced Stage-1 checkpoint hash has changed")
    reranker=BoundarySwapReranker(data["inputs"].shape[-1],data["raw_scores"].shape[-1],config["rerank_dim"],config["normalize_stage1_scores"]);reranker.load_state_dict(checkpoint["reranker"],strict=True);reranker.to(device).eval();cascade=BoundarySwapCascade(stage1,reranker,config["lock_retention"],config["final_retention"],config["candidate_retention"]);splits=json.loads((layer_dir/"splits.json").read_text());ids=torch.tensor(splits[split_name]);mean=stage1_checkpoint["mean"].float();std=stage1_checkpoint["std"].float().clamp_min(1e-6);x=((data["inputs"].index_select(0,ids).float()-mean)/std).to(device);oracle=data["raw_scores"].index_select(0,ids).float().to(device);gated=data["gated_activations"].index_select(0,ids).to(device);down=torch.load(target_dir/"down_projection.pt",map_location="cpu",weights_only=True);weight=down["weight"].to(device);bias=None if down["bias"] is None else down["bias"].to(device);final_k=retention_count(oracle.shape[-1],config["final_retention"])
    with torch.inference_mode():
        final=cascade.final_indices(x);stage1_scores=stage1(x);stage1_final=torch.topk(stage1_scores,final_k,-1,sorted=False).indices;metrics=reconstruct_metrics(gated,weight,final,bias);baseline=reconstruct_metrics(gated,weight,stage1_final,bias);summary=reconstruction_summary(metrics);stage1_summary=reconstruction_summary(baseline);final_bool=torch.zeros_like(oracle,dtype=torch.bool).scatter_(-1,final,True);base_bool=torch.zeros_like(final_bool).scatter_(-1,stage1_final,True);added=(final_bool&~base_bool).sum(-1).float();oracle_final=torch.topk(oracle,final_k,-1,sorted=False).indices;intersection=final_bool.gather(-1,oracle_final).sum(-1).float();mass=oracle.gather(-1,final).sum(-1)/oracle.sum(-1).clamp_min(1e-12);dc=metrics["cosine_similarity"]-baseline["cosine_similarity"];dl=metrics["relative_l2"]-baseline["relative_l2"];sorted_scores=torch.sort(stage1_scores,descending=True).values;margin=sorted_scores[:,final_k-1]-sorted_scores[:,final_k];summary.update(captured_mass=float(mass.mean()),oracle_top50_recall=float((intersection/final_k).mean()),oracle_jaccard=float((intersection/(2*final_k-intersection)).mean()),fraction_cosine_improved=float((dc>0).float().mean()),fraction_cosine_worsened=float((dc<0).float().mean()),fraction_l2_improved=float((dl<0).float().mean()),fraction_l2_worsened=float((dl>0).float().mean()),median_cosine_delta=float(dc.median()),median_l2_delta=float(dl.median()),mean_added=float(added.mean()),mean_removed=float(added.mean()),mean_changed_neurons=float((2*added).mean()),median_changed_neurons=float((2*added).median()),p95_changed_neurons=float((2*added).quantile(.95)),mean_swap_fraction=float((2*added).mean()/final_k),mean_cutoff_margin=float(margin.mean()),improved_cutoff_margin=float(margin[dc>0].mean()) if (dc>0).any() else None,worsened_cutoff_margin=float(margin[dc<0].mean()) if (dc<0).any() else None,learned_alpha=float(reranker.alpha))
    stage1_cost=predictor_accounting(stage1,x.shape[-1],oracle.shape[-1],config["final_retention"]);accounting=boundary_accounting(stage1_cost["macs_per_token"],stage1_cost["parameters"],x.shape[-1],oracle.shape[-1],config["lock_retention"],config["candidate_retention"],config["rerank_dim"],stage1_cost["dense_ffn_macs"]);result={"selection_split":split_name,"config":config,"metrics":summary,"stage1_metrics":stage1_summary,"accounting":accounting};(run_dir/"evaluation.json").write_text(json.dumps(result,indent=2)+"\n");return result
