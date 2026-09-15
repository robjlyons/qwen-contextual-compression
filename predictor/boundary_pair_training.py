"""Pairwise swap-supervised training for the unchanged boundary cascade."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from predictor.boundary_pairs import pairwise_swap_loss
from predictor.boundary_reranker import BoundarySwapCascade, BoundarySwapReranker
from predictor.boundary_training import _validation
from predictor.candidate_analysis import load_stage1
from predictor.output_aware import ensure_dense_output_cache
from predictor.reproducibility import epoch_permutation, set_global_seed, split_metadata


def _collate(records, ids, device):
    """Flatten variable-length token-owned pairs without changing the sample split."""
    selected=[];rejected=[];labels=[];weights=[];owners=[];gains=[]
    for owner, sample in enumerate(ids.tolist()):
        row=records[sample];count=len(row["gain"])
        selected.append(row["selected_ids"]);rejected.append(row["rejected_ids"]);labels.append(row["label"]);weights.append(row["weight"]);gains.append(row["gain"]);owners.append(torch.full((count,),owner,dtype=torch.long))
    return tuple(torch.cat(values).to(device) for values in (selected,rejected,labels,weights,owners,gains))


def _pair_diagnostics(cascade, x, records, ids, temperature, batch_size):
    margins=[];labels=[];gains=[]
    with torch.inference_mode():
        for batch in ids.split(batch_size):
            selected,rejected,label,_,owner,gain=_collate(records,batch,x.device)
            xb=x[batch];stage_scores=cascade.stage1(xb);_,boundary,boundary_scores,_=cascade.state(xb)
            i=cascade.reranker.score_ids(xb[owner],selected[:,None],stage_scores[owner],boundary_scores[owner]).squeeze(-1)
            j=cascade.reranker.score_ids(xb[owner],rejected[:,None],stage_scores[owner],boundary_scores[owner]).squeeze(-1)
            margins.append((j-i).cpu());labels.append(label.cpu());gains.append(gain.cpu())
    margin,label,gain=torch.cat(margins),torch.cat(labels).bool(),torch.cat(gains).float()
    correct=torch.where(label,margin>0,margin<=0)
    centered_margin=margin.float()-margin.float().mean();centered_gain=gain-gain.mean()
    pearson=float((centered_margin*centered_gain).mean()/(centered_margin.std(unbiased=False)*centered_gain.std(unbiased=False)).clamp_min(1e-12))
    return {"positive_pair_accuracy":float((margin[label]>0).float().mean()) if label.any() else None,"negative_pair_accuracy":float((margin[~label]<=0).float().mean()) if (~label).any() else None,"overall_pair_accuracy":float(correct.float().mean()),"positive_swap_capture_rate":float((margin[label]>0).float().mean()) if label.any() else None,"mean_positive_margin":float(margin[label].mean()) if label.any() else None,"mean_negative_margin":float(margin[~label].mean()) if (~label).any() else None,"margin_gain_pearson":pearson}


def train_boundary_pairwise(results_dir:Path,layer:int,stage1_run:str,pair_target_dir:Path,run_name:str,lock_retention=.45,candidate_retention=.65,final_retention=.5,rerank_dim=16,boundary_init="zero_embedding_alpha_one",loss="pairwise_swap",pair_temperature=1.,negative_weight=.5,epochs=50,patience=8,lr=5e-5,weight_decay=1e-4,batch_size=16,seed=42,device="cpu"):
    if loss!="pairwise_swap":raise ValueError("Phase A supports pairwise_swap only")
    if (lock_retention,final_retention,candidate_retention)!=(.45,.5,.65):raise ValueError("Phase 4.6 boundary configuration is fixed at 0.45/0.50/0.65")
    if rerank_dim!=16:raise ValueError("Phase 4.6 rerank_dim is fixed at 16")
    set_global_seed(seed);device=torch.device(device);layer_dir=Path(results_dir)/f"layer_{layer:03d}";target_dir=layer_dir/"targets";run_dir=layer_dir/run_name
    if run_dir.exists() and any(run_dir.iterdir()):raise FileExistsError(f"Refusing to overwrite pairwise run: {run_dir}")
    run_dir.mkdir(parents=True);stage1,stage_checkpoint,data,stage_hash=load_stage1(target_dir,layer_dir/stage1_run,device);cache=torch.load(Path(pair_target_dir)/"pairs.pt",map_location="cpu",weights_only=False);config_targets=cache["config"]
    expected=(stage1_run,stage_hash,lock_retention,final_retention,candidate_retention);actual=tuple(config_targets[k] for k in ("stage1_run","stage1_checkpoint_sha256","lock_retention","final_retention","candidate_retention"))
    if actual!=expected:raise ValueError(f"Pair target mismatch: expected {expected}, found {actual}")
    splits=cache["splits"]
    repository_splits=json.loads((layer_dir/"splits.json").read_text())
    if splits!=repository_splits:raise ValueError("Pair target sample splits do not match repository splits")
    tr=torch.tensor(splits["train"]);vi=torch.tensor(splits["validation"]);mean=stage_checkpoint["mean"].float();std=stage_checkpoint["std"].float().clamp_min(1e-6);x=((data["inputs"].float()-mean)/std).to(device);gated=data["gated_activations"].to(device);down=torch.load(target_dir/"down_projection.pt",map_location="cpu",weights_only=True);weight=down["weight"].to(device);bias=None if down["bias"] is None else down["bias"].to(device=device,dtype=weight.dtype);dense=torch.load(ensure_dense_output_cache(target_dir,device,batch_size),map_location="cpu",weights_only=True).to(device)
    reranker=BoundarySwapReranker(x.shape[-1],gated.shape[-1],rerank_dim,True,boundary_init).to(device);cascade=BoundarySwapCascade(stage1,reranker,lock_retention,final_retention,candidate_retention);optimizer=torch.optim.AdamW(reranker.parameters(),lr=lr,weight_decay=weight_decay);history=[];best=None;wait=0
    config={"kind":"boundary_pairwise_reranker","stage1_run":stage1_run,"stage1_checkpoint_sha256":stage_hash,"pair_target_dir":str(pair_target_dir),"lock_retention":lock_retention,"final_retention":final_retention,"candidate_retention":candidate_retention,"rerank_dim":rerank_dim,"boundary_init":boundary_init,"normalize_stage1_scores":True,"loss":loss,"pair_temperature":pair_temperature,"negative_weight":negative_weight,"checkpoint_objective":"validation_mean_cosine_minus_0.1_relative_l2","seed":seed,**split_metadata(splits)}
    for epoch in range(epochs):
        reranker.train();total=0.;pairs=0
        for batch in epoch_permutation(tr,seed,epoch).split(batch_size):
            selected,rejected,label,pair_weight,owner,_=_collate(cache["records"],batch,"cpu");selected=selected.to(device);rejected=rejected.to(device);label=label.to(device);pair_weight=pair_weight.to(device);owner=owner.to(device);xb=x[batch];stage_scores=stage1(xb);_,_,boundary_scores,_=cascade.state(xb)
            i=reranker.score_ids(xb[owner],selected[:,None],stage_scores[owner],boundary_scores[owner]).squeeze(-1);j=reranker.score_ids(xb[owner],rejected[:,None],stage_scores[owner],boundary_scores[owner]).squeeze(-1);objective=pairwise_swap_loss(i,j,label,pair_weight,pair_temperature,negative_weight);optimizer.zero_grad();objective.backward();optimizer.step();total+=float(objective.detach())*len(label);pairs+=len(label)
        reranker.eval();validation=_validation(cascade,x,gated,dense,weight,bias,vi,batch_size);diagnostics=_pair_diagnostics(cascade,x,cache["records"],vi,pair_temperature,batch_size);row={"epoch":epoch,"train_pair_loss":total/max(pairs,1),**validation,**diagnostics};history.append(row);state={"reranker":reranker.state_dict(),"config":config,"epoch":epoch,"validation":validation,"pair_diagnostics":diagnostics}
        torch.save(state,run_dir/"latest.pt")
        if best is None or validation["score"]>best["score"]:best=validation;wait=0;torch.save(state,run_dir/"best.pt")
        else:wait+=1
        if wait>=patience:break
    (run_dir/"config.json").write_text(json.dumps(config,indent=2)+"\n");(run_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n");return {"best_validation":best,"history":history,"epochs":len(history)}
