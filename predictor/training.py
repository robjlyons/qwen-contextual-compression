from __future__ import annotations
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from predictor.datasets import prompt_split
from predictor.losses import distribution_ce,score_loss
from predictor.metrics import selection_metrics
from predictor.models import FactorizedPredictor,LowRankMLP
from predictor.output_aware import ensure_dense_output_cache,hard_topk_mask,output_loss,reconstruct,ste_topk_mask

OUTPUT_LOSSES={"output_cosine","output_relative","output_hybrid","output_hybrid_rank"}
def _validation(model,x,gated,dense,weight,bias,ids,mean,std,retention,device,batch_size):
 cos=[];rel=[]
 with torch.inference_mode():
  for batch in ids.split(batch_size):
   scores=model(((x[batch]-mean)/std).to(device));mask=hard_topk_mask(scores,retention);pred=reconstruct(gated[batch].to(device),mask,weight,bias);truth=dense[batch].to(device).float();cos.append(F.cosine_similarity(pred,truth,-1).cpu());rel.append((torch.linalg.vector_norm(pred-truth,dim=-1)/torch.linalg.vector_norm(truth,dim=-1).clamp_min(1e-12)).cpu())
 cosine=torch.cat(cos);relative=torch.cat(rel);return {"mean_cosine":float(cosine.mean()),"p01_cosine":float(cosine.quantile(.01)),"relative_l2":float(relative.mean()),"p99_relative_l2":float(relative.quantile(.99)),"score":float(cosine.mean()-.1*relative.mean())}
def train(target_dir:Path,run_dir:Path,kind="factorized",latent_dim=128,loss="distribution_ce",device="cpu",epochs=100,batch_size=16,lr=1e-3,weight_decay=1e-4,dropout=.1,patience=10,seed=42,train_retention=.5,temperature_start=1.,temperature_end=.1,ste_normalize=True,cosine_weight=1.,relative_weight=.25,ranking_weight=.05,amp=True):
 started=time.perf_counter();torch.manual_seed(seed);data=torch.load(target_dir/"targets.pt",map_location="cpu",weights_only=True);meta=[json.loads(x) for x in (target_dir/"sample_metadata.jsonl").read_text().splitlines()];split_path=run_dir.parent/"splits.json";run_dir.mkdir(parents=True,exist_ok=True)
 if split_path.exists():splits=json.loads(split_path.read_text())
 else:splits=prompt_split([m.get("prompt_ids",m.get("sample_ids",i)) for i,m in enumerate(meta)],seed);split_path.write_text(json.dumps(splits,indent=2)+"\n")
 x=data["inputs"].float();y=data["scores"].float();gated=data["gated_activations"];tr=torch.tensor(splits["train"]);vi=torch.tensor(splits["validation"]);mean=x[tr].mean(0);std=x[tr].std(0).clamp_min(1e-6);model=(FactorizedPredictor if kind=="factorized" else LowRankMLP)(x.shape[1],y.shape[1],latent_dim,dropout).to(device);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay);latest=run_dir/"latest.pt";start=0;best=-float("inf");wait=0
 output_aware=loss in OUTPUT_LOSSES;weight=bias=dense=None
 if output_aware:
  cache=ensure_dense_output_cache(target_dir,device,batch_size);dense=torch.load(cache,map_location="cpu",weights_only=True);down=torch.load(target_dir/"down_projection.pt",map_location="cpu",weights_only=True);weight=down["weight"].float().to(device);bias=None if down["bias"] is None else down["bias"].float().to(device);del down
 if latest.exists():state=torch.load(latest,map_location="cpu",weights_only=False);model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"]);start=state["epoch"]+1;best=state["best"]
 history=[]
 for epoch in range(start,epochs):
  model.train();perm=tr[torch.randperm(len(tr))];total=0.;temperature=temperature_start+(temperature_end-temperature_start)*(epoch/max(epochs-1,1))
  for ids in perm.split(batch_size):
   inputs=((x[ids]-mean)/std).to(device);target=y[ids].to(device)
   with torch.autocast(device_type=torch.device(device).type,dtype=torch.float16,enabled=amp and torch.device(device).type=="cuda"):pred=model(inputs)
   if output_aware:
    mask=ste_topk_mask(pred,train_retention,temperature,ste_normalize);sparse=reconstruct(gated[ids].to(device),mask,weight,bias);l=output_loss(sparse,dense[ids].to(device),loss,cosine_weight,relative_weight)
    if loss=="output_hybrid_rank":l=l+ranking_weight*distribution_ce(pred,target)
   else:l=distribution_ce(pred,target) if loss=="distribution_ce" else score_loss(pred,target,loss)
   opt.zero_grad();l.backward();opt.step();total+=float(l)*len(ids)
  model.eval();validation=_validation(model,x,gated,dense,weight,bias,vi,mean,std,train_retention,device,batch_size) if output_aware else {"captured_mass":float(selection_metrics(model(((x[vi]-mean)/std).to(device)),y[vi].to(device),train_retention)["captured_mass"].mean()),"score":float(selection_metrics(model(((x[vi]-mean)/std).to(device)),y[vi].to(device),train_retention)["captured_mass"].mean())};history.append({"epoch":epoch,"temperature":temperature,"train_loss":total/max(len(tr),1),**validation});config={"kind":kind,"latent_dim":latent_dim,"loss":loss,"seed":seed,"train_retention":train_retention,"ste_normalize":ste_normalize,"temperature_start":temperature_start,"temperature_end":temperature_end,"cosine_weight":cosine_weight,"relative_weight":relative_weight,"ranking_weight":ranking_weight,"amp":amp};state={"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":epoch,"best":max(best,validation["score"]),"mean":mean,"std":std,"config":config}
  torch.save(state,latest)
  if validation["score"]>best:best=validation["score"];wait=0;torch.save(state,run_dir/"best.pt")
  else:wait+=1
  if wait>=patience:break
 (run_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n");return {"best_validation_score":best,"epochs":len(history),"training_seconds":time.perf_counter()-started,"batch_size":batch_size,"splits":{k:len(v) for k,v in splits.items()}}
