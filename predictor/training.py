from __future__ import annotations
import json,time
from pathlib import Path
import torch
from predictor.datasets import prompt_split
from predictor.losses import distribution_ce,score_loss
from predictor.metrics import selection_metrics
from predictor.models import FactorizedPredictor,LowRankMLP
def train(target_dir:Path,run_dir:Path,kind="factorized",latent_dim=128,loss="distribution_ce",device="cpu",epochs=100,batch_size=32,lr=1e-3,weight_decay=1e-4,dropout=.1,patience=10,seed=42):
 torch.manual_seed(seed);data=torch.load(target_dir/"targets.pt",map_location="cpu",weights_only=True);meta=[json.loads(x) for x in (target_dir/"sample_metadata.jsonl").read_text().splitlines()];split_path=run_dir.parent/"splits.json";run_dir.mkdir(parents=True,exist_ok=True)
 if split_path.exists():splits=json.loads(split_path.read_text())
 else:splits=prompt_split([m.get("prompt_ids",m.get("sample_ids",i)) for i,m in enumerate(meta)],seed);split_path.write_text(json.dumps(splits,indent=2)+"\n")
 x=data["inputs"].float();y=data["scores"].float();tr=torch.tensor(splits["train"]);mean=x[tr].mean(0);std=x[tr].std(0).clamp_min(1e-6);model=(FactorizedPredictor if kind=="factorized" else LowRankMLP)(x.shape[1],y.shape[1],latent_dim,dropout).to(device);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay);latest=run_dir/"latest.pt";start=0;best=-1.;wait=0
 if latest.exists():state=torch.load(latest,map_location=device,weights_only=False);model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"]);start=state["epoch"]+1;best=state["best"]
 history=[]
 for epoch in range(start,epochs):
  model.train();perm=tr[torch.randperm(len(tr))];total=0.
  for ids in perm.split(batch_size):pred=model(((x[ids]-mean)/std).to(device));target=y[ids].to(device);l=distribution_ce(pred,target) if loss=="distribution_ce" else score_loss(pred,target,loss);opt.zero_grad();l.backward();opt.step();total+=float(l)*len(ids)
  model.eval();vi=torch.tensor(splits["validation"]);captured=float(selection_metrics(model(((x[vi]-mean)/std).to(device)),y[vi].to(device),.5)["captured_mass"].mean()) if len(vi) else 0.;history.append({"epoch":epoch,"train_loss":total/max(len(tr),1),"validation_captured_mass":captured});state={"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":epoch,"best":max(best,captured),"mean":mean,"std":std,"config":{"kind":kind,"latent_dim":latent_dim,"loss":loss,"seed":seed}}
  torch.save(state,latest)
  if captured>best:best=captured;wait=0;torch.save(state,run_dir/"best.pt")
  else:wait+=1
  if wait>=patience:break
 (run_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n");return {"best_validation_captured_mass":best,"epochs":len(history),"splits":{k:len(v) for k,v in splits.items()}}
