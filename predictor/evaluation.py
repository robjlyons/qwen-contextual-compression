import json,time
from pathlib import Path
import torch
from predictor.datasets import prompt_split
from predictor.accounting import predictor_accounting
from predictor.metrics import reconstruct_metrics,selection_metrics,select_topk
from predictor.models import FactorizedPredictor,LowRankMLP,StaticHot
def benchmark(model,x,device,warmup=10,repeats=100):
 model=model.to(device);x=x[:1].to(device)
 for _ in range(warmup):model(x)
 values=[]
 if str(device).startswith("cuda"):
  for _ in range(repeats):a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();model(x);b.record();torch.cuda.synchronize();values.append(a.elapsed_time(b))
 else:
  for _ in range(repeats):a=time.perf_counter();model(x);values.append((time.perf_counter()-a)*1000)
 t=torch.tensor(values);return {"warmup":warmup,"median_ms":float(t.median()),"p90_ms":float(t.quantile(.9)),"p95_ms":float(t.quantile(.95))}
def evaluate(target_dir:Path,run_dir:Path,retentions,device="cpu"):
 data=torch.load(target_dir/"targets.pt",map_location="cpu",weights_only=True);down=torch.load(target_dir/"down_projection.pt",map_location=device,weights_only=True);split=json.loads((run_dir.parent/"splits.json").read_text());checkpoint=torch.load(run_dir/"best.pt",map_location=device,weights_only=False);c=checkpoint["config"];cls=FactorizedPredictor if c["kind"]=="factorized" else LowRankMLP;model=cls(data["inputs"].shape[1],data["scores"].shape[1],c["latent_dim"]).to(device);model.load_state_dict(checkpoint["model"]);ids=torch.tensor(split["test"]);x=((data["inputs"][ids].float()-checkpoint["mean"])/checkpoint["std"]).to(device);target=data["raw_scores"][ids].float().to(device);activations=data["gated_activations"][ids].to(device);pred=model(x);static=StaticHot().fit(data["raw_scores"][torch.tensor(split["train"])].float()).scores.to(device);rows=[]
 for r in retentions:
  for method,ranking in (("predictor",pred),("oracle",target),("static",static.expand_as(target))):
   m=selection_metrics(ranking,target,r);recon=reconstruct_metrics(activations,down["weight"],m["indices"],down["bias"]);row={"method":method,"retention":r,**{k:float(v.mean()) for k,v in m.items() if k!="indices"},"ffn_cosine":float(recon["cosine_similarity"].mean()),"ffn_cosine_p01":float(recon["cosine_similarity"].quantile(.01)),"ffn_cosine_p05":float(recon["cosine_similarity"].quantile(.05)),"relative_l2":float(recon["relative_l2"].mean()),"relative_l2_p95":float(recon["relative_l2"].quantile(.95)),"relative_l2_p99":float(recon["relative_l2"].quantile(.99)),**predictor_accounting(model,x.shape[1],target.shape[1],r)};rows.append(row)
 (run_dir/"metrics.json").write_text(json.dumps({"rows":rows,"latency":benchmark(model,data["inputs"].float(),device)},indent=2)+"\n");return rows
def evaluate_static(target_dir:Path,run_dir:Path,retentions):
 data=torch.load(target_dir/"targets.pt",map_location="cpu",weights_only=True);split_path=run_dir.parent/"splits.json"
 if split_path.exists():split=json.loads(split_path.read_text())
 else:
  meta=[json.loads(x) for x in (target_dir/"sample_metadata.jsonl").read_text().splitlines()];split=prompt_split([m.get("prompt_ids",m.get("sample_ids",i)) for i,m in enumerate(meta)]);split_path.write_text(json.dumps(split,indent=2)+"\n")
 train=torch.tensor(split["train"]);test=torch.tensor(split["test"]);model=StaticHot().fit(data["raw_scores"][train].float());rows=[]
 for r in retentions:
  m=selection_metrics(model(data["inputs"][test]),data["raw_scores"][test].float(),r);rows.append({"method":"static","retention":r,**{k:float(v.mean()) for k,v in m.items() if k!="indices"},"parameters":0,"macs_per_token":0})
 run_dir.mkdir(parents=True,exist_ok=True);(run_dir/"metrics.json").write_text(json.dumps({"rows":rows,"latency":{"not_applicable":"static indices are precomputed"}},indent=2)+"\n");return rows
