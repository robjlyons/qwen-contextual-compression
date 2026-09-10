"""Device-explicit predictor evaluation and reconstruction."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
from predictor.datasets import prompt_split
from predictor.accounting import predictor_accounting
from predictor.metrics import reconstruct_metrics,selection_metrics
from predictor.models import FactorizedPredictor,LowRankMLP,StaticHot

def benchmark(model,x,device,warmup=10,repeats=100):
 device=torch.device(device);model=model.to(device);x=x[:1].to(device)
 with torch.inference_mode():
  for _ in range(warmup):model(x)
  values=[]
  if device.type=="cuda":
   for _ in range(repeats):a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();model(x);b.record();torch.cuda.synchronize(device);values.append(a.elapsed_time(b))
  else:
   for _ in range(repeats):a=time.perf_counter();model(x);values.append((time.perf_counter()-a)*1000)
 t=torch.tensor(values);return {"warmup":warmup,"median_ms":float(t.median()),"p90_ms":float(t.quantile(.9)),"p95_ms":float(t.quantile(.95))}

def _required(*paths:Path):
 for path in paths:
  if not path.is_file():raise FileNotFoundError(f"Missing predictor evaluation file: {path}")

def normalise_inputs(inputs,mean,std,device):
 """Select on CPU first; move statistics only after the input device is known."""
 x=inputs.float().to(device);mean=mean.to(device=x.device,dtype=x.dtype);std=std.to(device=x.device,dtype=x.dtype).clamp_min(1e-6);return (x-mean)/std,mean,std

def prepare_evaluation(target_dir:Path,run_dir:Path,device="cpu"):
 target_path=target_dir/"targets.pt";down_path=target_dir/"down_projection.pt";split_path=run_dir.parent/"splits.json";checkpoint_path=run_dir/"best.pt";_required(target_path,down_path,split_path,checkpoint_path);device=torch.device(device)
 data=torch.load(target_path,map_location="cpu",weights_only=True);checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False);split=json.loads(split_path.read_text());config=checkpoint["config"];cls=FactorizedPredictor if config["kind"]=="factorized" else LowRankMLP;model=cls(data["inputs"].shape[1],data["scores"].shape[1],config["latent_dim"]);model.load_state_dict(checkpoint["model"]);model=model.to(device).eval()
 test_ids=torch.tensor(split["test"],dtype=torch.long,device="cpu");train_ids=torch.tensor(split["train"],dtype=torch.long,device="cpu");selected=data["inputs"].index_select(0,test_ids);x,mean,std=normalise_inputs(selected,checkpoint["mean"],checkpoint["std"],device);target=data["raw_scores"].index_select(0,test_ids).float().to(device);activations=data["gated_activations"].index_select(0,test_ids).to(device);static=StaticHot().fit(data["raw_scores"].index_select(0,train_ids).float()).scores.to(device)
 down_cpu=torch.load(down_path,map_location="cpu",weights_only=True);down_weight=down_cpu["weight"].to(device);down_bias=None if down_cpu["bias"] is None else down_cpu["bias"].to(device);del down_cpu
 tensors={"x":x,"mean":mean,"std":std,"target":target,"activations":activations,"static":static,"down_weight":down_weight,"down_bias":down_bias}
 expected=x.device;mismatches={name:str(value.device) for name,value in tensors.items() if isinstance(value,torch.Tensor) and value.device!=expected}
 if next(model.parameters()).device!=expected:mismatches["model"]=str(next(model.parameters()).device)
 if mismatches:raise RuntimeError(f"Predictor evaluation device mismatch; expected {expected}: {mismatches}")
 diagnostics={"device":str(device),"model_device":str(next(model.parameters()).device),"inputs_source_device":str(data["inputs"].device),"x_compute_device":str(x.device),"mean_device":str(mean.device),"std_device":str(std.device),"target_device":str(target.device),"gated_activations_device":str(activations.device),"down_projection_device":str(down_weight.device),"test_samples":len(test_ids)};print("Predictor evaluation\n"+json.dumps(diagnostics,indent=2));return data,model,tensors,diagnostics

def evaluate(target_dir:Path,run_dir:Path,retentions,device="cpu"):
 data,model,t,diagnostics=prepare_evaluation(target_dir,run_dir,device);rows=[]
 with torch.inference_mode():
  pred=model(t["x"])
  for r in retentions:
   for method,ranking in (("predictor",pred),("oracle",t["target"]),("static",t["static"].expand_as(t["target"]))):
    m=selection_metrics(ranking,t["target"],r);recon=reconstruct_metrics(t["activations"],t["down_weight"],m["indices"],t["down_bias"]);row={"method":method,"retention":r,**{k:float(v.mean()) for k,v in m.items() if k!="indices"},"ffn_cosine":float(recon["cosine_similarity"].mean()),"ffn_cosine_p01":float(recon["cosine_similarity"].quantile(.01)),"ffn_cosine_p05":float(recon["cosine_similarity"].quantile(.05)),"relative_l2":float(recon["relative_l2"].mean()),"relative_l2_p95":float(recon["relative_l2"].quantile(.95)),"relative_l2_p99":float(recon["relative_l2"].quantile(.99)),**predictor_accounting(model,t["x"].shape[1],t["target"].shape[1],r)};rows.append(row)
 (run_dir/"metrics.json").write_text(json.dumps({"device_diagnostics":diagnostics,"rows":rows,"latency":benchmark(model,data["inputs"].float(),device)},indent=2)+"\n");return rows

def evaluate_static(target_dir:Path,run_dir:Path,retentions):
 target_path=target_dir/"targets.pt";meta_path=target_dir/"sample_metadata.jsonl";_required(target_path,meta_path);data=torch.load(target_path,map_location="cpu",weights_only=True);split_path=run_dir.parent/"splits.json"
 if split_path.exists():split=json.loads(split_path.read_text())
 else:
  meta=[json.loads(x) for x in meta_path.read_text().splitlines()];split=prompt_split([m.get("prompt_ids",m.get("sample_ids",i)) for i,m in enumerate(meta)]);split_path.write_text(json.dumps(split,indent=2)+"\n")
 train=torch.tensor(split["train"],dtype=torch.long,device="cpu");test=torch.tensor(split["test"],dtype=torch.long,device="cpu");model=StaticHot().fit(data["raw_scores"].index_select(0,train).float());rows=[]
 for r in retentions:
  m=selection_metrics(model(data["inputs"].index_select(0,test)),data["raw_scores"].index_select(0,test).float(),r);rows.append({"method":"static","retention":r,**{k:float(v.mean()) for k,v in m.items() if k!="indices"},"parameters":0,"macs_per_token":0})
 run_dir.mkdir(parents=True,exist_ok=True);(run_dir/"metrics.json").write_text(json.dumps({"rows":rows,"latency":{"not_applicable":"static indices are precomputed"}},indent=2)+"\n");return rows
