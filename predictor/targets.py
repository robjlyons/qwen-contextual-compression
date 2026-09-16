from __future__ import annotations
import json,math
from pathlib import Path
import torch
from extract.extract_ffn import gated_activations
from oracle.importance import importance_scores
from end_to_end.oracle_sparse_mlp import compute_down_column_norms_chunked

TRANSFORMS=("distribution","max","log1p","rank")
def transform_scores(scores,kind):
 if kind=="distribution":return scores/scores.sum(-1,keepdim=True).clamp_min(1e-12)
 if kind=="max":return scores/scores.amax(-1,keepdim=True).clamp_min(1e-12)
 if kind=="log1p":return torch.log1p(scores)
 if kind=="rank":return scores.argsort(-1).argsort(-1).float()/max(scores.shape[-1]-1,1)
 raise ValueError(kind)
def oracle_scores(x,ffn):return importance_scores(gated_activations(x,ffn),ffn.down_proj.weight,"weighted_activation")
def build_targets(ffn,activation_dir:Path,output_dir:Path,layer:int,retentions=(.3,.4,.5,.6,.75),transform="distribution",max_samples=None):
 output_dir.mkdir(parents=True,exist_ok=True);xs=[];scores=[];gated=[];meta=[]
 for path in sorted(activation_dir.glob("shard_*.pt")):
  p=torch.load(path,map_location="cpu",weights_only=True);remaining=None if max_samples is None else max_samples-len(meta)
  if remaining is not None and remaining<=0:break
  take=len(p["inputs"]) if remaining is None else min(len(p["inputs"]),remaining);x=p["inputs"][:take];device=ffn.gate_proj.weight.device
  with torch.inference_mode():aa=gated_activations(x.to(device=device,dtype=ffn.gate_proj.weight.dtype),ffn);s=importance_scores(aa,ffn.down_proj.weight,"weighted_activation").cpu()
  xs.append(x.cpu());scores.append(s.to(torch.float16));gated.append(aa.cpu().half())
  for i in range(take):meta.append({k:int(p[k][i]) for k in ("sample_ids","prompt_ids","token_positions","token_ids") if k in p})
 x=torch.cat(xs);raw=torch.cat(scores).float();torch.save({"inputs":x,"scores":transform_scores(raw,transform).half(),"raw_scores":raw.half(),"gated_activations":torch.cat(gated)},output_dir/"targets.pt");torch.save({"weight":ffn.down_proj.weight.detach().cpu().half(),"bias":None if ffn.down_proj.bias is None else ffn.down_proj.bias.detach().cpu().half()},output_dir/"down_projection.pt");torch.save(compute_down_column_norms_chunked(ffn.down_proj.weight),output_dir/"down_column_norms.pt")
 top=output_dir/"topk";top.mkdir(exist_ok=True)
 for r in retentions:torch.save(torch.topk(raw,math.ceil(r*raw.shape[1]),-1).indices.to(torch.int32),top/f"retention_{int(r*100):03d}.pt")
 (output_dir/"sample_metadata.jsonl").write_text("".join(json.dumps(row)+"\n" for row in meta));info={"layer":layer,"samples":len(x),"hidden_size":x.shape[1],"intermediate_size":raw.shape[1],"transform":transform,"retentions":list(retentions),"importance":"weighted_activation"};(output_dir/"metadata.json").write_text(json.dumps(info,indent=2)+"\n");return info
