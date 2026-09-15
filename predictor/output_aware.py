"""Training-only straight-through selection and FFN reconstruction losses."""
import math,torch
import torch.nn.functional as F

class _StraightThroughMask(torch.autograd.Function):
 @staticmethod
 def forward(ctx,hard,soft):return hard.clone()
 @staticmethod
 def backward(ctx,gradient):return None,gradient

def hard_topk_mask(scores,retention):
 k=max(1,math.ceil(scores.shape[-1]*retention));indices=torch.topk(scores,k,-1,sorted=False).indices;return torch.zeros_like(scores).scatter_(-1,indices,1)
def ste_topk_mask(scores,retention,temperature=.5,normalize=True):
 if temperature<=0:raise ValueError("temperature must be positive")
 z=(scores-scores.mean(-1,keepdim=True))/scores.std(-1,keepdim=True,unbiased=False).clamp_min(1e-6) if normalize else scores
 hard=hard_topk_mask(scores,retention);k=max(1,math.ceil(scores.shape[-1]*retention));threshold=torch.topk(z,k,-1,sorted=True).values[...,-1:].detach();soft=torch.sigmoid((z-threshold)/temperature);return _StraightThroughMask.apply(hard,soft)
def reconstruct(gated,mask,weight,bias=None):return F.linear(gated.float()*mask.float(),weight,None if bias is None else bias)
def output_cosine_loss(sparse,dense):return (1-F.cosine_similarity(sparse.float(),dense.float(),dim=-1)).mean()
def output_relative_loss(sparse,dense,eps=1e-12):return (((sparse.float()-dense.float()).square().sum(-1))/(dense.float().square().sum(-1)+eps)).mean()
def output_loss(sparse,dense,kind="output_hybrid",cosine_weight=1.,relative_weight=.25):
 cosine=output_cosine_loss(sparse,dense);relative=output_relative_loss(sparse,dense)
 if kind=="output_cosine":return cosine
 if kind=="output_relative":return relative
 return cosine_weight*cosine+relative_weight*relative
def ensure_dense_output_cache(target_dir,device="cpu",batch_size=16):
 path=target_dir/"dense_ffn_outputs.pt"
 if path.exists():return path
 data=torch.load(target_dir/"targets.pt",map_location="cpu",weights_only=True);down=torch.load(target_dir/"down_projection.pt",map_location="cpu",weights_only=True);weight=down["weight"].to(device);bias=None if down["bias"] is None else down["bias"].to(device=device,dtype=weight.dtype);parts=[]
 with torch.inference_mode():
  for batch in data["gated_activations"].split(batch_size):parts.append(F.linear(batch.to(device=device,dtype=weight.dtype),weight,bias).cpu().half())
 torch.save(torch.cat(parts),path);return path
