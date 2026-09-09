import torch
def threshold_margin(scores,k):v=torch.topk(scores,k+1,-1).values;return v[...,-2]-v[...,-1]
def fallback_retention(confidence,base,extra,threshold):return torch.where(confidence<threshold,torch.as_tensor(min(1.,base+extra),device=confidence.device),torch.as_tensor(base,device=confidence.device))
