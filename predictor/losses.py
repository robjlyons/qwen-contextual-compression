import torch
def distribution_ce(logits,scores):q=scores/scores.sum(-1,keepdim=True).clamp_min(1e-12);return -(q*torch.log_softmax(logits,-1)).sum(-1).mean()
def score_loss(pred,target,kind="huber"):return torch.nn.functional.huber_loss(pred,target) if kind=="huber" else torch.nn.functional.mse_loss(pred,target)
def weighted_topk_bce(logits,scores,k):
 idx=torch.topk(scores,k,-1).indices;target=torch.zeros_like(scores).scatter_(-1,idx,1);weight=1+target*scores/scores.mean(-1,keepdim=True).clamp_min(1e-12);return torch.nn.functional.binary_cross_entropy_with_logits(logits,target,weight=weight)
def sampled_pairwise(logits,scores,pairs=256):
 hi=torch.topk(scores,min(pairs,scores.shape[-1]),-1).indices;lo=torch.randint(scores.shape[-1],hi.shape,device=scores.device);return torch.nn.functional.softplus(-(logits.gather(-1,hi)-logits.gather(-1,lo))).mean()
