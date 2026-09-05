import torch
from oracle.importance import importance_scores,topk_indices,block_topk_indices,random_indices
def test_scores_and_topk():
 a=torch.tensor([[1.,-4.,2.]]); w=torch.tensor([[3.,2.,4.],[0.,0.,0.]])
 assert topk_indices(importance_scores(a,w,"activation"),2).tolist()==[[1,2]]
 assert torch.allclose(importance_scores(a,w,"weighted_activation"),importance_scores(a,w,"exact_output_norm"))
def test_random_reproducible(): assert torch.equal(random_indices(3,10,4,7),random_indices(3,10,4,7))
def test_blocks_complete():
 idx=block_topk_indices(torch.tensor([[1.,1.,9.,9.,2.,2.,0.,0.]]),3,2)
 assert set(idx[0].tolist()) in ({2,3,4,5},{0,1,2,3}) and len(idx[0])%2==0

