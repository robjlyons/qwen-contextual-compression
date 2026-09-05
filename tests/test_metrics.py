import torch
from evaluation.metrics import output_metrics
def test_identical_metrics():
 x=torch.randn(3,4); m=output_metrics(x,x)
 assert torch.allclose(m["cosine_similarity"],torch.ones(3),atol=1e-6)
 assert all(torch.count_nonzero(m[k])==0 for k in ("relative_l2","mse","max_absolute_error"))

