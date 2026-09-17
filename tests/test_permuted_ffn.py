import numpy as np
import torch
from conftest import TinyFFN
from clustering.permutation import permuted_ffn_copy

def test_dense_permuted_ffn_is_exact_with_biases():
 torch.manual_seed(4); ffn=TinyFFN(h=7,n=13,bias=True); x=torch.randn(11,7)
 perm=np.random.default_rng(8).permutation(13); reordered=permuted_ffn_copy(ffn,perm)
 torch.testing.assert_close(ffn(x),reordered(x),rtol=1e-5,atol=1e-6)

def test_projection_axes_and_bias_are_permuted_consistently():
 ffn=TinyFFN(h=3,n=4,bias=True); original=ffn.gate_proj.weight.detach().clone(); bias=ffn.gate_proj.bias.detach().clone(); perm=np.array([3,1,0,2])
 reordered=permuted_ffn_copy(ffn,perm)
 assert torch.equal(reordered.gate_proj.weight,original[perm]); assert torch.equal(reordered.gate_proj.bias,bias[perm])

