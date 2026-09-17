from types import SimpleNamespace

import torch
from torch import nn

import runtime.cuda_selector as cuda_selector


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 32)
        self.residual_fc1 = nn.Linear(32, 32)
        self.residual_fc2 = nn.Linear(32, 32)
        self.layer_norm = nn.LayerNorm(32)
        self.neurons = nn.Linear(32, 8)


def test_latent_cuda_source_preserves_fc1_buffer():
    assert "float r[32],u[32],v[32]" in cuda_selector.CUDA
    assert "v[q]=s+r[q]" in cuda_selector.CUDA
    assert "u[q]=s+r[q]" not in cuda_selector.CUDA


def test_cuda_selector_evaluation_batch_runs_batch1_kernel(monkeypatch):
    calls = []

    class Extension:
        @staticmethod
        def scores(x, *unused):
            calls.append(tuple(x.shape))
            return x.sum(-1, keepdim=True).repeat(1, 8)

    monkeypatch.setattr(cuda_selector, "load_cuda_selector", lambda: (Extension(), {"available": True}))
    source = SimpleNamespace(predictor=Predictor().half())
    selector = cuda_selector.CudaSelector(source)
    result = selector.scores(torch.arange(12, dtype=torch.float16).reshape(3, 4))
    assert result.shape == (3, 8)
    assert calls == [(1, 4), (1, 4), (1, 4)]
