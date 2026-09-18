"""Host-backed token embedding gather."""
from __future__ import annotations

import torch


class HostBackedEmbedding:
    def __init__(self, weight: torch.Tensor, device):
        if weight.ndim != 2 or weight.device.type != "cpu":
            raise ValueError("embedding weight must be a two-dimensional CPU tensor")
        self.weight = weight
        self.device = torch.device(device)

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = token_ids.detach().to(device="cpu", dtype=torch.int64)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.weight.shape[0]):
            raise IndexError("token ID outside embedding vocabulary")
        gathered = self.weight.index_select(0, ids.reshape(-1)).reshape(*ids.shape, self.weight.shape[1])
        return gathered.to(self.device)
