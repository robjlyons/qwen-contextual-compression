"""Numerical and transfer diagnostics for low-VRAM layer validation."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def output_comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference, candidate = reference.float(), candidate.float()
    difference = candidate - reference
    return {
        "cosine": float(F.cosine_similarity(reference.reshape(1, -1), candidate.reshape(1, -1))),
        "relative_l2": float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)),
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "finite": bool(torch.isfinite(candidate).all()),
    }
