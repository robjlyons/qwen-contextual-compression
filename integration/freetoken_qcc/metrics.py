"""Shadow-mode metrics that deliberately compare stock NVFP4 and QCC FP16 outputs."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean

import torch
import torch.nn.functional as F


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


class ShadowMetrics:
    """Collect correctness metrics; tensor-to-host conversion makes shadow non-performance mode."""

    def __init__(self, output: Path | None = None):
        self.output = Path(output) if output else None
        self.records: list[dict] = []
        if self.output:
            self.output.parent.mkdir(parents=True, exist_ok=True)

    def record(self, stock: torch.Tensor, sparse: torch.Tensor, **metadata) -> dict:
        stock32, sparse32 = stock.detach().float(), sparse.detach().float()
        difference = sparse32 - stock32
        record = {
            **metadata,
            "stock_output_norm": float(torch.linalg.vector_norm(stock32)),
            "sparse_output_norm": float(torch.linalg.vector_norm(sparse32)),
            "cosine": float(F.cosine_similarity(stock32.reshape(1, -1), sparse32.reshape(1, -1))),
            "relative_l2": float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(stock32).clamp_min(1e-12)),
            "max_abs_difference": float(difference.abs().max()),
            "mean_abs_difference": float(difference.abs().mean()),
            "finite": bool(torch.isfinite(sparse32).all()),
        }
        self.records.append(record)
        if self.output:
            with self.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def summary(self) -> dict:
        cosine = [item["cosine"] for item in self.records]
        relative_l2 = [item["relative_l2"] for item in self.records]
        return {
            "count": len(self.records),
            "cosine_mean": fmean(cosine) if cosine else None,
            "cosine_p05": _percentile(cosine, 0.05),
            "cosine_min": min(cosine) if cosine else None,
            "relative_l2_mean": fmean(relative_l2) if relative_l2 else None,
            "relative_l2_p95": _percentile(relative_l2, 0.95),
            "relative_l2_max": max(relative_l2) if relative_l2 else None,
            "all_finite": all(item["finite"] for item in self.records),
            "comparison_warning": "Difference combines 50% sparsity with FP16 research weights versus FreeToken NVFP4 weights.",
        }
