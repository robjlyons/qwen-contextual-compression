"""Reproducibility metadata for predictor experiments."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


PHASE4_D64_CE_BASELINE = {
    "model": "factorized",
    "latent_dim": 64,
    "loss": "distribution_ce",
    "seed": 42,
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "scheduler": "none",
    "patience": 10,
    "early_stopping": True,
    "checkpoint_objective": "validation_captured_mass",
    "amp": True,
    "dropout": 0.1,
    "normalization": "train_split_mean_std",
    "shuffle": "global_rng_torch_randperm",
    "target_tensor": "scores",
    "target_interpretation": "per_token_importance_distribution",
}

PRESETS = {"phase4_d64_ce_baseline": PHASE4_D64_CE_BASELINE}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_permutation(indices: torch.Tensor, seed: int, epoch: int, strategy="global_rng_torch_randperm") -> torch.Tensor:
    if strategy == "global_rng_torch_randperm":
        return indices[torch.randperm(len(indices))]
    if strategy == "epoch_seeded_torch_randperm":
        generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
        return indices[torch.randperm(len(indices), generator=generator)]
    raise ValueError(f"Unknown shuffle strategy: {strategy}")


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = f"{value.dtype}:{tuple(value.shape)}:".encode()
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def split_metadata(splits: dict) -> dict:
    result = {}
    for name in ("train", "validation", "test"):
        values = sorted(map(int, splits[name]))
        result[f"{name}_size"] = len(values)
        result[f"{name}_split_sha256"] = canonical_hash(values)
    return result


def normalization_metadata(mean: torch.Tensor, std: torch.Tensor) -> dict:
    def describe(value):
        value = value.detach().cpu().float()
        return {"shape": list(value.shape), "dtype": str(value.dtype), "min": float(value.min()), "max": float(value.max()), "mean": float(value.mean()), "std": float(value.std(unbiased=False)), "sha256": tensor_hash(value)}
    return {"mean": describe(mean), "std": describe(std)}


def load_preset(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError(f"Unknown predictor preset: {name}")
    return dict(PRESETS[name])


def write_run_config(run_dir: Path, config: dict) -> None:
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
