"""Training and warm-start fine-tuning for pre-FFN predictors."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from predictor.datasets import prompt_split
from predictor.losses import distribution_ce, score_loss
from predictor.metrics import selection_metrics
from predictor.models import FactorizedPredictor, LowRankMLP
from predictor.output_aware import (
    ensure_dense_output_cache,
    hard_topk_mask,
    output_loss,
    reconstruct,
    ste_topk_mask,
)

OUTPUT_LOSSES = {"output_cosine", "output_relative", "output_hybrid", "output_hybrid_rank"}


def build_model(kind, input_dim, output_dim, latent_dim, dropout=0.0):
    cls = FactorizedPredictor if kind == "factorized" else LowRankMLP
    return cls(input_dim, output_dim, latent_dim, dropout)


def load_initial_checkpoint(model, path, expected, device="cpu"):
    """Load model/statistics only; optimizer and training progress are deliberately ignored."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing initialization checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    output_weights = [value for name, value in checkpoint["model"].items() if name.endswith("weight") and value.ndim == 2]
    actual = {
        "kind": config.get("kind"),
        "latent_dim": config.get("latent_dim"),
        "input_dim": config.get("input_dim", checkpoint["mean"].numel()),
        "output_dim": config.get("output_dim", output_weights[-1].shape[0]),
    }
    mismatches = {key: (actual[key], value) for key, value in expected.items() if actual[key] != value}
    if mismatches:
        raise ValueError(f"Initialization checkpoint architecture mismatch: {mismatches}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint["mean"].float(), checkpoint["std"].float(), checkpoint


def set_encoder_frozen(model, frozen):
    if not hasattr(model, "encoder"):
        if frozen:
            raise ValueError("Encoder freezing is only supported for models with an encoder")
        return
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not frozen)


def validation_checkpoint_score(metrics, target_p01=0.86):
    return metrics["mean_cosine"] - 0.10 * metrics["relative_l2"] - 0.05 * max(0.0, target_p01 - metrics["p01_cosine"])


def checkpoint_is_better(candidate, incumbent, min_p01=None):
    """Prefer eligible tail-safe checkpoints; fall back to score if none is eligible."""
    if incumbent is None:
        return True
    candidate_ok = min_p01 is None or candidate["p01_cosine"] >= min_p01
    incumbent_ok = min_p01 is None or incumbent["p01_cosine"] >= min_p01
    if candidate_ok != incumbent_ok:
        return candidate_ok
    return candidate["score"] > incumbent["score"]


def _validation(model, x, gated, dense, weight, bias, ids, mean, std, retention, device, batch_size):
    cosines, relatives = [], []
    with torch.inference_mode():
        for batch in ids.split(batch_size):
            scores = model(((x[batch] - mean) / std).to(device))
            mask = hard_topk_mask(scores, retention)  # evaluation is never soft/STE
            pred = reconstruct(gated[batch].to(device), mask, weight, bias)
            truth = dense[batch].to(device).float()
            cosines.append(F.cosine_similarity(pred, truth, -1).cpu())
            relatives.append((torch.linalg.vector_norm(pred - truth, dim=-1) / torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-12)).cpu())
    cosine, relative = torch.cat(cosines), torch.cat(relatives)
    metrics = {
        "mean_cosine": float(cosine.mean()),
        "p05_cosine": float(cosine.quantile(0.05)),
        "p01_cosine": float(cosine.quantile(0.01)),
        "relative_l2": float(relative.mean()),
        "p95_relative_l2": float(relative.quantile(0.95)),
        "p99_relative_l2": float(relative.quantile(0.99)),
    }
    metrics["score"] = validation_checkpoint_score(metrics)
    return metrics


def train(target_dir: Path, run_dir: Path, kind="factorized", latent_dim=128, loss="distribution_ce", device="cpu", epochs=100, batch_size=16, lr=None, weight_decay=1e-4, dropout=.1, patience=10, seed=42, train_retention=.5, temperature_start=1., temperature_end=.1, ste_normalize=True, cosine_weight=1., relative_weight=.25, ranking_weight=.05, amp=True, init_checkpoint=None, freeze_encoder_epochs=0, min_validation_p01=None, overwrite=False):
    started = time.perf_counter()
    torch.manual_seed(seed)
    target_dir, run_dir = Path(target_dir), Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not (run_dir / "latest.pt").exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing predictor run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    data = torch.load(target_dir / "targets.pt", map_location="cpu", weights_only=True)
    meta = [json.loads(line) for line in (target_dir / "sample_metadata.jsonl").read_text().splitlines()]
    split_path = run_dir.parent / "splits.json"
    if split_path.exists():
        splits = json.loads(split_path.read_text())
    else:
        splits = prompt_split([m.get("prompt_ids", m.get("sample_ids", i)) for i, m in enumerate(meta)], seed)
        split_path.write_text(json.dumps(splits, indent=2) + "\n")
    x, y, gated = data["inputs"].float(), data["scores"].float(), data["gated_activations"]
    tr, vi = torch.tensor(splits["train"]), torch.tensor(splits["validation"])
    model = build_model(kind, x.shape[1], y.shape[1], latent_dim, dropout)
    expected = {"kind": kind, "latent_dim": latent_dim, "input_dim": x.shape[1], "output_dim": y.shape[1]}
    normalization_source = "new_training_run"
    if init_checkpoint:
        mean, std, _ = load_initial_checkpoint(model, init_checkpoint, expected)
        normalization_source = "inherited_checkpoint"
    else:
        mean, std = x[tr].mean(0), x[tr].std(0)
    std = std.clamp_min(1e-6)
    model = model.to(device)
    effective_lr = lr if lr is not None else (5e-5 if init_checkpoint else 1e-3)
    # All parameters enter a fresh optimizer; requires_grad controls frozen epochs.
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=weight_decay)
    latest = run_dir / "latest.pt"
    start, best_metrics, wait = 0, None, 0
    output_aware = loss in OUTPUT_LOSSES
    weight = bias = dense = None
    if output_aware:
        cache = ensure_dense_output_cache(target_dir, device, batch_size)
        dense = torch.load(cache, map_location="cpu", weights_only=True)
        down = torch.load(target_dir / "down_projection.pt", map_location="cpu", weights_only=True)
        weight = down["weight"].float().to(device)
        bias = None if down["bias"] is None else down["bias"].float().to(device)
    if latest.exists() and not overwrite:
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best_metrics = state["epoch"] + 1, state.get("best_metrics")
    initial_validation = _validation(model, x, gated, dense, weight, bias, vi, mean, std, train_retention, device, batch_size) if output_aware else None
    history = [{"epoch": -1, "phase": "initial", **initial_validation}] if initial_validation else []
    for epoch in range(start, epochs):
        set_encoder_frozen(model, epoch < freeze_encoder_epochs)
        model.train()
        permutation, total = tr[torch.randperm(len(tr))], 0.0
        temperature = temperature_start + (temperature_end - temperature_start) * (epoch / max(epochs - 1, 1))
        for ids in permutation.split(batch_size):
            inputs, target = ((x[ids] - mean) / std).to(device), y[ids].to(device)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp and torch.device(device).type == "cuda"):
                pred = model(inputs)
            if output_aware:
                mask = ste_topk_mask(pred, train_retention, temperature, ste_normalize)
                sparse = reconstruct(gated[ids].to(device), mask, weight, bias)
                objective = output_loss(sparse, dense[ids].to(device), loss, cosine_weight, relative_weight)
                if loss == "output_hybrid_rank":
                    objective = objective + ranking_weight * distribution_ce(pred, target)
            else:
                objective = distribution_ce(pred, target) if loss == "distribution_ce" else score_loss(pred, target, loss)
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            total += float(objective) * len(ids)
        model.eval()
        if output_aware:
            validation = _validation(model, x, gated, dense, weight, bias, vi, mean, std, train_retention, device, batch_size)
        else:
            with torch.inference_mode():
                mass = selection_metrics(model(((x[vi] - mean) / std).to(device)), y[vi].to(device), train_retention)["captured_mass"].mean()
            validation = {"captured_mass": float(mass), "score": float(mass)}
        history.append({"epoch": epoch, "temperature": temperature, "train_loss": total / max(len(tr), 1), **validation})
        config = {**expected, "loss": loss, "seed": seed, "train_retention": train_retention, "ste_normalize": ste_normalize, "temperature_start": temperature_start, "temperature_end": temperature_end, "cosine_weight": cosine_weight, "relative_weight": relative_weight, "ranking_weight": ranking_weight, "amp": amp, "lr": effective_lr, "freeze_encoder_epochs": freeze_encoder_epochs, "normalization_source": normalization_source, "init_checkpoint": str(init_checkpoint) if init_checkpoint else None}
        candidate = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_metrics": validation, "mean": mean, "std": std, "config": config, "initial_validation": initial_validation}
        torch.save(candidate, latest)
        if checkpoint_is_better(validation, best_metrics, min_validation_p01):
            best_metrics, wait = validation, 0
            torch.save(candidate, run_dir / "best.pt")
        else:
            wait += 1
        if wait >= patience:
            break
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {"best_validation": best_metrics, "initial_validation": initial_validation, "epochs": len([h for h in history if h["epoch"] >= 0]), "training_seconds": time.perf_counter() - started, "batch_size": batch_size, "normalization_source": normalization_source, "splits": {key: len(value) for key, value in splits.items()}}
