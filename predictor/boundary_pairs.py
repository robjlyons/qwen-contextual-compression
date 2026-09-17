"""Training-only exact swap targets for the Phase 4.6 boundary scorer."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from predictor.boundary_reranker import stage1_boundary
from predictor.candidate_analysis import load_stage1


def add_gains(residual: torch.Tensor, contributions: torch.Tensor) -> torch.Tensor:
    """Squared-error reduction from adding each contribution."""
    return 2 * contributions @ residual - contributions.square().sum(-1)


def remove_gains(residual: torch.Tensor, contributions: torch.Tensor) -> torch.Tensor:
    """Squared-error reduction from removing each contribution."""
    return -2 * contributions @ residual - contributions.square().sum(-1)


def pair_gain_matrix(residual: torch.Tensor, selected: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    """Return exact remove-selected/add-rejected gains without a [I,J,H] tensor."""
    e_i = selected @ residual
    e_j = rejected @ residual
    i_norm = selected.square().sum(-1)
    j_norm = rejected.square().sum(-1)
    cross = selected @ rejected.T
    return 2 * (e_j[None, :] - e_i[:, None]) - (i_norm[:, None] + j_norm[None, :] - 2 * cross)


def shortlist_pairs(
    residual: torch.Tensor,
    selected_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    gated: torch.Tensor,
    down_weight: torch.Tensor,
    remove_shortlist: int = 32,
    add_shortlist: int = 64,
    positive_pairs: int = 64,
    negative_pairs: int = 64,
) -> dict[str, torch.Tensor]:
    """Build bounded exact swap pairs for one token."""
    if selected_ids.numel() == 0 or rejected_ids.numel() == 0:
        raise ValueError("selected and rejected boundary sets must be non-empty")
    selected_ids = selected_ids.to(down_weight.device)
    rejected_ids = rejected_ids.to(down_weight.device)
    selected_v = gated[selected_ids, None].to(down_weight) * down_weight[:, selected_ids].T
    rejected_v = gated[rejected_ids, None].to(down_weight) * down_weight[:, rejected_ids].T
    nr = min(remove_shortlist, selected_ids.numel())
    na = min(add_shortlist, rejected_ids.numel())
    selected_pos = torch.topk(remove_gains(residual, selected_v), nr).indices
    rejected_pos = torch.topk(add_gains(residual, rejected_v), na).indices
    selected_v, selected_ids = selected_v[selected_pos], selected_ids[selected_pos]
    rejected_v, rejected_ids = rejected_v[rejected_pos], rejected_ids[rejected_pos]
    gains = pair_gain_matrix(residual, selected_v, rejected_v)
    flat = gains.flatten()
    positive = torch.nonzero(flat > 0, as_tuple=False).flatten()
    if positive.numel():
        positive = positive[torch.topk(flat[positive], min(positive_pairs, positive.numel())).indices]
    negative = torch.nonzero(flat <= 0, as_tuple=False).flatten()
    if negative.numel():
        # Closest to zero are the confusing hard negatives.
        negative = negative[torch.topk(flat[negative], min(negative_pairs, negative.numel())).indices]
    chosen = torch.cat((positive, negative))
    labels = torch.cat((torch.ones_like(positive, dtype=torch.bool), torch.zeros_like(negative, dtype=torch.bool)))
    chosen_gain = flat[chosen]
    weights = torch.ones_like(chosen_gain)
    if positive.numel():
        weights[: positive.numel()] = (chosen_gain[: positive.numel()] / chosen_gain[: positive.numel()].mean().clamp_min(1e-12)).clamp(.25, 4)
    return {
        "selected_ids": selected_ids[chosen // na],
        "rejected_ids": rejected_ids[chosen % na],
        "gain": chosen_gain,
        "label": labels,
        "weight": weights,
        "positive_count": torch.tensor(positive.numel()),
        "negative_count": torch.tensor(negative.numel()),
        "best_single_gain": flat.max(),
    }


def pairwise_swap_loss(selected_scores, rejected_scores, labels, weights, temperature=1., negative_weight=.5):
    if temperature <= 0:
        raise ValueError("pair temperature must be positive")
    margin = (rejected_scores - selected_scores) / temperature
    positive = labels.bool()
    terms = []
    if positive.any():
        terms.append((weights[positive] * F.softplus(-margin[positive])).mean())
    if (~positive).any():
        terms.append(negative_weight * F.softplus(margin[~positive]).mean())
    if not terms:
        raise ValueError("pair batch contains no valid pairs")
    return sum(terms)


def pair_statistics(records: list[dict], sample_ids: list[int]) -> dict:
    rows = [records[i] for i in sample_ids]
    positives = torch.tensor([int(row["positive_count"]) for row in rows], dtype=torch.float32)
    negatives = torch.tensor([int(row["negative_count"]) for row in rows], dtype=torch.float32)
    gains = torch.cat([row["gain"].float() for row in rows]) if rows else torch.empty(0)
    positive_gains = gains[gains > 0]
    result = {"samples": len(rows), "mean_positive_pairs": float(positives.mean()), "median_positive_pairs": float(positives.median()), "mean_negative_pairs": float(negatives.mean()), "fraction_zero_positive": float((positives == 0).float().mean()), "fraction_1plus_positive": float((positives >= 1).float().mean()), "fraction_10plus_positive": float((positives >= 10).float().mean()), "fraction_64plus_positive": float((positives >= 64).float().mean())}
    if positive_gains.numel():
        result.update(gain_mean=float(positive_gains.mean()), gain_median=float(positive_gains.median()), gain_p95=float(positive_gains.quantile(.95)), gain_p99=float(positive_gains.quantile(.99)), gain_max=float(positive_gains.max()))
    return result


def build_boundary_pair_targets(results_dir: Path, layer: int, stage1_run: str, remove_shortlist=32, add_shortlist=64, positive_pairs=64, negative_pairs=64, lock_retention=.45, candidate_retention=.65, final_retention=.5, device="cpu"):
    """Build a compact cache from existing targets; never captures model activations."""
    if (lock_retention, final_retention, candidate_retention) != (.45, .5, .65):
        raise ValueError("Phase 4.6 boundary configuration is fixed at 0.45/0.50/0.65")
    started = time.perf_counter()
    layer_dir = Path(results_dir) / f"layer_{layer:03d}"
    target_dir = layer_dir / "targets"
    model, checkpoint, data, stage1_hash = load_stage1(target_dir, layer_dir / stage1_run, device)
    splits = json.loads((layer_dir / "splits.json").read_text())
    mean, std = checkpoint["mean"].float(), checkpoint["std"].float().clamp_min(1e-6)
    x = ((data["inputs"].float() - mean) / std).to(device)
    gated = data["gated_activations"].to(device)
    down = torch.load(target_dir / "down_projection.pt", map_location="cpu", weights_only=True)
    weight = down["weight"].to(device)
    bias = None if down["bias"] is None else down["bias"].to(device=device, dtype=weight.dtype)
    records = []
    with torch.inference_mode():
        scores = model(x)
        locked, boundary, _, needed = stage1_boundary(scores, lock_retention, final_retention, candidate_retention)
        for sample in range(len(x)):
            selected_ids, rejected_ids = boundary[sample, :needed], boundary[sample, needed:]
            final_ids = torch.cat((locked[sample], selected_ids))
            sparse = F.linear(torch.zeros_like(gated[sample]).scatter(0, final_ids, gated[sample, final_ids]).to(weight), weight, bias)
            dense = F.linear(gated[sample].to(weight), weight, bias)
            record = shortlist_pairs(dense - sparse, selected_ids, rejected_ids, gated[sample], weight, remove_shortlist, add_shortlist, positive_pairs, negative_pairs)
            record["sample_id"] = torch.tensor(sample)
            records.append({key: value.cpu() for key, value in record.items()})
    config = {"kind": "boundary_pair_targets", "stage1_run": stage1_run, "stage1_checkpoint_sha256": stage1_hash, "lock_retention": lock_retention, "final_retention": final_retention, "candidate_retention": candidate_retention, "remove_shortlist": remove_shortlist, "add_shortlist": add_shortlist, "positive_pairs": positive_pairs, "negative_pairs": negative_pairs, "split_unit": "sample"}
    output = layer_dir / "boundary_pair_targets"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite pair targets: {output}")
    output.mkdir(parents=True)
    cache_path = output / "pairs.pt"
    torch.save({"records": records, "config": config, "splits": splits}, cache_path)
    stats = {name: pair_statistics(records, ids) for name, ids in splits.items()}
    (output / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return {"output": str(output), "statistics": stats, "config": config, "build_seconds": time.perf_counter() - started, "storage_bytes": cache_path.stat().st_size}
