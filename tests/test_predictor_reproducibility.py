import json
from pathlib import Path

import torch

from predictor.losses import distribution_ce
from predictor.models import FactorizedPredictor
from predictor.reproducibility import (
    PHASE4_D64_CE_BASELINE,
    epoch_permutation,
    load_preset,
    set_global_seed,
    split_metadata,
)
from predictor.training import train


def _targets(root: Path):
    target = root / "layer_000" / "targets"
    target.mkdir(parents=True)
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(12, 4, generator=generator)
    raw = torch.rand(12, 6, generator=generator)
    scores = raw / raw.sum(-1, keepdim=True)
    torch.save({"inputs": inputs, "scores": scores, "raw_scores": raw, "gated_activations": torch.randn(12, 6, generator=generator)}, target / "targets.pt")
    (target / "sample_metadata.jsonl").write_text("".join(json.dumps({"prompt_ids": i}) + "\n" for i in range(12)))
    (target.parent / "splits.json").write_text(json.dumps({"train": list(range(8)), "validation": [8, 9], "test": [10, 11]}))
    return target


def test_baseline_preset_is_fixed_and_matches_documented_json():
    configured = json.loads(Path("configs/phase4_d64_ce_baseline.json").read_text())
    assert load_preset("phase4_d64_ce_baseline") == PHASE4_D64_CE_BASELINE == configured
    assert configured["learning_rate"] == 1e-3 and configured["checkpoint_objective"] == "validation_captured_mass"


def test_split_hash_is_order_independent_and_partition_specific():
    first = split_metadata({"train": [2, 1], "validation": [3], "test": [4]})
    second = split_metadata({"train": [1, 2], "validation": [3], "test": [4]})
    assert first == second
    assert len({first[f"{name}_split_sha256"] for name in ("train", "validation", "test")}) == 3


def test_fixed_seed_controls_initialization_and_epoch_shuffle():
    set_global_seed(42)
    first = FactorizedPredictor(4, 6, 3).state_dict()
    set_global_seed(42)
    second = FactorizedPredictor(4, 6, 3).state_dict()
    assert all(torch.equal(first[key], second[key]) for key in first)
    ids = torch.arange(20)
    assert torch.equal(epoch_permutation(ids, 42, 3, "epoch_seeded_torch_randperm"), epoch_permutation(ids, 42, 3, "epoch_seeded_torch_randperm"))
    set_global_seed(42); first_shuffle = epoch_permutation(ids, 42, 0)
    set_global_seed(42); second_shuffle = epoch_permutation(ids, 42, 0)
    assert torch.equal(first_shuffle, second_shuffle)


def test_distribution_ce_numerical_definition_and_intended_target():
    logits = torch.tensor([[1.0, 2.0, -1.0]])
    scores = torch.tensor([[2.0, 1.0, 1.0]])
    expected = -torch.sum(torch.tensor([[.5, .25, .25]]) * torch.log_softmax(logits, -1))
    torch.testing.assert_close(distribution_ce(logits, scores), expected)


def test_two_cpu_ce_runs_match_and_ste_options_do_not_affect_them(tmp_path):
    target = _targets(tmp_path)
    common = dict(kind="factorized", latent_dim=3, loss="distribution_ce", device="cpu", epochs=3, batch_size=4, seed=42, dropout=.1, early_stopping=False)
    train(target, target.parent / "repro_a", temperature_start=100, temperature_end=50, ste_normalize=False, **common)
    train(target, target.parent / "repro_b", temperature_start=.01, temperature_end=.001, ste_normalize=True, **common)
    a = torch.load(target.parent / "repro_a" / "best.pt", map_location="cpu", weights_only=False)
    b = torch.load(target.parent / "repro_b" / "best.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    torch.testing.assert_close(a["mean"], torch.load(target / "targets.pt", weights_only=True)["inputs"][:8].float().mean(0))
    assert a["config"]["lr"] == 1e-3
    config = json.loads((target.parent / "repro_a" / "config.json").read_text())
    assert config["train_size"] == 8 and config["test_size"] == 2
    assert config["target_tensor"] == "scores"
