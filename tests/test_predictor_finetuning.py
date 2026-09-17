import hashlib

import pytest
import torch

from predictor.losses import distribution_ce
from predictor.models import FactorizedPredictor
from predictor.training import (
    checkpoint_is_better,
    load_initial_checkpoint,
    set_encoder_frozen,
    validation_checkpoint_score,
)


def _checkpoint(path, input_dim=4, output_dim=6, latent_dim=3):
    model = FactorizedPredictor(input_dim, output_dim, latent_dim)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": {"old": "must not be loaded"},
        "mean": torch.arange(input_dim).float(),
        "std": torch.arange(input_dim).float() + 1,
        "config": {"kind": "factorized", "latent_dim": latent_dim, "input_dim": input_dim, "output_dim": output_dim},
    }
    torch.save(checkpoint, path)
    return model, checkpoint


def test_warm_start_loads_weights_and_inherits_statistics_without_mutating_source(tmp_path):
    path = tmp_path / "ce.pt"
    source, checkpoint = _checkpoint(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    warm = FactorizedPredictor(4, 6, 3)
    mean, std, loaded = load_initial_checkpoint(warm, path, {"kind": "factorized", "latent_dim": 3, "input_dim": 4, "output_dim": 6})
    x = torch.randn(5, 4)
    torch.testing.assert_close(warm(x), source(x))  # zero-step warm start
    torch.testing.assert_close(mean, checkpoint["mean"])
    torch.testing.assert_close(std, checkpoint["std"])
    assert "optimizer" in loaded and hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("key,value", [("kind", "mlp"), ("latent_dim", 7), ("input_dim", 5), ("output_dim", 8)])
def test_warm_start_architecture_mismatch_is_clear(tmp_path, key, value):
    path = tmp_path / "ce.pt"
    _checkpoint(path)
    expected = {"kind": "factorized", "latent_dim": 3, "input_dim": 4, "output_dim": 6}
    expected[key] = value
    with pytest.raises(ValueError, match="architecture mismatch"):
        load_initial_checkpoint(FactorizedPredictor(4, 6, 3), path, expected)


def test_encoder_freeze_and_unfreeze_preserve_head_training():
    model = FactorizedPredictor(4, 6, 3)
    set_encoder_frozen(model, True)
    distribution_ce(model(torch.randn(2, 4)), torch.rand(2, 6)).backward()
    assert model.encoder.weight.grad is None
    assert model.neurons.weight.grad is not None
    model.zero_grad(set_to_none=True)
    set_encoder_frozen(model, False)
    distribution_ce(model(torch.randn(2, 4)), torch.rand(2, 6)).backward()
    assert model.encoder.weight.grad is not None


def test_tail_aware_checkpoint_rule_and_score():
    weak_tail = {"mean_cosine": .99, "relative_l2": .1, "p01_cosine": .7}
    safe = {"mean_cosine": .98, "relative_l2": .11, "p01_cosine": .86}
    for metrics in (weak_tail, safe):
        metrics["score"] = validation_checkpoint_score(metrics)
    assert checkpoint_is_better(safe, weak_tail, min_p01=.85)
    assert not checkpoint_is_better(weak_tail, safe, min_p01=.85)


def test_ranking_weight_changes_hybrid_objective():
    predicted = torch.randn(3, 7)
    target = torch.rand(3, 7)
    output_objective = torch.tensor(.25)
    low = output_objective + .02 * distribution_ce(predicted, target)
    high = output_objective + .20 * distribution_ce(predicted, target)
    assert high > low
