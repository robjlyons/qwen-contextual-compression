import inspect
import json

import pytest
import torch

from predictor.accounting import predictor_accounting, predictor_macs
from predictor.models import FactorizedPredictor, ResidualFactorizedPredictor, architecture_metadata, create_predictor
from predictor.output_aware import hard_topk_mask
from predictor.training import load_initial_checkpoint, train


@pytest.mark.parametrize("latent", [32, 64, 96])
def test_residual_factorized_shape_and_x_only_api(latent):
    model = ResidualFactorizedPredictor(17, 29, latent)
    assert model(torch.randn(3, 17)).shape == (3, 29)
    assert list(inspect.signature(model.forward).parameters) == ["x"]


def test_residual_path_changes_output():
    model = ResidualFactorizedPredictor(5, 7, 4, dropout=0).eval()
    x = torch.randn(2, 5)
    with torch.no_grad():
        model.residual_fc1.weight.zero_(); model.residual_fc1.bias.zero_()
        model.residual_fc2.weight.zero_(); model.residual_fc2.bias.zero_()
        without_update = model(x)
        model.residual_fc2.bias.copy_(torch.arange(4))
        with_update = model(x)
    assert not torch.equal(without_update, with_update)


def test_d64_accounting_is_exact_and_below_compute_ceiling():
    model = ResidualFactorizedPredictor(5120, 17408, 64)
    expected_macs = 5120 * 64 + 64 * 64 + 64 * 64 + 64 * 17408
    accounting = predictor_accounting(model, 5120, 17408, .5)
    assert predictor_macs(model, 5120, 17408) == expected_macs == 1_449_984
    assert accounting["parameters"] == sum(parameter.numel() for parameter in model.parameters()) == 1_467_648
    assert accounting["dense_ffn_macs"] == 267_386_880
    assert accounting["predictor_mac_fraction"] == expected_macs / 267_386_880
    assert accounting["predictor_mac_fraction"] < .01 < .02


def test_metadata_and_checkpoint_architecture_are_strict(tmp_path):
    factorized = FactorizedPredictor(4, 6, 3)
    path = tmp_path / "factorized.pt"
    torch.save({"model": factorized.state_dict(), "mean": torch.zeros(4), "std": torch.ones(4), "config": architecture_metadata(factorized, "factorized", 4, 6, 3)}, path)
    residual = ResidualFactorizedPredictor(4, 6, 3)
    expected = architecture_metadata(residual, "residual_factorized", 4, 6, 3)
    assert expected["activation"] == "silu" and expected["residual_depth"] == 2 and expected["layer_norm"] is True
    with pytest.raises(ValueError, match="architecture mismatch"):
        load_initial_checkpoint(residual, path, expected)


def _tiny_targets(tmp_path):
    target = tmp_path / "layer_000" / "targets"; target.mkdir(parents=True)
    generator = torch.Generator().manual_seed(9); inputs = torch.randn(12, 4, generator=generator); raw = torch.rand(12, 6, generator=generator); scores = raw / raw.sum(-1, keepdim=True); gated = torch.randn(12, 6, generator=generator); weight = torch.randn(3, 6, generator=generator)
    torch.save({"inputs":inputs,"scores":scores,"raw_scores":raw,"gated_activations":gated},target/"targets.pt");torch.save({"weight":weight,"bias":None},target/"down_projection.pt");(target/"sample_metadata.jsonl").write_text("".join(json.dumps({"prompt_ids":i})+"\n" for i in range(12)));(target.parent/"splits.json").write_text(json.dumps({"train":list(range(8)),"validation":[8,9],"test":[10,11]}));return target


def test_residual_ce_and_warm_started_output_training_on_cpu(tmp_path):
    target = _tiny_targets(tmp_path); ce = target.parent / "residual_ce"
    train(target, ce, kind="residual_factorized", latent_dim=3, loss="distribution_ce", device="cpu", epochs=1, batch_size=4, early_stopping=False)
    checkpoint = torch.load(ce/"best.pt",map_location="cpu",weights_only=False);assert checkpoint["config"]["kind"] == "residual_factorized"
    fine = target.parent / "residual_fine"
    train(target, fine, kind="residual_factorized", latent_dim=3, loss="output_hybrid_rank", device="cpu", epochs=1, batch_size=4, init_checkpoint=ce/"best.pt", ranking_weight=.1)
    loaded=torch.load(fine/"best.pt",map_location="cpu",weights_only=False);assert loaded["config"]["init_checkpoint"] == str(ce/"best.pt")
    scores=create_predictor("residual_factorized",4,6,3)(torch.randn(2,4));assert bool((hard_topk_mask(scores,.5).sum(-1)==3).all())
