import importlib.util
from types import SimpleNamespace

import pytest
import torch

from integration.freetoken_qcc.bridge import FreeTokenQCCBridge, FreeTokenQCCConfig, install_patch, parse_dense_mlp_layer
from integration.freetoken_qcc.metrics import ShadowMetrics


class FakeMLP:
    def __init__(self, layer=0, expert=False):
        marker = ".experts.0" if expert else ""
        self.gate_up_proj = SimpleNamespace(prefix=f"model.layers.{layer}.mlp{marker}.gate_up_proj")
        self.original_calls = 0

    def original(self, x):
        self.original_calls += 1
        return x + 1


class FakeRuntime:
    constructions = 0

    def __init__(self, config):
        type(self).constructions += 1
        self.memory = {"qcc_added_allocated_mib": 1.0}

    def run(self, x):
        return (x.float() * 2).to(x.dtype), torch.arange(8704, device=x.device)


def make_config(tmp_path, mode="replace", layer=0):
    checkpoint = tmp_path / "best.pt"
    weights = tmp_path / "layer.safetensors"
    checkpoint.touch()
    weights.touch()
    return FreeTokenQCCConfig(mode, layer, checkpoint, weights)


def dispatch(bridge, mlp, x):
    return bridge.forward(lambda instance, value: instance.original(value), mlp, x)


def test_prefix_parser_rejects_experts():
    assert parse_dense_mlp_layer(FakeMLP(12)) == 12
    assert parse_dense_mlp_layer(FakeMLP(0, expert=True)) is None
    assert parse_dense_mlp_layer(SimpleNamespace(prefix="unrelated")) is None


def test_configuration_defaults_off_and_validates_sparse_artifacts(tmp_path):
    assert FreeTokenQCCConfig.from_env({}).mode == "off"
    with pytest.raises(ValueError, match="required"):
        FreeTokenQCCConfig.from_env({"QCC_FT_MODE": "replace"})
    with pytest.raises(ValueError, match="RETENTION"):
        FreeTokenQCCConfig("off", retention=0.49).validate()
    make_config(tmp_path, "shadow").validate()


def test_off_untargeted_and_prefill_never_initialize(tmp_path):
    FakeRuntime.constructions = 0
    off = FreeTokenQCCBridge(FreeTokenQCCConfig(), FakeRuntime)
    mlp = FakeMLP()
    x = torch.zeros(1, 5120)
    assert torch.equal(dispatch(off, mlp, x), x + 1)
    bridge = FreeTokenQCCBridge(make_config(tmp_path), FakeRuntime)
    assert torch.equal(dispatch(bridge, FakeMLP(1), x), x + 1)
    prefill = torch.zeros(3, 5120)
    assert torch.equal(dispatch(bridge, mlp, prefill), prefill + 1)
    assert bridge.counters["untargeted_layer_fallbacks"] == 1
    assert bridge.counters["prefill_fallbacks"] == 1
    assert FakeRuntime.constructions == 0


def test_replace_restores_dtype_and_initializes_once(tmp_path):
    FakeRuntime.constructions = 0
    bridge = FreeTokenQCCBridge(make_config(tmp_path), FakeRuntime)
    mlp = FakeMLP()
    x = torch.ones(1, 5120, dtype=torch.bfloat16)
    for _ in range(2):
        output = dispatch(bridge, mlp, x)
        assert output.dtype == x.dtype
        assert torch.equal(output, x * 2)
    assert mlp.original_calls == 0
    assert bridge.counters["sparse_decode_calls"] == 2
    assert FakeRuntime.constructions == 1


def test_shadow_returns_stock_and_aggregates_metrics(tmp_path):
    bridge = FreeTokenQCCBridge(make_config(tmp_path, "shadow"), FakeRuntime)
    mlp = FakeMLP()
    x = torch.ones(1, 5120)
    output = dispatch(bridge, mlp, x)
    assert torch.equal(output, x + 1)
    summary = bridge.metrics.summary()
    assert summary["count"] == 1
    assert summary["all_finite"] is True
    assert bridge.counters["shadow_decode_calls"] == 1


def test_wrong_shape_falls_back(tmp_path):
    bridge = FreeTokenQCCBridge(make_config(tmp_path), FakeRuntime)
    x = torch.zeros(1, 16)
    assert torch.equal(dispatch(bridge, FakeMLP(), x), x + 1)
    assert bridge.counters["wrong_shape_fallbacks"] == 1


def test_metrics_jsonl_and_percentiles(tmp_path):
    path = tmp_path / "metrics" / "shadow.jsonl"
    metrics = ShadowMetrics(path)
    metrics.record(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), call_index=1)
    metrics.record(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]), call_index=2)
    summary = metrics.summary()
    assert summary["count"] == 2
    assert summary["cosine_mean"] == pytest.approx(0.5)
    assert len(path.read_text().splitlines()) == 2


def test_monkeypatch_changes_only_target_layer(tmp_path):
    class Dense:
        def __init__(self, layer):
            self.gate_up_proj = SimpleNamespace(prefix=f"model.layers.{layer}.mlp.gate_up_proj")

        def forward(self, x):
            return x + 1

    bridge = install_patch(make_config(tmp_path), Dense, FakeRuntime)
    x = torch.ones(1, 5120)
    assert torch.equal(Dense(0).forward(x), x * 2)
    assert torch.equal(Dense(3).forward(x), x + 1)
    assert bridge.counters["sparse_decode_calls"] == 1
    assert bridge.counters["untargeted_layer_fallbacks"] == 1


@pytest.mark.skipif(importlib.util.find_spec("freetoken") is None, reason="FreeToken is not installed")
def test_optional_freetoken_dense_architecture_import():
    from freetoken.models.qwen3_5_moe.moe import Qwen3_5DenseMLP

    assert callable(Qwen3_5DenseMLP.forward)
