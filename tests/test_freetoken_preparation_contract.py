from contextlib import contextmanager
from types import SimpleNamespace

import torch

import integration.freetoken_qcc.low_vram.preparation as preparation


def test_prepare_supplies_tp_dtype_uses_bf16_and_dispatches_loader_once(monkeypatch, tmp_path, capsys):
    calls = {"load_weight": 0, "iter_weights": 0}
    observed = {}

    class DistributedInfo:
        def __init__(self, rank, size):
            self.rank, self.size = rank, size

    class EngineConfig:
        def __init__(self, model_path: str, tp_info: DistributedInfo, dtype: torch.dtype):
            observed["config"] = (model_path, tp_info.rank, tp_info.size, dtype)
            self.model_config = SimpleNamespace(name="fake")
            self.dtype = dtype

    class FakeModel:
        def __init__(self):
            self.model = SimpleNamespace(layers=SimpleNamespace(op_list=[object()]))

    def set_tp_info(rank, size):
        observed["set_tp"] = (rank, size)

    def set_rope_device(device):
        observed["rope_device"] = device.type

    def create_model(model_config):
        assert observed.get("rope_device") == "cpu"
        observed["placeholder_dtype"] = torch.empty(1).dtype
        observed["placeholder_device"] = torch.empty(1).device.type
        return FakeModel()

    def load_weight(model_path, device, include_moe_experts=False, include_vision=False):
        calls["load_weight"] += 1
        observed["loader"] = (model_path, device.type, include_moe_experts, include_vision)
        return iter([("normalized.weight", torch.ones(1, dtype=torch.bfloat16))])

    def iter_weights(weights):
        calls["iter_weights"] += 1
        return weights

    @contextmanager
    def torch_dtype(dtype):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(previous)

    attributes = {
        "set_tp_info": set_tp_info,
        "set_rope_device": set_rope_device,
        "EngineConfig": EngineConfig,
        "create_model": create_model,
        "load_weight": load_weight,
    }
    monkeypatch.setattr(preparation, "_find_attribute", lambda candidates: attributes[candidates[0][1]])
    monkeypatch.setattr(preparation, "inspect_installed_contract", lambda: {"version": "0.1.2-test", "signatures": {"qwen_weight.iter_weights": str(__import__("inspect").signature(iter_weights))}})
    monkeypatch.setattr(preparation, "validate_fingerprint", lambda: {})
    modules = {
        "freetoken.distributed": SimpleNamespace(DistributedInfo=DistributedInfo),
        "freetoken.utils": SimpleNamespace(torch_dtype=torch_dtype),
        "freetoken.models.qwen3_5_moe.weight": SimpleNamespace(iter_weights=iter_weights),
    }
    monkeypatch.setattr(preparation.importlib, "import_module", lambda name: modules[name])

    def write_cache(model, loaded, output, *args, **kwargs):
        observed["loaded"] = next(loaded)[0]
        observed["loader_diagnostics"] = kwargs["loader_diagnostics"].to_dict()
        return {"ok": True}

    monkeypatch.setattr(preparation, "build_stream_cache_streaming", write_cache)
    previous_dtype = torch.get_default_dtype()
    result = preparation.prepare_stream_cache("fake/model", tmp_path / "cache", expected_layers=1)
    assert result == {"ok": True}
    assert observed["config"] == ("fake/model", 0, 1, torch.bfloat16)
    assert observed["set_tp"] == (0, 1)
    assert observed["rope_device"] == "cpu"
    assert observed["placeholder_dtype"] == torch.bfloat16
    assert observed["placeholder_device"] == "meta"
    assert torch.get_default_dtype() == previous_dtype
    assert observed["loader"] == ("fake/model", "cpu", False, False)
    assert observed["loaded"] == "normalized.weight"
    assert observed["loader_diagnostics"]["enabled"] is False
    assert calls == {"load_weight": 1, "iter_weights": 0}
    assert "QCC STREAM CACHE FREETOKEN CONTRACT" in capsys.readouterr().out


def test_default_dtype_fallback_restores_previous(monkeypatch):
    monkeypatch.setattr(preparation.importlib, "import_module", lambda name: SimpleNamespace())
    previous = torch.get_default_dtype()
    with preparation._torch_dtype_context(torch.bfloat16):
        assert torch.get_default_dtype() == torch.bfloat16
    assert torch.get_default_dtype() == previous
