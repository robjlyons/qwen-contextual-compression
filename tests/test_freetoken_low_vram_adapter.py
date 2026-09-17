from types import SimpleNamespace

from integration.freetoken_qcc.low_vram.adapter import FreeTokenLowVRAMAdapter


def test_engine_loader_is_suppressed_and_model_hook_installs_cache(monkeypatch):
    import integration.freetoken_qcc.low_vram.adapter as adapter_module

    calls = []

    class Engine:
        def _load_weight_state_dict(self):
            calls.append("original_loader")
            return {"cuda": "forbidden"}

    class Model:
        def load_state_dict(self, state):
            calls.append("original_model_loader")

    modules = {
        "freetoken.engine.engine": SimpleNamespace(Engine=Engine),
        "freetoken.models.qwen3_5_moe.model": SimpleNamespace(Qwen3_5MoEForCausalLM=Model),
    }
    monkeypatch.setattr(adapter_module.importlib, "import_module", lambda name: modules[name])
    adapter = object.__new__(FreeTokenLowVRAMAdapter)
    adapter.config = SimpleNamespace()
    adapter.store = None
    adapter.contract = {}
    adapter.fingerprint = {}
    adapter.bridge = None
    adapter.original_engine_loader = None
    adapter.original_model_loader = None
    monkeypatch.setattr(adapter, "install_model", lambda model: calls.append("install_cache"))
    adapter.install()
    assert Engine()._load_weight_state_dict() == {}
    Model().load_state_dict({})
    assert calls == ["install_cache"]
    assert adapter.original_engine_loader is not None


def test_normal_checkpoint_mapping_is_rejected_by_model_hook(monkeypatch):
    import integration.freetoken_qcc.low_vram.adapter as adapter_module

    class Engine:
        def _load_weight_state_dict(self): return {}

    class Model:
        def load_state_dict(self, state): return None

    modules = {
        "freetoken.engine.engine": SimpleNamespace(Engine=Engine),
        "freetoken.models.qwen3_5_moe.model": SimpleNamespace(Qwen3_5MoEForCausalLM=Model),
    }
    monkeypatch.setattr(adapter_module.importlib, "import_module", lambda name: modules[name])
    adapter = object.__new__(FreeTokenLowVRAMAdapter)
    adapter.config = adapter.store = None
    adapter.bridge = None
    adapter.install()
    try:
        Model().load_state_dict({"unexpected": 1})
    except RuntimeError as error:
        assert "not suppressed" in str(error)
    else:
        raise AssertionError("normal checkpoint materialization was accepted")
