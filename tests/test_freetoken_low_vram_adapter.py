from types import SimpleNamespace

import pytest

from integration.freetoken_qcc.low_vram.adapter import (
    FreeTokenLowVRAMAdapter,
    LowVRAMCompatibilityError,
)
from integration.freetoken_qcc.low_vram.manifest import REPRESENTATION


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


def _install_adapter_with_finalize(monkeypatch, representation=REPRESENTATION):
    import integration.freetoken_qcc.low_vram.adapter as adapter_module

    calls = []
    module = SimpleNamespace()

    def original_finalize_quant(root, marker=None):
        calls.append((root, marker))
        return 37

    class Engine:
        def __init__(self):
            # Construction happens only after install() has replaced the symbol.
            assert module.finalize_quant is not original_finalize_quant

        def _load_weight_state_dict(self):
            return {"cuda": "forbidden"}

    class Model:
        def load_state_dict(self, state):
            return None

    module.Engine = Engine
    module.finalize_quant = original_finalize_quant
    modules = {
        "freetoken.engine.engine": module,
        "freetoken.models.qwen3_5_moe.model": SimpleNamespace(Qwen3_5MoEForCausalLM=Model),
    }
    monkeypatch.setattr(adapter_module.importlib, "import_module", lambda name: modules[name])
    adapter = object.__new__(FreeTokenLowVRAMAdapter)
    adapter.config = SimpleNamespace()
    adapter.store = SimpleNamespace(manifest={"representation": representation})
    adapter.contract = {}
    adapter.fingerprint = {}
    adapter.bridge = None
    adapter.original_engine_loader = None
    adapter.original_model_loader = None
    adapter.original_finalize_quant = None
    adapter.engine_module = None
    adapter.model_class = None
    adapter.install()
    return adapter, module, Engine, original_finalize_quant, calls


def test_finalize_quant_guard_delegates_skips_and_restores(monkeypatch, capsys):
    adapter, engine_module, Engine, original, calls = _install_adapter_with_finalize(monkeypatch)
    guarded = engine_module.finalize_quant
    assert guarded is not original
    Engine()  # proves the module symbol was patched before Engine construction

    normal_root = SimpleNamespace()
    assert guarded(normal_root, marker="normal") == 37
    assert calls == [(normal_root, "normal")]

    low_vram_root = SimpleNamespace(_qcc_low_vram_bridge=object())
    assert guarded(low_vram_root) == 0
    assert calls == [(normal_root, "normal")]
    output = capsys.readouterr().out
    assert "QCC LOW-VRAM FINALIZE_QUANT GUARD INSTALLED" in output
    assert "QCC LOW-VRAM FINALIZE_QUANT SKIPPED" in output

    adapter.close()
    assert engine_module.finalize_quant is original


def test_finalize_quant_guard_rejects_non_finalized_cache(monkeypatch):
    adapter, engine_module, _, original, calls = _install_adapter_with_finalize(
        monkeypatch, representation="checkpoint-layout"
    )
    root = SimpleNamespace(_qcc_low_vram_bridge=object())
    with pytest.raises(LowVRAMCompatibilityError, match="non-finalized cache"):
        engine_module.finalize_quant(root)
    assert calls == []
    adapter.close()
    assert engine_module.finalize_quant is original
