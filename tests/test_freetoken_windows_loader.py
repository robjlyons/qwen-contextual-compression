from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import integration.freetoken_qcc.low_vram.windows_loader as windows_loader
from integration.freetoken_qcc.low_vram.windows_loader import (
    OwnedLoaderDiagnostics,
    OwnedShard,
    make_owned_safe_open,
    windows_safe_freetoken_loader,
)


def test_owned_shard_surface():
    shard = OwnedShard({"a": torch.tensor([1]), "b": torch.tensor([2])})
    assert list(shard.keys()) == ["a", "b"]
    assert shard.get_tensor("b").item() == 2


def test_pread_mode_owns_tensor_after_context_exit(tmp_path):
    calls = []

    def load_file(path, device="cpu", backend=None):
        calls.append((path, device, backend))
        return {"weight": torch.tensor([7])}

    def unused_safe_open(*args, **kwargs):
        raise AssertionError("pread mode must not call safe_open")

    diagnostics = OwnedLoaderDiagnostics(True, "win32", loader_mode="pread")
    owned_open = make_owned_safe_open(unused_safe_open, load_file, "pread", diagnostics)
    path = tmp_path / "shard.safetensors"
    path.write_bytes(b"source")
    with owned_open(path, framework="pt", device="cpu") as shard:
        retained = shard.get_tensor("weight")
    assert retained.item() == 7
    assert calls == [(path, "cpu", "pread")]
    assert diagnostics.owned_shards_opened == diagnostics.owned_shards_released == 1
    assert diagnostics.owned_source_bytes == len(b"source")


def test_multiple_sequential_owned_shards(tmp_path):
    diagnostics = OwnedLoaderDiagnostics(True, "win32", loader_mode="pread")
    load_file = lambda path, device="cpu", backend=None: {"value": torch.tensor([int(path.stem)])}
    owned_open = make_owned_safe_open(None, load_file, "pread", diagnostics)
    retained = []
    for index in (1, 2):
        path = tmp_path / f"{index}.safetensors"
        path.write_bytes(b"x" * index)
        with owned_open(path) as shard:
            retained.append(shard.get_tensor("value"))
    assert [tensor.item() for tensor in retained] == [1, 2]
    assert diagnostics.owned_shards_opened == diagnostics.owned_shards_released == 2


def test_safe_open_copy_clones_before_borrowed_context_closes(tmp_path):
    source = torch.tensor([11])
    state = {"closed": False}

    @contextmanager
    def original_safe_open(path, framework="pt", device="cpu", **kwargs):
        class Borrowed:
            def keys(self):
                assert not state["closed"]
                return ["weight"]

            def get_tensor(self, name):
                assert not state["closed"]
                return source

        try:
            yield Borrowed()
        finally:
            state["closed"] = True

    diagnostics = OwnedLoaderDiagnostics(True, "win32", loader_mode="safe-open-copy")
    owned_open = make_owned_safe_open(original_safe_open, None, "safe-open-copy", diagnostics)
    path = tmp_path / "shard.safetensors"
    path.write_bytes(b"x")
    with owned_open(path) as shard:
        retained = shard.get_tensor("weight")
    source.fill_(99)
    assert state["closed"] is True
    assert retained.item() == 11
    assert retained.data_ptr() != source.data_ptr()


def _fake_modules(monkeypatch, load_file):
    @contextmanager
    def safe_open(*args, **kwargs):
        yield OwnedShard({})

    original_safetensors = SimpleNamespace(safe_open=safe_open, marker="delegated")
    qwen_weight = SimpleNamespace(safetensors=original_safetensors)
    modules = {
        "safetensors": SimpleNamespace(__version__="test"),
        "safetensors.torch": SimpleNamespace(load_file=load_file),
        "freetoken.models.qwen3_5_moe.weight": qwen_weight,
    }
    monkeypatch.setattr(windows_loader.importlib, "import_module", lambda name: modules[name])
    return qwen_weight, original_safetensors


def test_windows_selects_pread_and_restores_reference_on_error(monkeypatch, tmp_path):
    def load_file(path, device="cpu", backend=None):
        return {"weight": torch.tensor([3])}

    qwen_weight, original = _fake_modules(monkeypatch, load_file)
    with pytest.raises(RuntimeError, match="iteration failed"):
        with windows_safe_freetoken_loader(platform="win32") as diagnostics:
            assert diagnostics.loader_mode == "pread"
            assert diagnostics.backend_support is True
            assert qwen_weight.safetensors is not original
            assert qwen_weight.safetensors.marker == "delegated"
            raise RuntimeError("iteration failed")
    assert qwen_weight.safetensors is original


def test_windows_selects_safe_open_copy_without_backend(monkeypatch):
    def load_file(path, device="cpu"):
        raise AssertionError("fallback mode must use safe_open")

    qwen_weight, original = _fake_modules(monkeypatch, load_file)
    with windows_safe_freetoken_loader(platform="win32") as diagnostics:
        assert diagnostics.loader_mode == "safe-open-copy"
        assert diagnostics.backend_support is False
    assert qwen_weight.safetensors is original


def test_non_windows_is_noop_and_imports_nothing(monkeypatch):
    monkeypatch.setattr(
        windows_loader.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError(f"unexpected import: {name}")),
    )
    with windows_safe_freetoken_loader(platform="linux") as diagnostics:
        assert diagnostics.to_dict()["loader_mode"] == "native"
        assert diagnostics.enabled is False


def test_runtime_adapter_does_not_install_preparation_workaround():
    import integration.freetoken_qcc.low_vram.adapter as runtime_adapter

    assert not hasattr(runtime_adapter, "windows_safe_freetoken_loader")
