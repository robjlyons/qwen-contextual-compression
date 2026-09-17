from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import integration.freetoken_qcc.low_vram.windows_loader as windows_loader
from integration.freetoken_qcc.low_vram.windows_loader import (
    LazyOwnedShard,
    OwnedLoaderDiagnostics,
    make_owned_safe_open,
    windows_safe_freetoken_loader,
)


class CountingReader:
    def __init__(self, tensors):
        self.tensors = tensors
        self.keys_calls = 0
        self.get_calls = []

    def keys(self):
        self.keys_calls += 1
        return self.tensors.keys()

    def get_tensor(self, name):
        self.get_calls.append(name)
        return self.tensors[name]


def test_lazy_owned_shard_keys_do_not_load_or_clone():
    reader = CountingReader({"a": torch.tensor([1]), "b": torch.tensor([2])})
    diagnostics = OwnedLoaderDiagnostics(True, "win32", loader_mode="lazy-safe-open-clone")
    shard = LazyOwnedShard(reader, diagnostics)
    assert list(shard.keys()) == ["a", "b"]
    assert reader.keys_calls == 1
    assert reader.get_calls == []
    assert diagnostics.owned_tensors_copied == 0
    assert diagnostics.owned_tensor_bytes == 0


def test_opening_owned_context_and_keys_remain_lazy(tmp_path):
    path = tmp_path / "lazy.safetensors"
    path.write_bytes(b"source")
    reader = CountingReader({str(index): torch.tensor([index]) for index in range(100)})
    closed = []
    diagnostics = OwnedLoaderDiagnostics(True, "win32", loader_mode="lazy-safe-open-clone")
    owned_open = make_owned_safe_open(_safe_open_for({str(path): reader}, closed), diagnostics)
    with owned_open(path) as shard:
        assert reader.get_calls == []
        assert diagnostics.owned_tensors_copied == 0
        assert len(list(shard.keys())) == 100
        assert reader.get_calls == []
        assert diagnostics.owned_tensors_copied == 0
    assert diagnostics.owned_shards_opened == diagnostics.owned_shards_released == 1


def test_one_requested_tensor_clones_only_that_tensor():
    sources = {str(index): torch.tensor([index], dtype=torch.float32) for index in range(20)}
    reader = CountingReader(sources)
    diagnostics = OwnedLoaderDiagnostics(True, "win32", loader_mode="lazy-safe-open-clone")
    shard = LazyOwnedShard(reader, diagnostics)
    owned = shard.get_tensor("7")
    assert reader.get_calls == ["7"]
    assert diagnostics.owned_tensors_copied == 1
    assert diagnostics.owned_tensor_bytes == owned.numel() * owned.element_size()
    assert diagnostics.largest_owned_tensor_bytes == owned.numel() * owned.element_size()
    assert owned.data_ptr() != sources["7"].data_ptr()
    assert owned.dtype == sources["7"].dtype
    assert owned.shape == sources["7"].shape
    assert torch.equal(owned, sources["7"])


def _safe_open_for(readers, closed):
    @contextmanager
    def safe_open(path, framework="pt", device="cpu", **kwargs):
        reader = readers[str(path)]
        try:
            yield reader
        finally:
            closed.append(str(path))

    return safe_open


def test_owned_tensor_survives_source_context_closure(tmp_path):
    path = tmp_path / "one.safetensors"
    path.write_bytes(b"source")
    source = torch.tensor([11, 12])
    reader = CountingReader({"weight": source})
    closed = []
    diagnostics = OwnedLoaderDiagnostics(True, "win32", loader_mode="lazy-safe-open-clone")
    owned_open = make_owned_safe_open(_safe_open_for({str(path): reader}, closed), diagnostics)
    with owned_open(path) as shard:
        retained = shard.get_tensor("weight")
    source.fill_(99)
    assert closed == [str(path)]
    assert retained.tolist() == [11, 12]
    assert retained.data_ptr() != source.data_ptr()
    assert diagnostics.owned_shards_opened == diagnostics.owned_shards_released == 1
    assert diagnostics.owned_source_bytes == len(b"source")


def test_cross_shard_retained_tensors_remain_valid(tmp_path):
    paths = [tmp_path / "1.safetensors", tmp_path / "2.safetensors"]
    for path in paths:
        path.write_bytes(b"x")
    sources = [torch.tensor([3]), torch.tensor([4])]
    readers = {str(path): CountingReader({"weight": source}) for path, source in zip(paths, sources)}
    closed = []
    diagnostics = OwnedLoaderDiagnostics(True, "win32", loader_mode="lazy-safe-open-clone")
    owned_open = make_owned_safe_open(_safe_open_for(readers, closed), diagnostics)
    retained = []
    for path in paths:
        with owned_open(path) as shard:
            retained.append(shard.get_tensor("weight"))
    for source in sources:
        source.zero_()
    assert [tensor.item() for tensor in retained] == [3, 4]
    assert diagnostics.owned_tensors_copied == 2
    assert diagnostics.owned_shards_opened == diagnostics.owned_shards_released == 2
    diagnostics.assert_balanced()


def _fake_modules(monkeypatch):
    @contextmanager
    def safe_open(*args, **kwargs):
        yield CountingReader({})

    original_safetensors = SimpleNamespace(safe_open=safe_open, marker="delegated", __version__="test")
    qwen_weight = SimpleNamespace(safetensors=original_safetensors)
    modules = {
        "safetensors": original_safetensors,
        "freetoken.models.qwen3_5_moe.weight": qwen_weight,
    }
    monkeypatch.setattr(windows_loader.importlib, "import_module", lambda name: modules[name])
    return qwen_weight, original_safetensors


def test_windows_selects_lazy_clone_and_restores_reference_on_error(monkeypatch):
    qwen_weight, original = _fake_modules(monkeypatch)
    with pytest.raises(RuntimeError, match="iteration failed"):
        with windows_safe_freetoken_loader(platform="win32") as diagnostics:
            assert diagnostics.loader_mode == "lazy-safe-open-clone"
            assert diagnostics.load_file_signature is None
            assert diagnostics.backend_support is False
            assert qwen_weight.safetensors is not original
            assert qwen_weight.safetensors.marker == "delegated"
            raise RuntimeError("iteration failed")
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
