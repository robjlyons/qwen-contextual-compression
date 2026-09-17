"""Preparation-only owned-storage adapter for FreeToken's Windows shard reader."""
from __future__ import annotations

import importlib
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class WindowsLoaderCompatibilityError(RuntimeError):
    pass


class LazyOwnedShard:
    """A ``safe_open`` reader that clones only tensors requested by FreeToken."""

    def __init__(self, reader, diagnostics):
        self._reader = reader
        self._diagnostics = diagnostics

    def keys(self):
        return self._reader.keys()

    def get_tensor(self, name):
        source = self._reader.get_tensor(name)
        owned = source.clone()
        byte_size = owned.numel() * owned.element_size()
        self._diagnostics.owned_tensors_copied += 1
        self._diagnostics.owned_tensor_bytes += byte_size
        self._diagnostics.largest_owned_tensor_bytes = max(
            self._diagnostics.largest_owned_tensor_bytes, byte_size
        )
        return owned


@dataclass
class OwnedLoaderDiagnostics:
    enabled: bool
    platform: str
    safetensors_version: str | None = None
    load_file_signature: str | None = None
    backend_support: bool = False
    loader_mode: str = "native"
    owned_shards_opened: int = 0
    owned_shards_released: int = 0
    owned_source_bytes: int = 0
    owned_tensors_copied: int = 0
    owned_tensor_bytes: int = 0
    largest_owned_tensor_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "platform": self.platform,
            "safetensors_version": self.safetensors_version,
            "load_file_signature": self.load_file_signature,
            "backend_support": self.backend_support,
            "loader_mode": self.loader_mode,
            "owned_shards_opened": self.owned_shards_opened,
            "owned_shards_released": self.owned_shards_released,
            "owned_source_bytes": self.owned_source_bytes,
            "owned_tensors_copied": self.owned_tensors_copied,
            "owned_tensor_bytes": self.owned_tensor_bytes,
            "largest_owned_tensor_bytes": self.largest_owned_tensor_bytes,
        }

    def assert_balanced(self) -> None:
        if self.enabled and self.owned_shards_opened != self.owned_shards_released:
            raise WindowsLoaderCompatibilityError(
                "Owned shard context imbalance: "
                f"opened={self.owned_shards_opened}, released={self.owned_shards_released}"
            )


class _SafeTensorsProxy:
    def __init__(self, original, safe_open):
        self._original = original
        self.safe_open = safe_open

    def __getattr__(self, name):
        return getattr(self._original, name)


def _is_cpu_pt(framework, device) -> bool:
    device_type = getattr(device, "type", device)
    return framework == "pt" and str(device_type) == "cpu"


def _source_size(path) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def make_owned_safe_open(original_safe_open, diagnostics: OwnedLoaderDiagnostics):
    """Return a context-manager function compatible with ``safetensors.safe_open``."""
    shard_index = 0

    @contextmanager
    def owned_safe_open(path, framework="pt", device="cpu", **kwargs):
        nonlocal shard_index
        if not _is_cpu_pt(framework, device):
            with original_safe_open(path, framework=framework, device=device, **kwargs) as reader:
                yield reader
            return

        shard_index += 1
        index = shard_index
        source_bytes = _source_size(path)
        diagnostics.owned_shards_opened += 1
        diagnostics.owned_source_bytes += source_bytes
        print(
            "QCC OWNED SHARD OPEN: "
            f"index={index} size_gib={source_bytes / 2**30:.3f} mode={diagnostics.loader_mode}",
            flush=True,
        )
        try:
            with original_safe_open(path, framework="pt", device="cpu", **kwargs) as reader:
                yield LazyOwnedShard(reader, diagnostics)
        finally:
            diagnostics.owned_shards_released += 1
            print(f"QCC OWNED SHARD RELEASE: index={index}", flush=True)

    return owned_safe_open


@contextmanager
def windows_safe_freetoken_loader(platform: str | None = None):
    """Temporarily patch only the Qwen loader's local safetensors reference."""
    selected_platform = platform or sys.platform
    diagnostics = OwnedLoaderDiagnostics(enabled=False, platform=selected_platform)
    if selected_platform != "win32":
        yield diagnostics
        return

    safetensors = importlib.import_module("safetensors")
    qwen_weight = importlib.import_module("freetoken.models.qwen3_5_moe.weight")
    diagnostics = OwnedLoaderDiagnostics(
        enabled=True,
        platform=selected_platform,
        safetensors_version=getattr(safetensors, "__version__", None),
        loader_mode="lazy-safe-open-clone",
    )
    print("QCC WINDOWS-SAFE FREETOKEN LOADER: " + json.dumps(diagnostics.to_dict(), sort_keys=True), flush=True)

    module_reference = getattr(qwen_weight, "safetensors", None)
    direct_reference = getattr(qwen_weight, "safe_open", None)
    if module_reference is not None and callable(getattr(module_reference, "safe_open", None)):
        attribute = "safetensors"
        original = module_reference
        original_safe_open = module_reference.safe_open
        replacement = _SafeTensorsProxy(
            module_reference,
            make_owned_safe_open(original_safe_open, diagnostics),
        )
    elif callable(direct_reference):
        attribute = "safe_open"
        original = direct_reference
        replacement = make_owned_safe_open(direct_reference, diagnostics)
    else:
        raise WindowsLoaderCompatibilityError(
            "Installed FreeToken Qwen loader exposes neither safetensors.safe_open nor a local safe_open reference"
        )

    setattr(qwen_weight, attribute, replacement)
    try:
        yield diagnostics
    finally:
        setattr(qwen_weight, attribute, original)
