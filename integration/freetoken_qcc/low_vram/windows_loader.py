"""Preparation-only owned-storage adapter for FreeToken's Windows shard reader."""
from __future__ import annotations

import importlib
import inspect
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class WindowsLoaderCompatibilityError(RuntimeError):
    pass


class OwnedShard:
    """The small ``safe_open`` reader surface used by FreeToken."""

    def __init__(self, tensors):
        self._tensors = tensors

    def keys(self):
        return self._tensors.keys()

    def get_tensor(self, name):
        return self._tensors[name]


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


def make_owned_safe_open(original_safe_open, load_file, mode: str, diagnostics: OwnedLoaderDiagnostics):
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
            f"index={index} size_gib={source_bytes / 2**30:.3f} mode={mode}",
            flush=True,
        )
        tensors = None
        try:
            if mode == "pread":
                tensors = load_file(path, device="cpu", backend="pread")
            elif mode == "safe-open-copy":
                with original_safe_open(path, framework="pt", device="cpu", **kwargs) as reader:
                    tensors = {name: reader.get_tensor(name).clone() for name in reader.keys()}
            else:
                raise WindowsLoaderCompatibilityError(f"Unsupported owned shard loader mode: {mode}")
            yield OwnedShard(tensors)
        finally:
            tensors = None
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
    safetensors_torch = importlib.import_module("safetensors.torch")
    qwen_weight = importlib.import_module("freetoken.models.qwen3_5_moe.weight")
    load_file = getattr(safetensors_torch, "load_file", None)
    if not callable(load_file):
        raise WindowsLoaderCompatibilityError("safetensors.torch.load_file is unavailable")
    try:
        signature = inspect.signature(load_file)
    except (TypeError, ValueError) as error:
        raise WindowsLoaderCompatibilityError("Cannot inspect safetensors.torch.load_file") from error
    backend_support = "backend" in signature.parameters
    mode = "pread" if backend_support else "safe-open-copy"
    diagnostics = OwnedLoaderDiagnostics(
        enabled=True,
        platform=selected_platform,
        safetensors_version=getattr(safetensors, "__version__", None),
        load_file_signature=str(signature),
        backend_support=backend_support,
        loader_mode=mode,
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
            make_owned_safe_open(original_safe_open, load_file, mode, diagnostics),
        )
    elif callable(direct_reference):
        attribute = "safe_open"
        original = direct_reference
        replacement = make_owned_safe_open(direct_reference, load_file, mode, diagnostics)
    else:
        raise WindowsLoaderCompatibilityError(
            "Installed FreeToken Qwen loader exposes neither safetensors.safe_open nor a local safe_open reference"
        )

    setattr(qwen_weight, attribute, replacement)
    try:
        yield diagnostics
    finally:
        setattr(qwen_weight, attribute, original)
