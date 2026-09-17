"""Version-specific FreeToken 0.1.2 low-VRAM loader interception."""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

import torch

from integration.freetoken_qcc.low_vram.bridge import LowVRAMModelBridge
from integration.freetoken_qcc.low_vram.host_store import TransformerHostStore
from integration.freetoken_qcc.low_vram.manifest import REPRESENTATION
from integration.freetoken_qcc.low_vram.staging import _resolve


CRITICAL_FINGERPRINTS = {
    "models/qwen3_5_moe/model.py": "a892015b6d01762a5609a9839bbf3d1ab1323e1d2f638ad5bba80af7c38304eb",
    "models/qwen3_5_moe/moe.py": "04c6c25e027401d57e08fe6403a4db948421a3db82def00f9ed80bb575363307",
    "models/qwen3_5_moe/quant_linear.py": "e578ce67d563684c305c4f98d639c901cab9fa868921aaf8b876b4424f3fee02",
    "kernel/triton/nvfp4_linear.py": "6eb4e9e954bc82e7a3b764c3238b45a4b1117735397b0f515665e8201eed3615",
    "layers/base.py": "2080bdf735c51592199cc9dd6322eb0b40fb1ed38ca39375b56b600dbf151037",
    "layers/embedding.py": "2bf326c2905e7138ddaadf287f4ab9de4de4f8a52dcc6d800ff5774e90331ba0",
}


class LowVRAMCompatibilityError(RuntimeError):
    pass


def inspect_installed_contract() -> dict:
    freetoken = importlib.import_module("freetoken")
    models = importlib.import_module("freetoken.models")
    weight = importlib.import_module("freetoken.models.qwen3_5_moe.weight")
    engine_module = importlib.import_module("freetoken.engine.engine")
    scheduler = importlib.import_module("freetoken.scheduler.scheduler")
    callables = {
        "freetoken.models.load_weight": getattr(models, "load_weight", None),
        "qwen_weight.iter_weights": getattr(weight, "iter_weights", None),
        "Engine._load_weight_state_dict": getattr(engine_module.Engine, "_load_weight_state_dict", None),
    }
    signatures = {name: str(inspect.signature(value)) if callable(value) else None for name, value in callables.items()}
    sources = {}
    for name, value in callables.items():
        if not callable(value):
            sources[name] = None
            continue
        try:
            sources[name] = inspect.getsource(value)
        except (OSError, TypeError):
            sources[name] = None
    return {
        "version": getattr(freetoken, "__version__", "unknown"),
        "files": {"qwen_weight": weight.__file__, "engine": engine_module.__file__, "scheduler": scheduler.__file__},
        "signatures": signatures,
        "sources": sources,
    }


def validate_fingerprint() -> dict:
    freetoken = importlib.import_module("freetoken")
    root = Path(freetoken.__file__).resolve().parent
    results = {}
    for relative, expected in CRITICAL_FINGERPRINTS.items():
        path = root / relative
        if not path.is_file():
            results[relative] = "source-unavailable"
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise LowVRAMCompatibilityError(f"Behavior-critical FreeToken fingerprint mismatch: {relative}: {actual}")
        results[relative] = "matched"
    return results


def _bind_resident(model, store, device) -> int:
    resident = store.resident()
    total = 0
    for name, path in resident.bindings.items():
        tensor = resident.tensors[name].to(device)
        owner, attribute = _resolve(model, path)
        setattr(owner, attribute, tensor)
        total += tensor.numel() * tensor.element_size()
    return total


class FreeTokenLowVRAMAdapter:
    def __init__(self, config, expected_model=None):
        self.config = config
        self.store = TransformerHostStore(config.stream_cache, expected_model=expected_model, validate_checksums=True)
        self.contract = inspect_installed_contract()
        self.fingerprint = validate_fingerprint()
        manifest_fingerprint = self.store.manifest["adapter_fingerprint"]
        if manifest_fingerprint != self.fingerprint:
            raise LowVRAMCompatibilityError("Stream cache adapter fingerprint does not match installed FreeToken")
        self.bridge = None
        self.original_engine_loader = None
        self.original_model_loader = None
        self.original_finalize_quant = None
        self.engine_module = None
        self.model_class = None

    def install(self):
        if int(os.environ.get("WORLD_SIZE", os.environ.get("TP_SIZE", "1"))) != 1:
            raise LowVRAMCompatibilityError("Phase 5E-A.1 supports TP=1 only")
        engine_module = importlib.import_module("freetoken.engine.engine")
        model_module = importlib.import_module("freetoken.models.qwen3_5_moe.model")
        model_class = getattr(model_module, "Qwen3_5MoEForCausalLM", None)
        if model_class is None:
            raise LowVRAMCompatibilityError("Installed FreeToken lacks Qwen3_5MoEForCausalLM")
        loader = getattr(engine_module.Engine, "_load_weight_state_dict", None)
        if not callable(loader):
            raise LowVRAMCompatibilityError("Installed Engine lacks _load_weight_state_dict")
        finalize_quant = getattr(engine_module, "finalize_quant", None)
        if finalize_quant is not None and not callable(finalize_quant):
            raise LowVRAMCompatibilityError("Installed Engine finalize_quant symbol is not callable")
        self.engine_module = engine_module
        self.model_class = model_class
        self.original_engine_loader = loader
        self.original_model_loader = model_class.load_state_dict
        self.original_finalize_quant = finalize_quant
        adapter = self

        def suppress_full_cuda_loader(engine, *args, **kwargs):
            # Returning an empty mapping is intentional: the exact Qwen model
            # load hook below installs finalized cache tensors instead.
            return {}

        def install_cache(model, state_dict, *args, **kwargs):
            if state_dict:
                raise LowVRAMCompatibilityError("Normal checkpoint loader was not suppressed")
            adapter.install_model(model)
            return None

        def guarded_finalize_quant(root, *args, **kwargs):
            bridge = getattr(root, "_qcc_low_vram_bridge", None)
            if bridge is not None:
                representation = adapter.store.manifest.get("representation")
                if representation != REPRESENTATION:
                    raise LowVRAMCompatibilityError(
                        "Refusing to skip finalize_quant for a non-finalized cache"
                    )
                print(
                    "QCC LOW-VRAM FINALIZE_QUANT SKIPPED:\n"
                    "cache already contains finalized runtime tensors",
                    flush=True,
                )
                return 0
            return adapter.original_finalize_quant(root, *args, **kwargs)

        engine_module.Engine._load_weight_state_dict = suppress_full_cuda_loader
        model_class.load_state_dict = install_cache
        if finalize_quant is not None:
            engine_module.finalize_quant = guarded_finalize_quant
            print("QCC LOW-VRAM FINALIZE_QUANT GUARD INSTALLED", flush=True)
        return self

    def install_model(self, model):
        device = torch.device("cuda", torch.cuda.current_device())
        inner = model.model
        layers = inner.layers.op_list
        if len(layers) != self.store.layer_count:
            raise LowVRAMCompatibilityError("FreeToken/cache layer-count mismatch")
        resident_bytes = _bind_resident(model, self.store, device)
        embedding = self.store.embedding_weight()
        original_embedding = inner.embed_tokens.forward
        bridge = LowVRAMModelBridge(self.store, device)
        host_embedding = bridge.host_embedding()
        inner.embed_tokens.forward = host_embedding
        bridge._patched.append((inner.embed_tokens, original_embedding))
        bridge.attach_layers(layers)
        bridge.attach_model_forward(inner)
        model._qcc_low_vram_bridge = bridge
        self.bridge = bridge
        print(
            "QCC LOW-VRAM MODEL INSTALLED: "
            + json.dumps({"resident_cuda_bytes": resident_bytes, "embedding_device": "cpu/cache", "layers": len(layers), "cache_representation": REPRESENTATION}, sort_keys=True),
            flush=True,
        )
        return bridge

    def close(self):
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
        if self.engine_module is not None:
            if self.original_engine_loader is not None:
                self.engine_module.Engine._load_weight_state_dict = self.original_engine_loader
            if self.original_finalize_quant is not None:
                self.engine_module.finalize_quant = self.original_finalize_quant
        if self.model_class is not None and self.original_model_loader is not None:
            self.model_class.load_state_dict = self.original_model_loader


def install_low_vram_adapter(config, expected_model=None):
    adapter = FreeTokenLowVRAMAdapter(config, expected_model)
    adapter.install()
    print(
        "QCC LOW-VRAM FREETOKEN ACTIVE: "
        + json.dumps({"pid": os.getpid(), "freetoken_version": adapter.contract["version"], "cache_path": str(config.stream_cache), "representation": adapter.store.manifest["representation"], "layers": adapter.store.layer_count, "staging_slots": 2, "async": False, "tp": 1, "fingerprint": adapter.fingerprint}, sort_keys=True),
        flush=True,
    )
    return adapter
