"""In-process monkeypatch bridge from FreeToken's dense Qwen MLP to QCC layer 0."""
from __future__ import annotations

import atexit
import importlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from integration.freetoken_qcc.metrics import ShadowMetrics


EXPECTED_HIDDEN = 5120
EXPECTED_INTERMEDIATE = 17408
SUPPORTED_MODES = {"off", "shadow", "replace"}
_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp(?:\.|$)")


@dataclass(frozen=True)
class FreeTokenQCCConfig:
    mode: str = "off"
    layer: int = 0
    selector_checkpoint: Path | None = None
    runtime_weights: Path | None = None
    variant: str = "warp8"
    retention: float = 0.5
    metrics: Path | None = None
    log_every: int = 16

    @classmethod
    def from_env(cls, environ=None) -> "FreeTokenQCCConfig":
        environment = os.environ if environ is None else environ
        mode = environment.get("QCC_FT_MODE", "off").lower()
        try:
            config = cls(
                mode=mode,
                layer=int(environment.get("QCC_FT_LAYER", "0")),
                selector_checkpoint=_optional_path(environment.get("QCC_FT_SELECTOR_CHECKPOINT")),
                runtime_weights=_optional_path(environment.get("QCC_FT_RUNTIME_WEIGHTS")),
                variant=environment.get("QCC_FT_VARIANT", "warp8"),
                retention=float(environment.get("QCC_FT_RETENTION", "0.50")),
                metrics=_optional_path(environment.get("QCC_FT_METRICS")),
                log_every=int(environment.get("QCC_FT_LOG_EVERY", "16")),
            )
        except ValueError as error:
            raise ValueError(f"Invalid QCC FreeToken environment: {error}") from error
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"QCC_FT_MODE must be one of {sorted(SUPPORTED_MODES)}, got {self.mode!r}")
        if self.layer < 0 or self.log_every < 1:
            raise ValueError("QCC_FT_LAYER must be non-negative and QCC_FT_LOG_EVERY must be positive")
        if self.retention != 0.5:
            raise ValueError("Phase 5D-A requires QCC_FT_RETENTION=0.50")
        if self.variant != "warp8":
            raise ValueError("Phase 5D-A requires the proven QCC_FT_VARIANT=warp8")
        if self.mode != "off":
            for label, path in (("QCC_FT_SELECTOR_CHECKPOINT", self.selector_checkpoint), ("QCC_FT_RUNTIME_WEIGHTS", self.runtime_weights)):
                if path is None:
                    raise ValueError(f"{label} is required in {self.mode} mode")
                if not path.is_file():
                    raise FileNotFoundError(f"{label} does not exist: {path}")


def _optional_path(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


def parse_dense_mlp_layer(mlp) -> int | None:
    """Read the layer from FreeToken-owned prefixes, without treating experts as dense MLPs."""
    prefixes = []
    for owner in (mlp, getattr(mlp, "gate_up_proj", None), getattr(mlp, "down_proj", None)):
        prefix = getattr(owner, "prefix", None)
        if isinstance(prefix, str):
            prefixes.append(prefix)
    for prefix in prefixes:
        if ".experts." in prefix or ".shared_expert" in prefix:
            continue
        match = _LAYER_PATTERN.search(prefix)
        if match:
            return int(match.group(1))
    return None


class LazyQCCRuntime:
    """Own the additional FP16 layer and frozen selector, initialized on first decode."""

    def __init__(self, config: FreeTokenQCCConfig):
        self.config = config
        self.device = None
        self.memory: dict = {}
        self.initialized = False
        self._initialize_lock = threading.Lock()

    def initialize(self, device: torch.device) -> None:
        if self.initialized:
            if self.device != device:
                raise RuntimeError(f"QCC runtime initialized on {self.device}, but received input on {device}")
            return
        with self._initialize_lock:
            if self.initialized:
                if self.device != device:
                    raise RuntimeError(f"QCC runtime initialized on {self.device}, but received input on {device}")
                return
            self._initialize(device)

    def _initialize(self, device: torch.device) -> None:
        from runtime.cuda_indexed import VARIANTS
        from runtime.cuda_selector import CudaSelector
        from runtime.selector import RuntimeSelector, selector_variants
        from runtime.weights import load_runtime_weights, validate_weight_shapes

        if device.type != "cuda":
            raise RuntimeError("QCC FreeToken replacement requires a CUDA input tensor")
        if self.config.variant not in VARIANTS:
            raise ValueError(f"Unknown CUDA indexed variant: {self.config.variant}")
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        weights, metadata = load_runtime_weights(self.config.runtime_weights, device)
        hidden, intermediate = validate_weight_shapes(weights)
        if (hidden, intermediate) != (EXPECTED_HIDDEN, EXPECTED_INTERMEDIATE):
            raise ValueError(f"QCC runtime weights are {(hidden, intermediate)}, expected {(EXPECTED_HIDDEN, EXPECTED_INTERMEDIATE)}")
        if any(tensor.dtype != torch.float16 for tensor in weights.values()):
            raise ValueError("Phase 5D-A runtime weights must be FP16")
        selector = RuntimeSelector.from_checkpoint(self.config.selector_checkpoint, self.config.retention, device)
        folded = selector_variants(selector)["fp16_folded_norm"].to(device).eval()
        if folded.predictor.encoder.in_features != hidden or folded.predictor.neurons.out_features != intermediate:
            raise ValueError("Selector dimensions do not match Qwen3.8-27B layer 0")
        self.weights = weights
        self.selector = CudaSelector(folded)
        expected_k = EXPECTED_INTERMEDIATE // 2
        if self.selector.k != expected_k:
            raise ValueError(f"Selector chooses {self.selector.k} neurons, expected {expected_k} at 50% retention")
        self.device = device
        self.initialized = True
        allocated_after = torch.cuda.memory_allocated(device)
        reserved_after = torch.cuda.memory_reserved(device)
        weight_bytes = sum(tensor.numel() * tensor.element_size() for tensor in weights.values())
        selector_bytes = sum(tensor.numel() * tensor.element_size() for tensor in folded.state_dict().values())
        self.memory = {
            "allocated_before": allocated_before,
            "allocated_after": allocated_after,
            "reserved_before": reserved_before,
            "reserved_after": reserved_after,
            "qcc_added_allocated_mib": (allocated_after - allocated_before) / 2**20,
            "ffn_weight_bytes": weight_bytes,
            "selector_bytes": selector_bytes,
            "metadata": metadata,
            "warning": "Phase 5D-A duplicates one FP16 FFN layer; it does not save model VRAM.",
        }
        print("QCC FreeToken lazy initialization: " + json.dumps(self.memory, default=str, sort_keys=True))

    @torch.inference_mode()
    def run(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.initialize(x.device)
        from runtime.cuda_indexed import cuda_indexed_ffn

        x_fp16 = x.to(dtype=torch.float16).contiguous()
        selected_ids = self.selector.select(x_fp16)[0].contiguous()
        output = cuda_indexed_ffn(
            x_fp16,
            selected_ids,
            self.weights["gate_proj.weight"],
            self.weights["up_proj.weight"],
            self.weights["down_proj.weight"],
            variant=self.config.variant,
        )
        return output.to(dtype=x.dtype), selected_ids


class FreeTokenQCCBridge:
    def __init__(self, config: FreeTokenQCCConfig, runtime_factory: Callable | None = None):
        config.validate()
        self.config = config
        self.runtime_factory = runtime_factory or LazyQCCRuntime
        self._runtime = None
        self._runtime_lock = threading.Lock()
        self.metrics = ShadowMetrics(config.metrics)
        self.counters = {
            "sparse_decode_calls": 0,
            "shadow_decode_calls": 0,
            "dense_fallback_calls": 0,
            "untargeted_layer_fallbacks": 0,
            "prefill_fallbacks": 0,
            "wrong_shape_fallbacks": 0,
        }
        self.closed = False
        self.integration = {"freetoken_version": None, "architecture": None}

    @property
    def runtime(self):
        if self._runtime is None:
            with self._runtime_lock:
                if self._runtime is None:
                    self._runtime = self.runtime_factory(self.config)
        return self._runtime

    def _fallback(self, reason: str, original, instance, x, args, kwargs):
        self.counters["dense_fallback_calls"] += 1
        self.counters[reason] += 1
        return original(instance, x, *args, **kwargs)

    def forward(self, original, instance, x, *args, **kwargs):
        if self.config.mode == "off":
            return original(instance, x, *args, **kwargs)
        layer = parse_dense_mlp_layer(instance)
        if layer != self.config.layer:
            return self._fallback("untargeted_layer_fallbacks", original, instance, x, args, kwargs)
        if x.ndim == 2 and x.shape[0] > 1:
            return self._fallback("prefill_fallbacks", original, instance, x, args, kwargs)
        if x.ndim != 2 or tuple(x.shape) != (1, EXPECTED_HIDDEN):
            return self._fallback("wrong_shape_fallbacks", original, instance, x, args, kwargs)
        if self.config.mode == "shadow":
            stock = original(instance, x, *args, **kwargs)
            sparse, selected_ids = self.runtime.run(x)
            self.counters["shadow_decode_calls"] += 1
            call_index = self.counters["shadow_decode_calls"]
            self.metrics.record(
                stock,
                sparse,
                layer=layer,
                call_index=call_index,
                input_dtype=str(x.dtype),
                input_shape=list(x.shape),
                k=int(selected_ids.numel()),
                retention=self.config.retention,
            )
            if call_index % self.config.log_every == 0:
                print("QCC shadow summary: " + json.dumps(self.metrics.summary(), sort_keys=True))
            return stock
        sparse, _ = self.runtime.run(x)
        self.counters["sparse_decode_calls"] += 1
        return sparse

    def summary(self) -> dict:
        return {
            "mode": self.config.mode,
            "target_layer": self.config.layer,
            "counters": dict(self.counters),
            "shadow": self.metrics.summary(),
            "memory": getattr(self._runtime, "memory", None),
            "integration": dict(self.integration),
            "scope_warning": "Single Qwen3.8 layer-0 integration only; no full-model throughput or VRAM claim.",
        }

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            print("QCC FreeToken final summary: " + json.dumps(self.summary(), default=str, sort_keys=True))


def install_patch(config: FreeTokenQCCConfig | None = None, dense_mlp_class=None, runtime_factory=None) -> FreeTokenQCCBridge:
    """Patch only FreeToken's dense Qwen3.5/3.8 MLP class for this process."""
    config = config or FreeTokenQCCConfig.from_env()
    bridge = FreeTokenQCCBridge(config, runtime_factory)
    if config.mode == "off":
        return bridge
    if dense_mlp_class is None:
        freetoken = importlib.import_module("freetoken")
        module = importlib.import_module("freetoken.models.qwen3_5_moe.moe")
        dense_mlp_class = module.Qwen3_5DenseMLP
        bridge.integration = {
            "freetoken_version": getattr(freetoken, "__version__", "unknown"),
            "architecture": f"{dense_mlp_class.__module__}.{dense_mlp_class.__name__}",
        }
        print("QCC FreeToken integration detected: " + json.dumps(bridge.integration, sort_keys=True))
    existing = getattr(dense_mlp_class, "_qcc_original_forward", None)
    if existing is not None:
        raise RuntimeError("Qwen3_5DenseMLP is already patched by QCC in this process")
    original = dense_mlp_class.forward

    def qcc_forward(instance, x, *args, **kwargs):
        return bridge.forward(original, instance, x, *args, **kwargs)

    dense_mlp_class._qcc_original_forward = original
    dense_mlp_class.forward = qcc_forward
    atexit.register(bridge.close)
    return bridge
