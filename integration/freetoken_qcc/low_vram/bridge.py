"""Model-object residency bridge; checkpoint interception is adapter-specific."""
from __future__ import annotations

import json
import time

import torch

from integration.freetoken_qcc.low_vram.embedding import HostBackedEmbedding
from integration.freetoken_qcc.low_vram.staging import LayerStager


class LowVRAMModelBridge:
    """Attach deterministic staging to already-created FreeToken decoder layers.

    This deliberately does not claim to solve checkpoint construction.  A
    version-specific loader adapter must construct layer objects without first
    materializing the full checkpoint on CUDA.
    """

    def __init__(self, host_store, device):
        self.host_store = host_store
        self.device = torch.device(device)
        self.stager = LayerStager(host_store, self.device, slots=2, asynchronous=False)
        self.prefill_executions = 0
        self.decode_executions = 0
        self.requests = 0
        self.finite_outputs = True
        self.started = time.perf_counter()
        self._patched = []

    def host_embedding(self) -> HostBackedEmbedding:
        return HostBackedEmbedding(self.host_store.embedding_weight(), self.device)

    def attach_layers(self, layers) -> None:
        if len(layers) != self.host_store.layer_count:
            raise ValueError(f"model has {len(layers)} layers but stream cache has {self.host_store.layer_count}")
        for layer_id, layer in enumerate(layers):
            original = layer.forward

            def streamed_forward(*args, _layer=layer, _id=layer_id, _original=original, **kwargs):
                slot = _id % 2
                self.stager.stage(_id, slot)
                self.stager.wait(slot)
                self.stager.activate(_id, slot, _layer)
                try:
                    output = _original(*args, **kwargs)
                    tensor = output[0] if isinstance(output, tuple) else output
                    if isinstance(tensor, torch.Tensor):
                        self.finite_outputs = self.finite_outputs and bool(torch.isfinite(tensor).all())
                    if args and isinstance(args[0], torch.Tensor) and args[0].shape[0] == 1:
                        self.decode_executions += 1
                    else:
                        self.prefill_executions += 1
                    return output
                finally:
                    self.stager.deactivate(slot)

            layer.forward = streamed_forward
            self._patched.append((layer, original))

    def restore(self) -> None:
        for layer, original in self._patched:
            layer.forward = original
        self._patched.clear()

    def summary(self) -> dict:
        result = self.stager.summary()
        result.update(
            requests=self.requests,
            prefill_executions=self.prefill_executions,
            decode_executions=self.decode_executions,
            finite_outputs=self.finite_outputs,
            elapsed_seconds=time.perf_counter() - self.started,
        )
        return result

    def print_startup(self, model, freetoken_version) -> None:
        print(
            "QCC LOW-VRAM FREETOKEN ACTIVE: "
            + json.dumps(
                {
                    "model": model,
                    "freetoken_version": freetoken_version,
                    "layers": self.host_store.layer_count,
                    "host_transformer_bytes": self.host_store.host_transformer_bytes,
                    "host_embedding_bytes": self.host_store.host_embedding_bytes,
                    "staging_slots": 2,
                    "async": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def close(self) -> None:
        print("LOW-VRAM FINAL SUMMARY: " + json.dumps(self.summary(), sort_keys=True), flush=True)
