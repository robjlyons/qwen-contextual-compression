"""Pageable-CPU store for finalized per-layer FreeToken runtime tensors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from integration.freetoken_qcc.low_vram.manifest import load_manifest, validate_cache_files


@dataclass(frozen=True)
class HostLayer:
    layer_id: int
    tensors: dict[str, torch.Tensor]
    bindings: dict[str, str]
    byte_size: int

    def tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    def neuron_tensor(self, name: str, neuron_ids: torch.Tensor) -> torch.Tensor:
        """Future Phase 5E-B access without constraining the cache to whole layers."""
        return self.tensors[name].index_select(0, neuron_ids.cpu())


class TransformerHostStore:
    def __init__(self, root: Path, expected_model=None, validate_checksums=False):
        self.root = Path(root)
        self.manifest = load_manifest(self.root, expected_model)
        validate_cache_files(self.root, self.manifest, validate_checksums)
        self._layers: dict[int, HostLayer] = {}

    @property
    def layer_count(self) -> int:
        return self.manifest["layer_count"]

    @property
    def host_transformer_bytes(self) -> int:
        return sum(record["tensor_bytes"] for record in self.manifest["layers"])

    @property
    def host_embedding_bytes(self) -> int:
        return self.manifest["embedding"]["tensor_bytes"]

    def layer(self, layer_id: int) -> HostLayer:
        if not 0 <= layer_id < self.layer_count:
            raise IndexError(f"layer {layer_id} outside [0, {self.layer_count})")
        if layer_id not in self._layers:
            record = self.manifest["layers"][layer_id]
            tensors = load_file(str(self.root / record["file"]), device="cpu")
            if any(tensor.device.type != "cpu" for tensor in tensors.values()):
                raise RuntimeError("Stream-cache tensors must remain host-backed")
            self._layers[layer_id] = HostLayer(layer_id, tensors, record["bindings"], record["tensor_bytes"])
        return self._layers[layer_id]

    def embedding_weight(self) -> torch.Tensor:
        record = self.manifest["embedding"]
        tensors = load_file(str(self.root / record["file"]), device="cpu")
        return tensors[record["tensor_name"]]
