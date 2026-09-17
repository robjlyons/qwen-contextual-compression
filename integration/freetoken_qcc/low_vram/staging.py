"""Two-slot deterministic layer staging with explicit activation/deactivation."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


def _resolve(root, path: str):
    owner = root
    pieces = path.split(".")
    for piece in pieces[:-1]:
        owner = getattr(owner, piece)
    return owner, pieces[-1]


@dataclass
class StagingSlot:
    slot_id: int
    layer_id: int | None = None
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    ready: bool = False


class LayerStager:
    def __init__(self, host_store, device, slots=2, asynchronous=False):
        if slots != 2:
            raise ValueError("Phase 5E-A LayerStager requires exactly two slots")
        if asynchronous:
            raise ValueError("Asynchronous staging is not enabled in Phase 5E-A")
        self.host_store = host_store
        self.device = torch.device(device)
        self.slots = [StagingSlot(index) for index in range(slots)]
        self.events = []
        self.total_bytes = 0
        self.total_seconds = 0.0
        self.peak_cuda_allocated = 0
        self.peak_cuda_reserved = 0
        self._active = {}
        self._first_stage_logged = False

    def stage(self, layer_id: int, slot: int) -> StagingSlot:
        target = self.slots[slot]
        if target.slot_id in self._active:
            raise RuntimeError(f"cannot reuse active staging slot {slot}")
        layer = self.host_store.layer(layer_id)
        started = time.perf_counter()
        tensors = {name: tensor.to(self.device) for name, tensor in layer.tensors.items()}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_cuda_allocated = max(self.peak_cuda_allocated, torch.cuda.memory_allocated(self.device))
            self.peak_cuda_reserved = max(self.peak_cuda_reserved, torch.cuda.memory_reserved(self.device))
        elapsed = time.perf_counter() - started
        byte_size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        target.layer_id, target.tensors, target.ready = layer_id, tensors, True
        event = {"layer_id": layer_id, "slot_id": slot, "bytes": byte_size, "h2d_seconds": elapsed}
        self.events.append(event)
        self.total_bytes += byte_size
        self.total_seconds += elapsed
        if not self._first_stage_logged:
            self._first_stage_logged = True
            print(
                f"QCC LOW-VRAM REAL LAYER STAGE CONFIRMED: layer={layer_id} slot={slot} bytes={byte_size} h2d_ms={elapsed * 1000:.3f}",
                flush=True,
            )
        return target

    def wait(self, slot: int) -> None:
        if not self.slots[slot].ready:
            raise RuntimeError(f"staging slot {slot} is not ready")

    def activate(self, layer_id: int, slot: int, layer_object) -> None:
        target = self.slots[slot]
        if not target.ready or target.layer_id != layer_id:
            raise RuntimeError(f"slot {slot} does not contain layer {layer_id}")
        restore = []
        for tensor_name, attribute_path in self.host_store.layer(layer_id).bindings.items():
            owner, attribute = _resolve(layer_object, attribute_path)
            restore.append((owner, attribute, getattr(owner, attribute)))
            setattr(owner, attribute, target.tensors[tensor_name])
        for attribute_path, value in self.host_store.layer(layer_id).runtime_attributes.items():
            owner, attribute = _resolve(layer_object, attribute_path)
            restore.append((owner, attribute, getattr(owner, attribute)))
            setattr(owner, attribute, value)
        self._active[slot] = restore

    def deactivate(self, slot: int) -> None:
        restore = self._active.pop(slot, None)
        if restore is None:
            raise RuntimeError(f"staging slot {slot} is not active")
        for owner, attribute, previous in restore:
            setattr(owner, attribute, previous)

    def summary(self) -> dict:
        return {
            "layer_stages": len(self.events),
            "bytes_h2d": self.total_bytes,
            "h2d_seconds": self.total_seconds,
            "slot_bytes": [sum(t.numel() * t.element_size() for t in slot.tensors.values()) for slot in self.slots],
            "peak_cuda_allocated": self.peak_cuda_allocated,
            "peak_cuda_reserved": self.peak_cuda_reserved,
        }
