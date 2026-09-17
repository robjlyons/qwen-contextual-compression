"""Partition a FreeToken-finalized CPU model into the stream-cache format."""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file

from integration.freetoken_qcc.low_vram.manifest import FORMAT, FORMAT_VERSION, REPRESENTATION, file_sha256


LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
EMBED_PREFIX = "model.embed_tokens."


def collect_runtime_attributes(model) -> dict[int, dict[str, object]]:
    """Record finalized non-tensor state, notably NVFP4 `_transposed`."""
    result: dict[int, dict[str, object]] = {}
    layers = model.model.layers.op_list
    for layer_id, layer in enumerate(layers):
        result[layer_id] = collect_object_runtime_attributes(layer)
    return result


def collect_object_runtime_attributes(root) -> dict[str, object]:
    """Collect runtime flags relative to an arbitrary layer-like object."""
    attributes = {}
    stack = [("", root)]
    visited = set()
    while stack:
        prefix, value = stack.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        if hasattr(value, "_transposed"):
            attributes[prefix + "_transposed"] = bool(value._transposed)
        for name, child in vars(value).items():
            if name.startswith("_") or isinstance(child, torch.Tensor):
                continue
            if hasattr(child, "__dict__"):
                stack.append((prefix + name + ".", child))
    return attributes


def validate_finalized_nvfp4(model) -> int:
    count = 0
    stack, visited = [model], set()
    while stack:
        value = stack.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        if value.__class__.__name__ in {"Nvfp4DenseLinear", "Nvfp4DenseColMerged"}:
            count += 1
            if not getattr(value, "_transposed", False):
                raise RuntimeError("NVFP4 module is not finalized/transposed")
            if value.weight.dtype != torch.int32 or value.weight_scale.dtype != torch.float8_e4m3fn or value.weight_global.dtype != torch.float16:
                raise RuntimeError("NVFP4 finalized tensor dtypes do not match the installed runtime contract")
            if value.weight.ndim != 2 or value.weight_scale.ndim != 2:
                raise RuntimeError("NVFP4 finalized tensors must use K-major matrices")
        for child in vars(value).values() if hasattr(value, "__dict__") else ():
            if not isinstance(child, torch.Tensor) and hasattr(child, "__dict__"):
                stack.append(child)
    return count


def finalized_state(model) -> dict[str, torch.Tensor]:
    """Use FreeToken's state traversal and reject meta/CUDA cache inputs."""
    state = model.state_dict()
    if not state:
        raise RuntimeError("FreeToken model state traversal returned no tensors")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"FreeToken state value is not a tensor: {name}")
        if tensor.is_meta:
            raise RuntimeError(f"Finalized FreeToken state remains meta: {name}")
        if tensor.device.type != "cpu":
            raise RuntimeError(f"Finalized FreeToken cache input is not CPU: {name} on {tensor.device}")
    return state


def partition_finalized_state(state: dict[str, torch.Tensor], layer_count: int):
    embedding, resident = {}, {}
    layers = [dict() for _ in range(layer_count)]
    seen = set()
    for name, tensor in state.items():
        if name.startswith(EMBED_PREFIX):
            relative = name[len(EMBED_PREFIX):]
            embedding[relative] = tensor
        else:
            match = LAYER_PATTERN.match(name)
            if match:
                layer_id, relative = int(match.group(1)), match.group(2)
                if not 0 <= layer_id < layer_count:
                    raise ValueError(f"State contains out-of-range layer: {name}")
                layers[layer_id][relative] = tensor
            else:
                resident[name] = tensor
        seen.add(name)
    if len(seen) != len(state) or sum(len(group) for group in [embedding, resident, *layers]) != len(state):
        raise RuntimeError("Stream-cache partition did not classify every tensor exactly once")
    if not embedding or not resident or any(not group for group in layers):
        raise RuntimeError("Stream-cache partition has an empty embedding, resident, or layer payload")
    return embedding, resident, layers


def _write_payload(root: Path, filename: str, tensors: dict, bindings: dict, runtime_attributes=None) -> dict:
    path = root / filename
    contiguous = {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
    save_file(contiguous, path)
    return {
        "file": filename,
        "file_bytes": path.stat().st_size,
        "tensor_bytes": sum(t.numel() * t.element_size() for t in contiguous.values()),
        "tensor_count": len(contiguous),
        "sha256": file_sha256(path),
        "bindings": bindings,
        "runtime_attributes": runtime_attributes or {},
        "tensors": {name: {"shape": list(t.shape), "dtype": str(t.dtype)} for name, t in contiguous.items()},
    }


def write_stream_cache(model, output: Path, source_model: str, source_revision: str, freetoken_version: str, adapter_fingerprint: dict) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    layer_count = len(model.model.layers.op_list)
    state = finalized_state(model)
    embedding, resident, layers = partition_finalized_state(state, layer_count)
    runtime_attributes = collect_runtime_attributes(model)
    nvfp4_count = validate_finalized_nvfp4(model)
    embedding_record = _write_payload(output, "embedding.safetensors", embedding, {name: f"model.embed_tokens.{name}" for name in embedding})
    embedding_record["tensor_name"] = "weight" if "weight" in embedding else next(iter(embedding))
    resident_record = _write_payload(output, "resident.safetensors", resident, {name: name for name in resident})
    layer_records = []
    for index, tensors in enumerate(layers):
        layer_records.append(_write_payload(output, f"layer_{index:03d}.safetensors", tensors, {name: name for name in tensors}, runtime_attributes[index]))
    records = [embedding_record, resident_record, *layer_records]
    dtype_counts = {}
    for record in records:
        for metadata in record["tensors"].values():
            dtype_counts[metadata["dtype"]] = dtype_counts.get(metadata["dtype"], 0) + 1
    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "representation": REPRESENTATION,
        "source_model": source_model,
        "source_revision": source_revision,
        "freetoken_version": freetoken_version,
        "adapter_fingerprint": adapter_fingerprint,
        "layer_count": layer_count,
        "nvfp4_transposed_modules": nvfp4_count,
        "dtype_counts": dtype_counts,
        "tensor_count": sum(record["tensor_count"] for record in records),
        "tensor_bytes": sum(record["tensor_bytes"] for record in records),
        "file_bytes": sum(record["file_bytes"] for record in records),
        "embedding": embedding_record,
        "resident": resident_record,
        "layers": layer_records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
