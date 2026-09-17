"""Bounded-memory, disk-spooled FreeToken stream-cache construction."""
from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from integration.freetoken_qcc.low_vram.cache_builder import (
    _write_payload,
    collect_object_runtime_attributes,
    validate_finalized_nvfp4,
)
from integration.freetoken_qcc.low_vram.manifest import FORMAT, FORMAT_VERSION, REPRESENTATION
from integration.freetoken_qcc.low_vram.staging import _resolve


LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
EMBED_PREFIX = "model.embed_tokens."


class StreamBuildError(RuntimeError):
    pass


class UnsupportedResidentFinalizerError(StreamBuildError):
    pass


@dataclass(frozen=True)
class ExpectedTensor:
    shape: tuple[int, ...]
    dtype: torch.dtype
    tensor: torch.Tensor


def _memory_diagnostic(label: str, **values) -> None:
    try:
        import psutil

        memory = psutil.Process().memory_info()
        values.update(rss_gib=memory.rss / 2**30, vms_gib=memory.vms / 2**30)
    except ImportError:
        pass
    print(f"QCC STREAM MEMORY {label}: {json.dumps(values, sort_keys=True)}", flush=True)


def classify_key(name: str, layer_count: int) -> tuple[str, int | None, str]:
    if name.startswith(EMBED_PREFIX):
        return "embedding", None, name[len(EMBED_PREFIX):]
    match = LAYER_PATTERN.match(name)
    if match:
        layer_id = int(match.group(1))
        if not 0 <= layer_id < layer_count:
            raise StreamBuildError(f"Normalized key has out-of-range layer: {name}")
        return "layer", layer_id, match.group(2)
    return "resident", None, name


def expected_tensors(model) -> dict[str, ExpectedTensor]:
    state = model.state_dict()
    if not state:
        raise StreamBuildError("Meta model state traversal returned no expected tensors")
    return {name: ExpectedTensor(tuple(tensor.shape), tensor.dtype, tensor) for name, tensor in state.items()}


def classify_loader_extra(name: str, tensor: torch.Tensor, expected: dict[str, ExpectedTensor]) -> str | None:
    """Identify narrowly proven loader metadata absent from this runtime model."""
    if name in expected:
        return None
    if (
        name.endswith(".input_scale")
        and tensor.device.type == "cpu"
        and not tensor.is_meta
        and tensor.numel() == 1
        and tensor.dtype == torch.float32
    ):
        return "runtime-does-not-expose-input-scale"
    return None


def spool_normalized_weights(loaded, expected: dict[str, ExpectedTensor], building: Path, layer_count: int) -> dict:
    """Consume the loader exactly once and persist one tensor per spool file."""
    spool = building / "spool"
    spool.mkdir(parents=True, exist_ok=False)
    entries, seen, emitted_names = [], set(), set()
    ignored_entries = []
    total_bytes = largest_bytes = 0
    print("QCC STREAM NORMALIZATION", flush=True)
    _memory_diagnostic("normalization-start")
    for sequence, item in enumerate(loaded):
        if not isinstance(item, tuple) or len(item) != 2:
            raise StreamBuildError(f"FreeToken loader emitted unsupported item {item!r}")
        name, tensor = item
        if name in emitted_names:
            raise StreamBuildError(f"Duplicate normalized state key: {name}")
        if not isinstance(tensor, torch.Tensor) or tensor.is_meta:
            raise StreamBuildError(f"Loader emitted invalid tensor for {name}")
        if tensor.device.type != "cpu":
            raise StreamBuildError(f"CPU preparation found {name} on {tensor.device}; refusing CUDA fallback")
        emitted_names.add(name)
        if name not in expected:
            reason = classify_loader_extra(name, tensor, expected)
            if reason is None:
                raise StreamBuildError(f"Unexpected normalized state key: {name}")
            byte_size = tensor.numel() * tensor.element_size()
            ignored_entries.append({
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "tensor_bytes": byte_size,
                "reason": reason,
            })
            if len(ignored_entries) == 1:
                print(
                    "QCC STREAM LOADER-ONLY METADATA:\n"
                    f"ignoring {name}\nreason={reason}",
                    flush=True,
                )
            del tensor, item
            continue
        group, layer_id, relative = classify_key(name, layer_count)
        filename = f"{sequence:06d}.safetensors"
        contiguous = tensor.detach().contiguous()
        save_file({"tensor": contiguous}, spool / filename)
        byte_size = tensor.numel() * tensor.element_size()
        entries.append({
            "name": name,
            "file": filename,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "tensor_bytes": byte_size,
            "group": group,
            "layer_id": layer_id,
            "relative": relative,
        })
        seen.add(name)
        total_bytes += byte_size
        largest_bytes = max(largest_bytes, byte_size)
        del contiguous, tensor, item
        if (sequence + 1) % 32 == 0:
            gc.collect()
            print(f"normalized tensors: {sequence + 1}; normalized GiB: {total_bytes / 2**30:.3f}", flush=True)
    missing = set(expected) - seen
    if missing:
        raise StreamBuildError(f"Missing {len(missing)} expected normalized keys: {sorted(missing)[:8]}")
    reason_counts = {}
    for entry in ignored_entries:
        reason = entry["reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    ignored = {
        "count": len(ignored_entries),
        "tensor_bytes": sum(entry["tensor_bytes"] for entry in ignored_entries),
        "reason_counts": reason_counts,
        "entries": ignored_entries,
    }
    index = {
        "entries": entries,
        "tensor_count": len(entries),
        "tensor_bytes": total_bytes,
        "largest_tensor_bytes": largest_bytes,
        "ignored_loader_tensors": ignored,
    }
    (building / "spool_index.json").write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    category_summary = {}
    for entry in entries:
        category = f"layer_{entry['layer_id']:03d}" if entry["group"] == "layer" else entry["group"]
        summary = category_summary.setdefault(category, {"tensor_count": 0, "tensor_bytes": 0})
        summary["tensor_count"] += 1
        summary["tensor_bytes"] += entry["tensor_bytes"]
    print(f"normalized tensors: {len(entries)}; normalized GiB: {total_bytes / 2**30:.3f}; spool GiB: {total_bytes / 2**30:.3f}", flush=True)
    print(
        f"ignored loader tensors: {ignored['count']}; ignored loader bytes: {ignored['tensor_bytes']}; "
        f"reason counts: {json.dumps(reason_counts, sort_keys=True)}",
        flush=True,
    )
    print("QCC STREAM NORMALIZATION GROUPS: " + json.dumps(category_summary, sort_keys=True), flush=True)
    _memory_diagnostic("normalization-complete", spool_bytes=total_bytes, largest_normalized_tensor=largest_bytes)
    return index


def _load_entries(building: Path, entries: list[dict], expected: dict[str, ExpectedTensor]) -> dict[str, torch.Tensor]:
    tensors = {}
    for entry in entries:
        tensor = load_file(building / "spool" / entry["file"], device="cpu")["tensor"]
        specification = expected[entry["name"]]
        if tuple(tensor.shape) != specification.shape or tensor.dtype != specification.dtype:
            raise StreamBuildError(
                f"Normalized tensor contract mismatch for {entry['name']}: "
                f"{tuple(tensor.shape)}/{tensor.dtype} != {specification.shape}/{specification.dtype}"
            )
        tensors[entry["relative"]] = tensor
    return tensors


def _delete_entries(building: Path, entries: list[dict]) -> None:
    for entry in entries:
        (building / "spool" / entry["file"]).unlink()


def _snapshot_layer(layer, relative_names: list[str]):
    tensor_bindings = []
    for path in relative_names:
        owner, attribute = _resolve(layer, path)
        tensor_bindings.append((owner, attribute, getattr(owner, attribute)))
    runtime = []
    for path in collect_object_runtime_attributes(layer):
        owner, attribute = _resolve(layer, path)
        runtime.append((owner, attribute, getattr(owner, attribute)))
    return tensor_bindings, runtime


def _restore_layer(snapshot) -> None:
    for owner, attribute, value in (*snapshot[0], *snapshot[1]):
        setattr(owner, attribute, value)


def _relative_finalized_state(layer, layer_id: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_id}."
    result = {}
    for name, tensor in layer.state_dict().items():
        relative = name[len(prefix):] if name.startswith(prefix) else name
        if tensor.is_meta or tensor.device.type != "cpu":
            raise StreamBuildError(f"Finalized layer tensor is not CPU: {name} on {tensor.device}")
        result[relative] = tensor
    if not result:
        raise StreamBuildError(f"Finalized layer {layer_id} returned no tensors")
    return result


def _build_manifest(records, source_model, source_revision, freetoken_version, adapter_fingerprint, nvfp4_count,
                    ignored_loader_tensors):
    embedding_record, resident_record, layer_records = records
    all_records = [embedding_record, resident_record, *layer_records]
    dtype_counts = {}
    for record in all_records:
        for metadata in record["tensors"].values():
            dtype_counts[metadata["dtype"]] = dtype_counts.get(metadata["dtype"], 0) + 1
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "representation": REPRESENTATION,
        "source_model": source_model,
        "source_revision": source_revision,
        "freetoken_version": freetoken_version,
        "adapter_fingerprint": adapter_fingerprint,
        "layer_count": len(layer_records),
        "nvfp4_transposed_modules": nvfp4_count,
        "dtype_counts": dtype_counts,
        "tensor_count": sum(record["tensor_count"] for record in all_records),
        "tensor_bytes": sum(record["tensor_bytes"] for record in all_records),
        "file_bytes": sum(record["file_bytes"] for record in all_records),
        "ignored_loader_tensors": ignored_loader_tensors,
        "embedding": embedding_record,
        "resident": resident_record,
        "layers": layer_records,
    }


def build_stream_cache_streaming(model, loaded, output: Path, source_model: str, source_revision: str,
                                 freetoken_version: str, adapter_fingerprint: dict, contract: dict) -> dict:
    """Build the final cache while retaining at most one logical group of tensors."""
    output = Path(output)
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(f"Stream-cache output or build directory already exists: {output}, {building}")
    building.mkdir(parents=True)
    try:
        layers = model.model.layers.op_list
        expected = expected_tensors(model)
        index = spool_normalized_weights(loaded, expected, building, len(layers))
        entries = index["entries"]

        embedding_entries = [entry for entry in entries if entry["group"] == "embedding"]
        embedding = _load_entries(building, embedding_entries, expected)
        embedding_record = _write_payload(building, "embedding.safetensors", embedding,
                                          {name: f"model.embed_tokens.{name}" for name in embedding})
        embedding_record["tensor_name"] = "weight" if "weight" in embedding else next(iter(embedding))
        _delete_entries(building, embedding_entries)
        del embedding
        gc.collect()
        _memory_diagnostic("after-embedding")

        resident_entries = [entry for entry in entries if entry["group"] == "resident"]
        unsupported = [entry["name"] for entry in resident_entries
                       if not (entry["name"].startswith("model.norm.") or entry["name"].startswith("lm_head."))]
        if unsupported:
            raise UnsupportedResidentFinalizerError(
                f"Resident tensors may require an unsupported module finalizer: {unsupported[:8]}"
            )
        resident_relative = _load_entries(building, resident_entries, expected)
        resident = {entry["name"]: resident_relative[entry["relative"]] for entry in resident_entries}
        resident_record = _write_payload(building, "resident.safetensors", resident, {name: name for name in resident})
        _delete_entries(building, resident_entries)
        del resident_relative, resident
        gc.collect()
        _memory_diagnostic("after-resident")

        print("QCC FINALIZING LAYERS", flush=True)
        layer_records, nvfp4_count, largest_layer = [], 0, 0
        for layer_id, layer in enumerate(layers):
            layer_entries = [entry for entry in entries if entry["group"] == "layer" and entry["layer_id"] == layer_id]
            if not layer_entries:
                raise StreamBuildError(f"Layer {layer_id} has no normalized tensors")
            relative_names = [entry["relative"] for entry in layer_entries]
            snapshot = _snapshot_layer(layer, relative_names)
            layer_state = _load_entries(building, layer_entries, expected)
            try:
                leftovers = layer.load_state_dict(layer_state)
                if leftovers not in (None, {}, [], ()):
                    raise StreamBuildError(f"Layer {layer_id} load returned leftovers: {leftovers!r}")
                if layer_state:
                    raise StreamBuildError(f"Layer {layer_id} left {len(layer_state)} unconsumed tensors")
                nvfp4_count += validate_finalized_nvfp4(layer)
                finalized = _relative_finalized_state(layer, layer_id)
                attributes = collect_object_runtime_attributes(layer)
                record = _write_payload(building, f"layer_{layer_id:03d}.safetensors", finalized,
                                        {name: name for name in finalized}, attributes)
                largest_layer = max(largest_layer, record["tensor_bytes"])
                layer_records.append(record)
            finally:
                _restore_layer(snapshot)
                del layer_state
            _delete_entries(building, layer_entries)
            del finalized
            gc.collect()
            if layer_id % 8 == 0 or layer_id == len(layers) - 1:
                print(f"layer {layer_id:02d}/{len(layers) - 1:02d}", flush=True)
                _memory_diagnostic(f"after-layer-{layer_id:02d}", largest_finalized_layer=largest_layer)

        manifest = _build_manifest((embedding_record, resident_record, layer_records), source_model,
                                   source_revision, freetoken_version, adapter_fingerprint, nvfp4_count,
                                   index["ignored_loader_tensors"])
        (building / "freetoken_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
        manifest_temporary = building / "manifest.json.tmp"
        manifest_temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifest_temporary.replace(building / "manifest.json")
        (building / "spool_index.json").unlink()
        (building / "spool").rmdir()
        building.replace(output)
        _memory_diagnostic("cache-complete", spool_bytes=index["tensor_bytes"], final_bytes=manifest["file_bytes"],
                           largest_normalized_tensor=index["largest_tensor_bytes"], largest_finalized_layer=largest_layer)
        return manifest
    except BaseException:
        # Deliberately retain the spool for diagnosis; its lack of manifest.json
        # prevents it from being mistaken for a valid cache.
        manifest = building / "manifest.json"
        if manifest.exists():
            manifest.unlink()
        print(f"QCC stream-cache build failed; temporary files retained at {building}", flush=True)
        raise
