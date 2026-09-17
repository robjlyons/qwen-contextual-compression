import hashlib
import json

import torch
from safetensors.torch import save_file

from integration.freetoken_qcc.low_vram.manifest import FORMAT, FORMAT_VERSION, REPRESENTATION


def make_cache(root, layer_tensors, embedding=None):
    root.mkdir()
    embedding = torch.arange(40, dtype=torch.float16).reshape(10, 4) if embedding is None else embedding

    def write(name, tensors, **extra):
        path = root / name
        save_file(tensors, path)
        return {
            "file": name,
            "file_bytes": path.stat().st_size,
            "tensor_bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
            "tensor_count": len(tensors),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bindings": {name: name for name in tensors},
            "runtime_attributes": {},
            **extra,
        }

    embedding_record = write("embedding.safetensors", {"weight": embedding}, tensor_name="weight")
    resident_record = write("resident.safetensors", {"norm": torch.ones(4)})
    layers = []
    for index, tensors in enumerate(layer_tensors):
        layers.append(write(f"layer_{index:03d}.safetensors", tensors))
    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "representation": REPRESENTATION,
        "source_model": "fake/model",
        "source_revision": "test",
        "freetoken_version": "test",
        "adapter_fingerprint": {},
        "layer_count": len(layers),
        "tensor_count": sum(record["tensor_count"] for record in [embedding_record, resident_record, *layers]),
        "tensor_bytes": sum(record["tensor_bytes"] for record in [embedding_record, resident_record, *layers]),
        "embedding": embedding_record,
        "resident": resident_record,
        "layers": layers,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest
