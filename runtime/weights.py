"""Selective extraction and validation of one FFN's safetensors weights."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from extract.checkpoint_plan import INDEX_FILENAME


ROLES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")


def resolve_ffn_tensor_names(weight_map: dict[str, str], layer: int) -> dict[str, str]:
    """Resolve the main language-model MLP, explicitly excluding Qwen MTP."""
    result = {}
    marker = f".layers.{layer}.mlp."
    for role in ROLES:
        matches = [name for name in weight_map if marker in name and name.endswith(f".{role}") and not name.startswith("mtp.") and ".mtp." not in name]
        preferred = [name for name in matches if name.startswith("model.language_model.layers.")]
        if preferred:
            matches = preferred
        if len(matches) != 1:
            raise RuntimeError(f"Expected one layer-{layer} {role}; found {matches or '<none>'}")
        result[role] = matches[0]
    return result


def validate_weight_shapes(weights: dict[str, torch.Tensor]) -> tuple[int, int]:
    gate, up, down = (weights[name] for name in ROLES)
    if gate.ndim != 2 or up.ndim != 2 or down.ndim != 2:
        raise ValueError("FFN weights must all be matrices")
    if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError(f"Incompatible FFN shapes: gate={tuple(gate.shape)}, up={tuple(up.shape)}, down={tuple(down.shape)}")
    return gate.shape[1], gate.shape[0]


def _index_path(model: str | None, model_dir: Path | None, revision, cache_dir, token) -> tuple[Path, dict]:
    if (model is None) == (model_dir is None):
        raise ValueError("Specify exactly one of model or model_dir")
    if model_dir is not None:
        path = model_dir / INDEX_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"Missing safetensors index: {path}")
        return path, {"local_dir": model_dir}
    common = {"repo_id": model, "revision": revision, "cache_dir": cache_dir, "token": token}
    return Path(hf_hub_download(filename=INDEX_FILENAME, **common)), common


def extract_ffn_layer_weights(output: Path, layer=0, model=None, model_dir=None, revision=None, cache_dir=None, token=None, dtype="fp16") -> dict:
    """Read/download only shards containing the requested FFN tensors."""
    model_dir = None if model_dir is None else Path(model_dir)
    index_path, source = _index_path(model, model_dir, revision, cache_dir, token)
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"{INDEX_FILENAME} has no valid weight_map")
    names = resolve_ffn_tensor_names(weight_map, layer)
    shard_names = sorted({weight_map[name] for name in names.values()})
    wanted = set(names.values()); loaded = {}
    for shard_name in shard_names:
        shard_path = model_dir / shard_name if model_dir is not None else Path(hf_hub_download(filename=shard_name, **source))
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for role, full_name in names.items():
                if full_name in wanted and full_name in shard.keys():
                    loaded[role] = shard.get_tensor(full_name)
    missing = set(ROLES) - loaded.keys()
    if missing:
        raise RuntimeError(f"Selected shards did not contain {sorted(missing)}")
    target_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    loaded = {name: tensor.to(target_dtype).contiguous() for name, tensor in loaded.items()}
    hidden, intermediate = validate_weight_shapes(loaded)
    metadata = {"model": model or str(model_dir), "revision": revision or "", "layer": str(layer), "hidden_size": str(hidden), "intermediate_size": str(intermediate), "dtype": dtype, "source_tensor_names": json.dumps(names), "tensor_shapes": json.dumps({k: list(v.shape) for k, v in loaded.items()}), "source_shards": json.dumps(shard_names)}
    output = Path(output);output.parent.mkdir(parents=True, exist_ok=True);save_file(loaded, output, metadata=metadata)
    return {"output": str(output), "metadata": metadata}


def load_runtime_weights(path: Path, device="cpu") -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    weights = load_file(str(path), device=str(device));validate_weight_shapes(weights)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    return weights, metadata


def save_neuron_major_layout(source: Path, output: Path) -> dict:
    """Create an optional runtime artifact with contiguous per-neuron down rows."""
    weights, metadata = load_runtime_weights(source, "cpu")
    packed = {"gate_proj.weight": weights["gate_proj.weight"], "up_proj.weight": weights["up_proj.weight"], "down_t.weight": weights["down_proj.weight"].T.contiguous()}
    output = Path(output);output.parent.mkdir(parents=True, exist_ok=True)
    new_metadata = dict(metadata);new_metadata.update(runtime_layout="neuron_major_separate", down_layout="transposed_contiguous", source_artifact=str(source))
    save_file(packed, output, metadata=new_metadata)
    return {"output": str(output), "metadata": new_metadata}
