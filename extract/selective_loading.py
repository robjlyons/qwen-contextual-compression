"""Load just the checkpoint shards needed for an early-layer experiment.

The model object is constructed on PyTorch's ``meta`` device from the official
Transformers config.  Only the embedding and selected layer parameters are then
materialized.  This preserves the official module implementation without ever
asking ``from_pretrained`` to resolve every checkpoint shard.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from extract.checkpoint_plan import (
    INDEX_FILENAME,
    SelectivePlan,
    plan_selective_load,
    resolve_checkpoint_tensor_names,
)


def _target_device(device: str | None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_selective_model(
    model_name: str,
    layer_index: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    device: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | None = None,
    trust_remote_code: bool = False,
):
    """Download and materialize only shards containing embeddings and one layer.

    ``hf_hub_download`` first fetches the tiny index, then exactly the shard set
    derived from its ``weight_map``.  ``from_pretrained`` is intentionally absent
    from this code path because it would resolve the complete checkpoint.
    """
    common = {"repo_id": model_name, "revision": revision, "cache_dir": cache_dir, "token": token}
    index_path = hf_hub_download(filename=INDEX_FILENAME, **common)
    index = json.loads(Path(index_path).read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"{INDEX_FILENAME} has no valid 'weight_map' object")
    plan = plan_selective_load(weight_map, layer_index)

    config = AutoConfig.from_pretrained(
        model_name, revision=revision, cache_dir=cache_dir, token=token,
        trust_remote_code=trust_remote_code,
    )
    # Keep constructor-created non-persistent rotary/cache buffers real; only the
    # enormous parameter set belongs on meta.
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)

    wanted = set(plan.tensor_names)
    model_state = model.state_dict()
    checkpoint_to_model = resolve_checkpoint_tensor_names(wanted, model_state.keys())
    loaded: set[str] = set()
    target = _target_device(device)
    for shard_name in plan.shard_names:
        shard_path = hf_hub_download(filename=shard_name, **common)
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if name not in wanted:
                    continue
                tensor = shard.get_tensor(name)
                model_name = checkpoint_to_model[name]
                expected_shape = tuple(model_state[model_name].shape)
                if tuple(tensor.shape) != expected_shape:
                    raise RuntimeError(
                        f"Shape mismatch for {name!r} -> {model_name!r}: checkpoint "
                        f"{tuple(tensor.shape)}, model {expected_shape}"
                    )
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=dtype)
                set_module_tensor_to_device(model, model_name, target, value=tensor)
                loaded.add(name)
    missing = wanted - loaded
    if missing:
        raise RuntimeError(f"Selected shards did not contain {len(missing)} indexed tensors: {sorted(missing)[:8]}")

    for name, buffer in model.named_buffers():
        if buffer.device.type != "meta" and buffer.device != target:
            set_module_tensor_to_device(model, name, target, value=buffer.to(target))

    model.eval()
    model._selective_load_plan = plan  # runtime provenance used by inspection/capture
    model._checkpoint_to_model_names = checkpoint_to_model
    model._is_selectively_loaded = True
    # Validate the materialized MLP against an independent equation before any
    # experiment consumes it.  This calls the official module implementation.
    from extract.extract_ffn import explicit_dense_ffn, locate_ffn
    ffn = locate_ffn(model, layer_index)
    probe = torch.linspace(
        -0.5, 0.5, ffn.gate_proj.weight.shape[1], device=target, dtype=dtype
    ).reshape(1, 1, -1)
    with torch.inference_mode():
        official = ffn.module(probe)
        explicit = explicit_dense_ffn(probe, ffn)
    tolerance = 1e-5 if dtype == torch.float32 else 5e-3
    torch.testing.assert_close(official, explicit, rtol=tolerance, atol=tolerance)
    model._selective_mlp_verified = True
    return model
