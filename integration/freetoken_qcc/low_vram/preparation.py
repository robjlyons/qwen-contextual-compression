"""Feature-detected CPU cache preparation against the installed FreeToken API."""
from __future__ import annotations

import importlib
import inspect
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from integration.freetoken_qcc.low_vram.adapter import inspect_installed_contract, validate_fingerprint
from integration.freetoken_qcc.low_vram.cache_builder import write_stream_cache


class PreparationContractError(RuntimeError):
    pass


def _call_supported(function, values):
    signature = inspect.signature(function)
    kwargs = {}
    for name, parameter in signature.parameters.items():
        if name in values:
            kwargs[name] = values[name]
        elif parameter.default is inspect.Parameter.empty and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise PreparationContractError(f"Unsupported required parameter {name!r} in {function}: {signature}")
    return function(**kwargs)


def _find_attribute(candidates):
    errors = []
    for module_name, attribute in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            errors.append(str(error))
            continue
        value = getattr(module, attribute, None)
        if value is not None:
            return value
    raise PreparationContractError(f"Installed FreeToken lacks expected callable candidates: {candidates}; {errors}")


def _normalize_state(loaded) -> dict[str, torch.Tensor]:
    if isinstance(loaded, dict):
        state = loaded
    else:
        try:
            state = dict(loaded)
        except (TypeError, ValueError) as error:
            raise PreparationContractError(f"FreeToken loader returned unsupported {type(loaded)!r}") from error
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.is_meta:
            raise PreparationContractError(f"Loader emitted invalid tensor for {name}")
        if tensor.device.type != "cpu":
            raise PreparationContractError(f"CPU preflight found {name} on {tensor.device}; refusing CUDA fallback")
    return state


@contextmanager
def _temporary_default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _torch_dtype_context(dtype):
    utils = importlib.import_module("freetoken.utils")
    helper = getattr(utils, "torch_dtype", None)
    return helper(dtype) if callable(helper) else _temporary_default_dtype(dtype)


def prepare_stream_cache(model_path: str, output: Path, revision="", expected_layers=64) -> dict:
    contract = inspect_installed_contract()
    fingerprint = validate_fingerprint()
    distributed = importlib.import_module("freetoken.distributed")
    distributed_info_class = getattr(distributed, "DistributedInfo", None)
    if distributed_info_class is None:
        raise PreparationContractError("Installed freetoken.distributed lacks DistributedInfo")
    tp_info = distributed_info_class(0, 1)
    set_tp_info = _find_attribute([
        ("freetoken.distributed", "set_tp_info"),
        ("freetoken.utils", "set_tp_info"),
        ("freetoken.layers.base", "set_tp_info"),
    ])
    _call_supported(set_tp_info, {"rank": 0, "size": 1, "tp_rank": 0, "tp_size": 1, "tp_info": tp_info})
    config_class = _find_attribute([
        ("freetoken.config", "EngineConfig"),
        ("freetoken.engine.config", "EngineConfig"),
    ])
    contract["signatures"]["EngineConfig"] = str(inspect.signature(config_class))
    config = _call_supported(
        config_class,
        {
            "model": model_path,
            "model_path": model_path,
            "device": torch.device("cpu"),
            "gpu": 0,
            "tp_info": tp_info,
            "dtype": torch.bfloat16,
        },
    )
    model_config = getattr(config, "model_config", config)
    create_model = _find_attribute([
        ("freetoken.models", "create_model"),
        ("freetoken.models", "get_model"),
    ])
    values = {"config": model_config, "model_config": model_config, "device": torch.device("meta")}
    with torch.device("meta"), _torch_dtype_context(config.dtype):
        model = _call_supported(create_model, values)
    load_weight = _find_attribute([("freetoken.models", "load_weight")])
    print(
        "QCC STREAM CACHE FREETOKEN CONTRACT: "
        + json.dumps(
            {
                "freetoken_version": contract["version"],
                "engine_config_signature": str(inspect.signature(config_class)),
                "load_weight_signature": str(inspect.signature(load_weight)),
                "qwen_iter_weights_signature": contract["signatures"].get("qwen_weight.iter_weights"),
                "model_dtype": str(config.dtype),
                "tp_size": 1,
                "target_device": "cpu",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    loaded = _call_supported(
        load_weight,
        {
            "model": model_path,
            "model_path": model_path,
            "path": model_path,
            "device": torch.device("cpu"),
            "include_moe_experts": False,
            "include_vision": False,
        },
    )
    state = _normalize_state(loaded)
    # FreeToken's custom loaders pop consumed entries. Pass the original mapping
    # so raw checkpoint tensors can be released while finalized tensors are
    # installed; a shallow copy would unnecessarily retain the full checkpoint.
    leftovers = model.load_state_dict(state)
    if leftovers not in (None, {}, [], ()):
        raise PreparationContractError(f"FreeToken model load returned unexpected leftovers: {leftovers!r}")
    if state:
        raise PreparationContractError(f"FreeToken model load left {len(state)} unconsumed tensors")
    if len(model.model.layers.op_list) != expected_layers:
        raise PreparationContractError(f"Expected {expected_layers} layers, got {len(model.model.layers.op_list)}")
    manifest = write_stream_cache(model, output, model_path, revision, contract["version"], fingerprint)
    (Path(output) / "freetoken_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
