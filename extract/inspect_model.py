"""Loading and reporting helpers shared by command-line tools."""
from __future__ import annotations
from pathlib import Path
import torch
from extract.extract_ffn import locate_ffn, projection_parameter_count
from extract.selective_loading import load_selective_model

DTYPES = {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16,
          "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
          "auto": "auto"}


def load_model(name: str, dtype: str = "bfloat16", device_map: str | None = "auto",
               offload_folder: str | None = None, revision: str | None = None,
               *, layer: int = 0, device: str | None = None, full_model: bool = False,
               cache_dir: str | None = None, token: str | None = None,
               trust_remote_code: bool = False):
    if not full_model:
        selected_dtype = DTYPES[dtype]
        if selected_dtype == "auto":
            raise ValueError("--dtype auto is only supported with --full-model; choose float32, float16, or bfloat16")
        return load_selective_model(name, layer, selected_dtype, device, revision, cache_dir, token,
                                    trust_remote_code)
    from transformers import AutoModelForCausalLM
    kwargs = {"torch_dtype": DTYPES[dtype], "revision": revision, "low_cpu_mem_usage": True,
              "cache_dir": cache_dir, "token": token, "trust_remote_code": trust_remote_code}
    if device_map and device_map != "none": kwargs["device_map"] = device_map
    if offload_folder:
        Path(offload_folder).mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = offload_folder
    return AutoModelForCausalLM.from_pretrained(name, **kwargs)


def inspect(model, layer_index: int) -> dict:
    f = locate_ffn(model, layer_index)
    backbone_name = f.layers_path.rsplit(".layers", 1)[0]
    backbone = model.get_submodule(backbone_name) if backbone_name else model
    def shape(m): return list(m.weight.shape)
    return {"model_class": type(model).__name__, "backbone_class": type(backbone).__name__,
            "layers_path": f.layers_path, "num_layers": len(model.get_submodule(f.layers_path)),
            "layer": layer_index, "layer_class": type(f.layer).__name__, "ffn_path": f.path,
            "ffn_class": type(f.module).__name__, "gate_proj": shape(f.gate_proj),
            "up_proj": shape(f.up_proj), "down_proj": shape(f.down_proj),
            "dtype": str(f.gate_proj.weight.dtype), "device": str(f.gate_proj.weight.device),
            "ffn_parameter_count": projection_parameter_count(f),
            "model_parameter_count": sum(p.numel() for p in model.parameters()),
            "selective_load": bool(getattr(model, "_is_selectively_loaded", False)),
            "official_mlp_verified": bool(getattr(model, "_selective_mlp_verified", False)),
            "downloaded_shards": list(getattr(getattr(model, "_selective_load_plan", None), "shard_names", ())) }
