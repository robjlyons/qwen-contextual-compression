# Phase 5E-A low-VRAM foundations

Phase 5E-A keeps sparse QCC replacement disabled and targets deterministic whole-layer streaming. The stream cache schema stores **FreeToken-finalized runtime tensors**, not raw Hugging Face checkpoint tensors. This distinction is mandatory: NVFP4/FP8 fusion and finalization must happen once while preparing the cache, never once per layer per token.

The cache manifest is versioned and binds each tensor name to the attribute path used by its FreeToken layer object. It records the source model/revision, FreeToken version, tensor/file byte sizes, and SHA-256 checksums. Payloads are split into `embedding.safetensors`, `resident.safetensors`, and one `layer_NNN.safetensors` file per transformer layer. Layer payloads remain pageable CPU tensors. The host store exposes both whole tensors and row selection so Phase 5E-B can add neuron-level MLP transfer without changing the cache API.

`LayerStager` provides two deterministic synchronous slots. A layer is copied to a slot, bound to the existing FreeToken layer object for exactly one forward call, and then the original host/meta attributes are restored. There is no LRU policy. `HostBackedEmbedding` gathers only requested rows on CPU and transfers the gathered result. LM-head and norm tensors belong in the resident payload.

## Required installed-package audit

The build environment used for this repository does not contain the Windows FreeToken installation or the `RadixArk/Qwen3.8-27B-NVFP4` checkpoint. The exact `0.1.2+g141c31a8d` checkpoint iterator, `_load_maybe_quantized`, finalization, and model-loop hooks therefore cannot be truthfully selected or tested here. Run the source audit on the target machine:

```powershell
python scripts/inspect_freetoken_install.py `
  --package-root C:\Users\Rob\AppData\Local\FreeToken\venv\Lib\site-packages\freetoken `
  --output results\runtime\freetoken_0.1.2_source_audit.json
```

The version-specific adapter now fingerprints the behavior-critical installed sources, intercepts `Engine._load_weight_state_dict`, and installs the finalized stream cache through `Qwen3_5MoEForCausalLM.load_state_dict`. It rejects source mismatches, TP other than one, cache/model mismatches, and any non-empty state mapping reaching the model hook. This bypass is intentionally narrow to FreeToken's Qwen3.5/Qwen3.8 class.

Prepare the cache in the FreeToken environment:

```powershell
& $FTPython scripts/prepare_freetoken_stream_cache.py --model RadixArk/Qwen3.8-27B-NVFP4 --output results\freetoken_stream_cache\qwen38_27b_nvfp4
```

Then launch with `QCC_FT_MODE=off`, `QCC_FT_LOW_VRAM=1`, `QCC_FT_STREAM_ASYNC=0`, and `QCC_FT_STREAM_CACHE` pointing to that directory. CUDA graphs must remain disabled. The Linux development environment still cannot claim full-model boot or generation success because it has neither the installed FreeToken build, checkpoint, nor target GPU.
