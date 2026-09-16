# Phase 5D-A: FreeToken layer-0 bridge

Phase 5D-A patches the installed FreeToken `Qwen3_5DenseMLP.forward` **in process**. It does not edit or vendor FreeToken. Only Qwen3.8 layer 0, decode shape `[1, 5120]`, is eligible; prefill, other shapes, and all other layers use the stock dense method.

FreeToken uses multiprocessing `spawn`. In shadow/replace modes the parent launcher therefore replaces only FreeToken's scheduler process target with the importable `integration.freetoken_qcc.worker.qcc_scheduler_entry`. The fresh scheduler process installs the dense-MLP patch and owns metrics and lazy CUDA state; tokenizer workers and the parent API process remain unpatched. Parent and worker PID diagnostics plus a one-time `QCC REAL DECODE INTERCEPT CONFIRMED` line prove that the model process reached the bridge. Off mode does not replace any target.

## Launch modes

Run these commands from the repository using the Python environment in which FreeToken is installed. CUDA graphs are intentionally disabled for this integration experiment.

```powershell
$env:QCC_FT_MODE = "off"
python -m integration.freetoken_qcc.launch serve RadixArk/Qwen3.8-27B-NVFP4 --cuda-graph-max-bs 0
```

For `shadow` or `replace`, also configure the frozen artifacts:

```powershell
$env:QCC_FT_MODE = "shadow" # later: replace
$env:QCC_FT_LAYER = "0"
$env:QCC_FT_SELECTOR_CHECKPOINT = "results/predictor_2000/layer_000/residual_factorized_d32_ce_to_hybrid_rank_rw010/best.pt"
$env:QCC_FT_RUNTIME_WEIGHTS = "results/runtime_weights/layer_000.safetensors"
$env:QCC_FT_VARIANT = "warp8"
$env:QCC_FT_RETENTION = "0.50"
$env:QCC_FT_METRICS = "results/runtime/freetoken_shadow.jsonl"
python -m integration.freetoken_qcc.launch serve RadixArk/Qwen3.8-27B-NVFP4 --cuda-graph-max-bs 0
```

Shadow computes both MLPs and returns the stock output, so it is a correctness mode, not a performance benchmark. Replace executes QCC only for eligible layer-0 decode calls. Initialization deliberately adds a complete FP16 FFN layer (about 510 MiB); this experiment demonstrates integration, not VRAM savings.

## Controlled request benchmark

With the server on port 1919, run the same client settings once against stock and once against replacement:

```powershell
python scripts/benchmark_freetoken_api.py --model RadixArk/Qwen3.8-27B-NVFP4 --runs 2 --warmup-runs 1 --max-tokens 64 --temperature 0 --output results/runtime/freetoken_stock.json
```

The reported rate is request-level completion throughput and includes time to first token. A one-layer result must not be presented as model-wide contextual sparsity or as an isolated decode-kernel rate.

For a Windows shadow verification, issue at least one generation request and confirm that the logs contain distinct parent and scheduler PIDs, the one-time real-decode message, and a final scheduler summary with `shadow_decode_calls` greater than zero. A completed request with zero shadow calls is an integration failure.
