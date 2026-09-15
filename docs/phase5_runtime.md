# Phase 5A: single-layer sparse FFN runtime

Phase 5A benchmarks only the Qwen3.8-27B layer-0 FFN. It is not a full-model
generation benchmark and reports FFN evaluations/second, never model tokens/s.
GPU-resident tests retain all dense weights, so they demonstrate compute/read
sparsity rather than full-model VRAM savings.

First selectively extract only the indexed layer-0 FFN shards:

```bash
python scripts/extract_ffn_layer_weights.py --model Qwen/Qwen3.8-27B \
  --layer 0 --dtype fp16 --output results/runtime_weights/layer_000.safetensors
```

Then run the cross-platform Python benchmark:

```bash
python scripts/benchmark_sparse_ffn.py --results-dir results/predictor_2000 \
  --runtime-weights results/runtime_weights/layer_000.safetensors --layer 0 \
  --stage1-run residual_factorized_d32_ce_to_hybrid_rank_rw010 \
  --retention .50 --device cuda --dtype fp16 --warmup 20 --iterations 200
```

The harness stops before timing if recomputed gated activations or dense outputs
do not match the stored Phase-4 targets. It compares dense GPU, dynamic PyTorch
gather, static-prepacked diagnostic, and materialization-free Triton indexed
paths. Triton is skipped when unavailable. `--benchmark-host-transfer` adds an
analysis-only pinned-memory payload experiment.

The separately measured FreeToken result (Qwen3.6-27B-NVFP4, RTX 3070 Ti 8 GB,
approximately 0.5 model tokens/s) is not directly comparable to this one-layer
Qwen3.8 FP16 experiment. Full sparse generation additionally requires predictors
for all layers and a sparse residency/paging design.

## Phase 5B tracks

Phase 5B keeps the layer-0 predictor and 50% mask fixed and separates three
measurements:

1. `benchmark_selector.py` compares baseline FP32, folded-normalization FP32,
   FP16, and folded-normalization FP16 inference copies. Folding uses PyTorch's
   `[out, in]` linear layout: `W_eff = W / std[None, :]` and
   `b_eff = -(W_eff @ mean)`. No variant is promoted automatically; score,
   selected-set, and sparse-output quality accompany latency.
2. `benchmark_indexed_ffn.py` reports selector-only, FFN-only, and combined
   latency for dynamic gather, static-packed, Triton, and optional custom CUDA
   paths. Both original `[H,I]` and neuron-major `[I,H]` down layouts are tested.
   ID sorting, when requested, is included in combined latency.
3. `analyze_temporal_cache.py` requires explicit prompt IDs and token positions.
   It refuses to infer adjacency from split order, and reports an unsupported
   status when metadata is insufficient. Its LRU simulations reset per prompt
   and use the Phase 5A 13 GB/s PCIe measurement only as a refill-time estimate.

The custom CUDA extension is correctness-first and builds lazily through
`torch.utils.cpp_extension`. On Windows it reports a clear skip unless CUDA and
an MSVC Developer Prompt are available. Indexed kernels retain dense weights
but allocate only selected activations, IDs, output, and small workspaces; they
do not prove full-model VRAM reduction.

```bash
python scripts/benchmark_selector.py --results-dir results/predictor_2000 \
  --runtime-weights results/runtime_weights/layer_000.safetensors \
  --stage1-run residual_factorized_d32_ce_to_hybrid_rank_rw010
python scripts/benchmark_indexed_ffn.py --results-dir results/predictor_2000 \
  --runtime-weights results/runtime_weights/layer_000.safetensors \
  --stage1-run residual_factorized_d32_ce_to_hybrid_rank_rw010
python scripts/analyze_temporal_cache.py --results-dir results/predictor_2000 \
  --stage1-run residual_factorized_d32_ce_to_hybrid_rank_rw010
```

The FreeToken Qwen3.6-NVFP4 result remains a separate full-model hardware
reference and must not be compared to these Qwen3.8 single-layer FFN/s results.
