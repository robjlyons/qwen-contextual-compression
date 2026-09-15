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
