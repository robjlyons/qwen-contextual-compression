param(
  [string]$ResultsDir = "results/predictor_2000",
  [string]$Weights = "results/runtime_weights/layer_000.safetensors",
  [string]$Stage1Run = "residual_factorized_d32_ce_to_hybrid_rank_rw010"
)
python scripts/benchmark_sparse_ffn.py --results-dir $ResultsDir --runtime-weights $Weights --layer 0 --stage1-run $Stage1Run --retention .50 --device cuda --dtype fp16 --warmup 20 --iterations 200
