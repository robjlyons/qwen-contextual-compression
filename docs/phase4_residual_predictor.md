# Phase 4.3 residual-factorized predictor

The residual predictor receives only the externally normalized pre-FFN hidden
state. It applies `Linear(input,d)`, SiLU, a two-layer SiLU residual block,
latent LayerNorm, and `Linear(d,intermediate)`. Training-only FFN supervision
does not enter its `forward` API.

For Qwen's 5120/17408 FFN dimensions, projection MAC accounting is:

| d | Parameters | MACs/token | Dense-FFN MAC fraction |
|---:|-----------:|-----------:|-----------------------:|
| 32 | 740,480 | 722,944 | 0.2704% |
| 64 | 1,467,648 | 1,449,984 | 0.5423% |
| 96 | 2,198,912 | 2,181,120 | 0.8157% |

The primary d32 and d64 experiments use the existing layer-0 targets and split:

```bash
for d in 32 64; do
  python scripts/train_predictor.py \
    --results-dir results/predictor_2000 --layer 0 \
    --model residual_factorized --latent-dim "$d" --loss distribution_ce \
    --run-name "residual_factorized_d${d}_distribution_ce" \
    --seed 42 --epochs 100 --batch-size 16 --lr 1e-3 --device cuda

  python scripts/train_predictor.py \
    --results-dir results/predictor_2000 --layer 0 \
    --model residual_factorized --latent-dim "$d" --loss output_hybrid_rank \
    --run-name "residual_factorized_d${d}_ce_to_hybrid_rank_rw010" \
    --init-checkpoint "results/predictor_2000/layer_000/residual_factorized_d${d}_distribution_ce/best.pt" \
    --train-retention .5 --ranking-weight .10 --freeze-encoder-epochs 0 \
    --lr 5e-5 --epochs 50 --patience 8 --batch-size 16 --device cuda

  python scripts/evaluate_predictor.py \
    --results-dir results/predictor_2000 --layer 0 \
    --model residual_factorized --latent-dim "$d" \
    --run-name "residual_factorized_d${d}_ce_to_hybrid_rank_rw010" \
    --retention .3,.4,.5,.6,.75 --device cuda
done
```

The d96 experiment should only be run if d64 materially improves over the
factorized d64 reference. `scripts/compare_predictors.py --results-dir
results/predictor_2000 --layer 0 --retention .5` produces the compact Phase 4.3
quality/compute table after evaluation.
