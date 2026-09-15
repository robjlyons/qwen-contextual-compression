# Phase 4.5 boundary-swap reranking

The boundary cascade freezes Stage 1, structurally locks its highest-ranked
neurons, and permits changes only inside a rank-derived boundary band. Its
public inference method accepts normalized pre-FFN `x` only. The learned r16
correction is gated by a scalar `alpha` initialized to zero, so epoch-zero
selection is exactly the Stage-1 Top-50 set.

Select the configuration on validation only:

```bash
python scripts/analyze_boundary_swap.py \
  --results-dir results/predictor_2000 --layer 0 \
  --stage1-run residual_factorized_d32_ce_to_hybrid_rank_rw010 \
  --lock-fractions .40,.45 --candidate-fractions .60,.65 \
  --final-retention .50 --split validation --device cuda
```

The analysis chooses the largest lock and then smallest candidate fraction
within 0.001 cosine of the best validation boundary ceiling. Substitute those
reported fractions below; `.45/.65` are placeholders, not a test-selected
conclusion:

```bash
python scripts/train_boundary_reranker.py \
  --results-dir results/predictor_2000 --layer 0 \
  --stage1-run residual_factorized_d32_ce_to_hybrid_rank_rw010 \
  --lock-retention .45 --candidate-retention .65 --final-retention .50 \
  --rerank-dim 16 --loss output_hybrid --ranking-weight 0 \
  --boundary-init zero_embedding_alpha_one \
  --run-name boundary_l45_c65_r16_output --epochs 50 --patience 8 \
  --lr 5e-5 --batch-size 16 --seed 42 --device cuda

python scripts/evaluate_boundary_reranker.py \
  --results-dir results/predictor_2000 --layer 0 \
  --run-name boundary_l45_c65_r16_output --split test --device cuda
```

The optional `0.02` ranking regularizer should be run only if the output-only
validation result is useful. Earlier phases repeatedly inspected the test set;
that limitation must accompany final test results even though Phase 4.5 model
selection itself uses validation only.
