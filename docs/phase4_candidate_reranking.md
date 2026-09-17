# Phase 4.4 candidate-pool reranking

Phase 4.4 freezes the existing residual-d32 selector, selects an exact hard
candidate pool, and trains a candidate-only scorer. The public cascade inference
API accepts only the normalized pre-FFN hidden state. Oracle scores, gated
activations, dense FFN outputs, and down-projection weights remain training or
analysis supervision and never enter `CandidateCascade.forward`.

Candidate retention is selected on validation data. The analysis recommends the
smallest pool whose restricted-oracle cosine is within 0.0005 of the best tested
ceiling; test data is reserved for evaluation of the selected configuration.

```bash
python scripts/analyze_candidate_pool.py \
  --results-dir results/predictor_2000 --layer 0 \
  --stage1-run residual_factorized_d32_ce_to_hybrid_rank_rw010 \
  --candidate-retention .55,.60,.65 --final-retention .50 --device cuda
```

After reading the validation recommendation (shown here as `0.60` only as a
command placeholder), run the fixed r16 experiment:

```bash
python scripts/train_candidate_reranker.py \
  --results-dir results/predictor_2000 --layer 0 \
  --stage1-run residual_factorized_d32_ce_to_hybrid_rank_rw010 \
  --candidate-retention .60 --final-retention .50 --rerank-dim 16 \
  --loss distribution_ce --run-name residual_d32_c60_r16_ce \
  --epochs 50 --patience 8 --lr 1e-3 --batch-size 16 --device cuda

python scripts/train_candidate_reranker.py \
  --results-dir results/predictor_2000 --layer 0 \
  --stage1-run residual_factorized_d32_ce_to_hybrid_rank_rw010 \
  --candidate-retention .60 --final-retention .50 --rerank-dim 16 \
  --loss output_hybrid_rank --run-name residual_d32_c60_r16_hybrid_rank \
  --init-checkpoint results/predictor_2000/layer_000/residual_d32_c60_r16_ce/best.pt \
  --ranking-weight .10 --lr 5e-5 --epochs 50 --patience 8 \
  --batch-size 16 --device cuda

python scripts/evaluate_candidate_reranker.py \
  --results-dir results/predictor_2000 --layer 0 \
  --run-name residual_d32_c60_r16_hybrid_rank --device cuda
```

Run r32 only when validation shows that r16 improves Stage 1 while material
restricted-oracle headroom remains. The project has repeatedly inspected its
test set in earlier phases; reported test results must explicitly retain that
research limitation.
